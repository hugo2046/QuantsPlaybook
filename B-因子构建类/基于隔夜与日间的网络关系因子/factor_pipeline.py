"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-11-04 22:36:05
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-11-04 22:36:05
Description: 领先-滞后因子计算流水线

Status: ACTIVE - 当前使用的主入口
Usage: 推荐用于新项目开发和生产环境
Features:
- 简化的流水线接口
- 优化的性能实现
- 集成最新的GPU加速算法
- 完整的错误处理

Quick Start:
    from factor_pipeline import FactorPipeline
    pipeline = FactorPipeline(codes="ashares", start_dt="2020-01-01", end_dt="2025-10-27")
    final_factor_df = pipeline.run()
"""

from typing import List, Tuple, Union,Optional,Dict


import numpy as np
import pandas as pd
from dlesc_clustering import DLESCClustering
from factor_computation import LeadLagFactorCalculator
from lead_lag_network import _compute_pearson_matrix, _compute_spearman_matrix
from qlib_data_provider import QlibDataProvider
from tqdm import tqdm

try:
    from DataFeed.dataserver.apis.base import get_trade_days
except ImportError:
    from qlib_data_provider import get_trade_days
from utils import sliding_window


class FactorPipeline:
    """
    一个完整的领先-滞后因子计算流水线。

    该类封装了从数据获取到因子计算的整个流程，包括：
    1. 使用QlibDataProvider获取收益率数据。
    2. 对数据进行滑动窗口处理。
    3. 迭代每个窗口，执行d-LE-SC聚类。
    4. 【新】委托LeadLagFactorCalculator计算得分和信号。
    5. 根据信号选择多空股票。
    6. 生成最终的因子DataFrame。
    """

    def __init__(
        self,
        codes: Union[List[str], str],
        start_dt: str,
        end_dt: str,
        window: int = 60,
        network_type: str = "overnight_lead_daytime",
        correlation_method: str = "pearson",
        n_iterations: int = 20,
        random_state: int = 42,
        lead_percentile: float = 0.5,
        top_percentile: float = 0.4,
        bottom_percentile: float = 0.2,
    ):
        """
        初始化因子计算流水线。
        """
        self.codes = codes
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.window = window
        self.network_type = network_type
        self.correlation_method = self._validate_correlation_method(correlation_method)

        self.long_df: Optional[pd.DataFrame] = None
        self.short_df: Optional[pd.DataFrame] = None
        
        # 初始化聚类模型
        self.clustering_model = DLESCClustering(
            n_iterations=n_iterations, random_state=random_state
        )

        # 初始化因子计算器
        self.factor_calculator = LeadLagFactorCalculator(
            lead_percentile=lead_percentile,
            top_percentile=top_percentile,
            bottom_percentile=bottom_percentile,
        )

        # 选择相关性计算函数
        self.corr_func = (
            _compute_pearson_matrix
            if self.correlation_method == "pearson"
            else _compute_spearman_matrix
        )

    def _prepare_data(self) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.Index]:
        """获取并预处理数据，创建滑动窗口。"""
        begin_dt = get_trade_days(end_date=self.start_dt, count=self.window)[
            0
        ].strftime("%Y-%m-%d")
        provider = QlibDataProvider(self.codes, begin_dt, self.end_dt)

        if self.network_type == "overnight_lead_daytime":
            lead_returns: pd.DataFrame = provider.overnight_return_df.fillna(0)
            lag_returns: pd.DataFrame = provider.daytime_return_df.fillna(0)
        elif self.network_type == "daytime_lead_overnight":
            lead_returns: pd.DataFrame = provider.daytime_return_df.shift(1).fillna(0)
            lag_returns: pd.DataFrame = provider.overnight_return_df.fillna(0)
        elif self.network_type == "preclose_lead_close":
            lead_returns: pd.DataFrame = provider.daily_return_df.shift(1).fillna(0)
            lag_returns: pd.DataFrame = provider.daily_return_df.fillna(0)
        else:
            raise ValueError(f"不支持的network_type: {self.network_type}")

        lead_returns, lag_returns = lead_returns.align(lag_returns, join="inner")

        features_arr = np.stack([lead_returns.values, lag_returns.values], axis=1)

        date_index = lead_returns.index[self.window - 1 :]
        stock_index = lead_returns.columns

        sliced_features = sliding_window(features_arr, window=self.window)

        return sliced_features, date_index, stock_index

    def run(self) -> pd.DataFrame:
        """
        执行完整的因子计算流程。

        Returns:
            pd.DataFrame: 计算出的因子值矩阵。
        """
        sliced_features, date_index, stock_index = self._prepare_data()
        # factor_df:pd.DataFrame = pd.DataFrame(0.0, index=date_index, columns=stock_index)
        size: int = len(date_index)
        # 创建一个从股票代码到整数索引的映射
        stock_codes:List[str] = stock_index.tolist()
        code_to_idx: Dict[str, int] = {code: i for i, code in enumerate(stock_codes)}

        # 初始化多空收益矩阵
        long_ret_mat:np.ndarray = np.zeros((size, len(stock_index)), dtype=np.float64)
        short_ret_mat:np.ndarray = np.zeros((size, len(stock_index)), dtype=np.float64)
        
        for day_idx, window_data in enumerate(
            tqdm(sliced_features, total=size, desc="Calculating Factor")
        ):
            slice_lead_arr = window_data[:, 0, :]
            slice_lag_arr = window_data[:, 1, :]

            current_returns = slice_lead_arr[-1]

            # 1. 计算相关性矩阵
            M = self.corr_func(slice_lead_arr, slice_lag_arr)

            # 2. 执行聚类
            cluster_result = self.clustering_model.fit_single(M)
            A = np.abs(M)

            # 3. 计算得分
            scores:Dict[str,Dict[int,float]] = self.factor_calculator.compute_lead_lag_scores(
                A, cluster_result, M, stock_codes
            )

            # 4. 生成信号
            signal:float = self.factor_calculator.generate_trading_signal(
                scores["lead_scores"], current_returns, stock_codes
            )

            # 5. 选择多空股票
            long_stock_codes, short_stock_codes = (
                self.factor_calculator.select_top_and_bottom_stocks(
                    scores["lag_scores"], signal
                )
            )
            # 6. 填充因子值（收益率）
            if long_stock_codes:
                long_indices = [
                    code_to_idx[code]
                    for code in long_stock_codes
                    if code in code_to_idx
                ]
                if long_indices:
                    long_ret_mat[day_idx, long_indices] = current_returns[long_indices]

            if short_stock_codes:
                short_indices = [
                    code_to_idx[code]
                    for code in short_stock_codes
                    if code in code_to_idx
                ]
                if short_indices:
                    short_ret_mat[day_idx, short_indices] = current_returns[short_indices]
                    
            # 保存为 DataFrame
            self.long_df = pd.DataFrame(long_ret_mat, index=date_index, columns=stock_index)
            self.short_df = pd.DataFrame(short_ret_mat, index=date_index, columns=stock_index)

        return self.long_df

    def _validate_correlation_method(self, method: str) -> str:
        """验证相关性计算方法的有效性。"""
        supported_methods = ["pearson", "spearman"]
        method_lower = method.lower()
        if method_lower not in supported_methods:
            raise ValueError(
                f"不支持的相关性方法: {method}. " f"支持的方法: {supported_methods}"
            )
        return method_lower


if __name__ == "__main__":
    # 使用示例
    pipeline = FactorPipeline(
        codes="ashares",
        start_dt="2020-01-01",
        end_dt="2025-10-27",
        window=60,
        top_percentile=0.4,
        bottom_percentile=0.2,
        lead_percentile=0.5,
    )

    final_factor_df = pipeline.run()

    print("因子计算完成，结果预览：")
    print(final_factor_df.head())

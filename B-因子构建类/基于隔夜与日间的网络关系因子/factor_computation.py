"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-11-04 22:36:05
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-11-04 22:36:05
Description: 特征计算模块

本模块包含了基于d-LE-SC算法的领先-滞后网络因子计算的核心逻辑。
主要功能包括：
- 领先-滞后得分计算
- 交易信号生成
- 多空股票选择
- 因子值计算

对应论文4.1节的投资组合构建策略。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeadLagFactorCalculator:
    """
    领先-滞后因子聚类

    基于d-LE-SC聚类结果，计算领先-滞后网络的交易因子。
    实现论文4.1节中的因子计算逻辑。
    """

    def __init__(self,
                 lead_percentile: float = 0.5,
                 top_percentile: float = 0.2,
                 bottom_percentile: float = 0.2):
        """
        初始化因子计算器

        Args:
            lead_percentile (float): 选择领先股票的百分比阈值，默认0.5
            top_percentile (float): 做多股票比例，默认0.2
            bottom_percentile (float): 做空股票比例，默认0.2
        """
        self.lead_percentile = lead_percentile
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile

        logger.info(f"初始化LeadLagFactorCalculator，lead_percentile={lead_percentile}, "
                   f"top_percentile={top_percentile}, bottom_percentile={bottom_percentile}")

    def compute_lead_lag_scores(
        self,
        adjacency_matrix: np.ndarray,
        clustering_results: Dict[str, np.ndarray],
        signed_matrix: np.ndarray,
        stock_codes: List[str]
    ) -> Dict[str, Dict[int, float]]:
        """
        计算领先-滞后得分

        对应论文中的得分计算逻辑，基于聚类结果计算领导者和滞后者股票的得分。

        Args:
            adjacency_matrix (np.ndarray): 形状为(m,m)，绝对值邻接矩阵，表示股票间相关性强度
            clustering_results (Dict[str, np.ndarray]): 包含'lead_cluster'和'lag_cluster'的字典
            signed_matrix (np.ndarray): 形状为(m,m)，带符号的领先-滞后矩阵，包含方向信息
            stock_codes (List[str]): 股票代码列表，与矩阵行列对应

        Returns:
            Dict[str, Dict[int, float]]: 包含'lead_scores'和'lag_scores'的字典，
                key为股票代码，value为得分

        Examples:
            >>> calculator = LeadLagFactorCalculator()
            >>> scores = calculator.compute_lead_lag_scores(
            ...     adjacency_matrix, clustering_results, signed_matrix, stock_codes
            ... )
            >>> lead_scores = scores['lead_scores']
        """
        lead_cluster = clustering_results["lead_cluster"]
        lag_cluster = clustering_results["lag_cluster"]

        # 计算领先得分
        lead_adjacency_values = adjacency_matrix[lead_cluster]
        lead_scores_array = np.sum(lead_adjacency_values, axis=1)
        lead_scores:Dict[int,float] = {
            stock_codes[idx]: float(lead_scores_array[i])
            for i, idx in enumerate(lead_cluster)
        }

        # 计算滞后得分
        lag_submatrix = signed_matrix[np.ix_(lead_cluster, lag_cluster)]
        lag_scores_array = np.sum(lag_submatrix, axis=0)
        lag_scores = {
            stock_codes[idx]: float(lag_scores_array[i])
            for i, idx in enumerate(lag_cluster)
        }

        logger.debug(f"计算领先-滞后得分完成，领先组{len(lead_scores)}只股票，滞后组{len(lag_scores)}只股票")

        return {"lead_scores": lead_scores, "lag_scores": lag_scores}

    def sorted_values(
        self,
        scores: Dict[str, float],
        quantile: float = None,
        reverse: bool = True
    ) -> List[str]:
        """
        按分值排序选择股票

        Args:
            scores (Dict[str, float]): 股票评分字典 {股票代码: 评分}
            quantile (float, optional): 选择比例，默认使用lead_percentile
            reverse (bool): 是否降序排列，默认True

        Returns:
            List[str]: 排序后的股票代码列表

        Examples:
            >>> calculator = LeadLagFactorCalculator()
            >>> scores = {'A': 0.1, 'B': 0.5, 'C': 0.8}
            >>> top_stocks = calculator.sorted_values(scores, 0.5)
        """
        if quantile is None:
            quantile = self.lead_percentile

        sorted_arr = sorted(scores.items(), key=lambda x: x[1], reverse=reverse)
        n_top = max(1, int(len(sorted_arr) * quantile))
        top_stocks = [stock for stock, _ in sorted_arr[:n_top]]

        return top_stocks

    def generate_trading_signal(
        self,
        lead_scores: Dict[str, float],
        daily_returns: np.ndarray,
        stock_codes: List[str]
    ) -> float:
        """
        生成交易信号

        基于领先股票的得分和收益率生成交易信号。
        对应论文公式：S = (1/|C_a|) * sum(returns)

        Args:
            lead_scores (Dict[str, float]): 领先股票的得分字典
            daily_returns (np.ndarray): 形状为(m,)，每日收益率数组
            stock_codes (List[str]): 股票代码列表，与收益率数组对应

        Returns:
            float: 交易信号值，正数表示做多，负数表示做空

        Examples:
            >>> calculator = LeadLagFactorCalculator()
            >>> signal = calculator.generate_trading_signal(lead_scores, returns, codes)
            >>> if signal > 0:
            ...     print("做多信号")
        """
        top_leads = self.sorted_values(lead_scores, self.lead_percentile, True)

        if not daily_returns.ndim == 1:
            raise ValueError(f"daily_returns应为一维数组，当前形状为{daily_returns.shape}")

        # 获取top lead股票对应的收益率
        lead_indices = [stock_codes.index(stock) for stock in top_leads if stock in stock_codes]

        if not lead_indices:
            return 0.0

        lead_returns = daily_returns[lead_indices]
        signal = np.mean(np.nan_to_num(lead_returns))

        logger.debug(f"生成交易信号：{signal:.6f}，基于{len(lead_indices)}只领先股票")

        return signal

    def select_top_and_bottom_stocks(
        self,
        scores: Dict[str, float],
        signal: float
    ) -> Tuple[List[str], List[str]]:
        """
        根据信号方向选择多空股票

        对应论文4.1节：
        "go long the top 20% and short the bottom 20% of stocks in C_lag"

        Args:
            scores (Dict[str, float]): 股票评分字典 {股票代码: 评分}
            signal (float): 交易信号方向

        Returns:
            Tuple[List[str], List[str]]: (做多股票列表, 做空股票列表)

        Examples:
            >>> calculator = LeadLagFactorCalculator()
            >>> scores = {'A': 0.1, 'B': 0.5, 'C': 0.8, 'D': 0.3}
            >>> long, short = calculator.select_top_and_bottom_stocks(scores, 0.1)
        """
        if not scores:
            return [], []

        # 按评分降序排列
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 计算选择数量
        n_top = max(1, int(len(sorted_items) * self.top_percentile))
        n_bottom = max(1, int(len(sorted_items) * self.bottom_percentile))

        if signal > 0:
            # 正信号：做多top，做空bottom
            long_stocks = [stock for stock, _ in sorted_items[:n_top]]
            short_stocks = [stock for stock, _ in sorted_items[-n_bottom:]]
        else:
            # 负信号：做多bottom，做空top
            long_stocks = [stock for stock, _ in sorted_items[-n_bottom:]]
            short_stocks = [stock for stock, _ in sorted_items[:n_top]]

        logger.debug(f"选择股票完成，做多{len(long_stocks)}只，做空{len(short_stocks)}只，信号方向：{signal:.6f}")

        return long_stocks, short_stocks

    def compute_factor_values(
        self,
        adjacency_matrices: np.ndarray,
        signed_matrices: np.ndarray,
        clustering_results: List[Dict[str, Any]],
        returns_matrix: np.ndarray,
        stock_codes: List[str],
        date_index: pd.DatetimeIndex,
        network_type: str = "overnight_lead_daytime"
    ) -> pd.DataFrame:
        """
        计算完整的因子值时间序列

        Args:
            adjacency_matrices (np.ndarray): 形状为(n,m,m)，邻接矩阵序列
            signed_matrices (np.ndarray): 形状为(n,m,m)，带符号矩阵序列
            clustering_results (List[Dict]): 聚类结果列表
            returns_matrix (np.ndarray): 形状为(n,m)，收益率矩阵
            stock_codes (List[str]): 股票代码列表
            date_index (pd.DatetimeIndex): 日期索引
            network_type (str): 网络类型，用于确定使用哪种收益率

        Returns:
            pd.DataFrame: 因子值矩阵，行为日期，列为股票

        Examples:
            >>> calculator = LeadLagFactorCalculator()
            >>> factor_df = calculator.compute_factor_values(
            ...     A, M, results, returns, codes, dates
            ... )
        """
        logger.info(f"开始计算因子值，网络类型：{network_type}")

        factor_df = pd.DataFrame(index=date_index, columns=stock_codes)

        for i, (adj_matrix, signed_matrix, clustering, daily_returns) in enumerate(
            zip(adjacency_matrices, signed_matrices, clustering_results, returns_matrix)
        ):
            try:
                # 计算得分
                scores = self.compute_lead_lag_scores(
                    adj_matrix, clustering, signed_matrix, stock_codes
                )

                # 生成信号
                signal = self.generate_trading_signal(
                    scores["lead_scores"], daily_returns, stock_codes
                )

                # 选择多空股票
                long_stocks, short_stocks = self.select_top_and_bottom_stocks(
                    scores["lag_scores"], signal
                )

                # 计算因子值（A股仅做多，将做空股票忽略）
                if long_stocks:
                    long_indices = [stock_codes.index(stock) for stock in long_stocks if stock in stock_codes]
                    if long_indices:
                        factor_df.iloc[i, long_indices] = daily_returns[long_indices]

                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(adjacency_matrices)} 个交易日")

            except Exception as e:
                logger.error(f"处理第{i}天时出错: {e}")
                continue

        logger.info("因子值计算完成")
        return factor_df


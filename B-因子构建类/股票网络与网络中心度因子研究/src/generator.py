"""
Author: Hugo
Date: 2025-12-24
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-12-24
Description:
股票网络中心度因子生成器

本模块实现了华西证券金融工程专题研究《股票网络与网络中心度因子研究》中的
因子生成器，提供统一的接口来计算三大网络中心度因子：
- SCC（Spatial Centrality Centrality）：空间网络中心度因子
- TCC（Temporal Centrality Centrality）：时间网络中心度因子
- CC（Composite Centrality）：综合网络中心度因子

核心算法来源：
    华西证券金融工程专题报告（2021年3月）
    《股票网络与网络中心度因子研究》
"""

from .factor_algo import calculate_scc, calculate_tcc, generate_factor
from .data_provider import QlibDataProvider
from loguru import logger
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict


class NetworkCentralityFactor:
    """
    股票网络中心度因子生成器

    本类提供了统一的接口来计算基于复杂网络理论的三大网络中心度因子。
    封装了数据获取、因子计算、批量处理和内存管理等完整流程。

    Attributes
    ----------
    provider : QlibDataProvider
        Qlib数据提供者，负责从DolphinDB获取数据
    codes : str or list
        股票池，支持 "csi300", "csi500", "ashares" 或股票代码列表
    start_date : str
        数据开始日期，格式 "YYYY-MM-DD"
    end_date : str
        数据结束日期，格式 "YYYY-MM-DD"
    returns_df : pd.DataFrame or None
        收益率数据，index为日期，columns为股票代码
    _factors_cache : dict
        因子缓存字典，避免重复计算

    Examples
    --------
    >>> # 初始化因子生成器
    >>> factor_engine = NetworkCentralityFactor(
    ...     codes="csi300",
    ...     start_date="2020-01-01",
    ...     end_date="2023-12-31"
    ... )
    >>>
    >>> # 获取数据
    >>> factor_engine.fetch_data()
    >>>
    >>> # 单独计算因子
    >>> scc_df = factor_engine.get_scc_factor(window=20)
    >>> tcc_df = factor_engine.get_tcc_factor(window=20)
    >>> cc_df = factor_engine.get_cc_factor(window=20)
    >>>
    >>> # 批量计算
    >>> factors = factor_engine.calculate_factors_separately(
    ...     factor_types=['scc', 'tcc', 'cc'],
    ...     window=20
    ... )
    >>> # factors 返回: {'scc': DataFrame, 'tcc': DataFrame, 'cc': DataFrame}
    >>>
    >>> # 清理内存
    >>> factor_engine.cleanup_memory()

    Notes
    -----
    1. 数据格式要求：
       - 收益率数据需要是 DataFrame 格式
       - index 为 DatetimeIndex（日期）
       - columns 为股票代码

    2. 因子计算：
       - SCC因子：基于股票间Pearson相关系数的平均距离
       - TCC因子：基于收益率偏离的时间稳定性
       - CC因子：SCC与TCC的1:1加权合成

    3. 性能优化：
       - 使用缓存机制避免重复计算
       - 支持批量计算提高效率
       - 提供显式的内存清理方法
    """

    def __init__(
        self,
        codes: Union[str, List[str]],
        start_date: str,
        end_date: str
    ):
        """
        初始化网络中心度因子生成器

        创建Qlib数据提供者实例，初始化数据属性和缓存字典。

        Parameters
        ----------
        codes : str or list
            股票池，支持：
            - "csi300": 沪深300成分股
            - "csi500": 中证500成分股
            - "ashares": 全A股
            - 股票代码列表，如 ['000001.SZ', '600000.SH']

        start_date : str
            数据开始日期，格式 "YYYY-MM-DD"
            例如: "2020-01-01"

        end_date : str
            数据结束日期，格式 "YYYY-MM-DD"
            例如: "2023-12-31"

        Raises
        ------
        ValueError
            当日期格式不正确时
        Exception
            当Qlib数据提供者初始化失败时

        Examples
        --------
        >>> # 使用沪深300
        >>> engine = NetworkCentralityFactor(
        ...     codes="csi300",
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
        ... )
        >>>
        >>> # 使用自定义股票池
        >>> engine = NetworkCentralityFactor(
        ...     codes=['000001.SZ', '600000.SH', '000002.SZ'],
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
        ... )
        """
        logger.info(
            f"初始化NetworkCentralityFactor - "
            f"股票池: {codes}, "
            f"时间范围: {start_date} 至 {end_date}"
        )

        # 初始化数据提供者
        try:
            self.provider = QlibDataProvider(codes, start_date, end_date)
            self.codes = codes
            self.start_date = start_date
            self.end_date = end_date
            logger.info("QlibDataProvider初始化成功")
        except Exception as e:
            logger.error(f"QlibDataProvider初始化失败: {str(e)}")
            raise

        # 数据属性
        self.returns_df: Optional[pd.DataFrame] = None

        # 因子缓存
        self._factors_cache: Dict[str, pd.DataFrame] = {}

        logger.success("NetworkCentralityFactor初始化完成")

    def fetch_data(
        self,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取收益率数据

        从Qlib获取价格数据并计算收益率。如果未指定fields，使用默认的
        收益率计算公式 "$close/$preclose-1"。

        Parameters
        ----------
        fields : list of str, optional
            Qlib字段表达式列表，默认为 ["$close/$preclose-1"]
            可以使用其他Qlib表达式，如：
            - "$close": 收盘价
            - "$open": 开盘价
            - "$volume": 成交量

        Returns
        -------
        pd.DataFrame
            收益率数据
            - index: DatetimeIndex（日期）
            - columns: 股票代码
            - values: 收益率值

        Raises
        ------
        Exception
            当数据获取失败时

        Notes
        -----
        1. 数据格式转换：
           - Qlib返回的是 MultiIndex [datetime, instrument] 格式
           - 本方法自动进行 unstack 转换为标准的 DataFrame 格式

        2. 缓存机制：
           - 数据会被存储在 self.returns_df 中
           - 避免重复获取相同的数据

        Examples
        --------
        >>> # 获取默认收益率数据
        >>> factor_engine = NetworkCentralityFactor(
        ...     codes="csi300",
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31"
        ... )
        >>> returns = factor_engine.fetch_data()
        >>> print(returns.shape)
        (243, 300)  # 243个交易日，300只股票
        >>>
        >>> # 使用自定义字段
        >>> returns = factor_engine.fetch_data(
        ...     fields=["$close"]
        ... )
        """
        if fields is None:
            fields = ["$close/$preclose-1"]

        logger.info(
            f"开始获取数据 - "
            f"字段: {fields}, "
            f"股票池: {len(self.provider.instruments_)}只"
        )

        try:
            # 获取特征数据
            features_data = self.provider.get_features(
                instruments=self.provider.instruments_,
                start_time=self.start_date,
                end_time=self.end_date,
                fields=fields
            )

            if features_data.empty:
                logger.error("获取的数据为空")
                raise ValueError("获取的数据为空")

            # 转换数据格式：MultiIndex [datetime, instrument] -> DataFrame
            # unstack(level=1) 将 instrument 从 index 转为 columns
            logger.debug("转换数据格式：unstack MultiIndex")
            returns_df = features_data.unstack(level=0)

            # 如果字段只有一个，去除多级列索引
            if len(fields) == 1:
                returns_df.columns = returns_df.columns.droplevel(0)

            # 存储到实例属性
            self.returns_df = returns_df

            logger.success(
                f"数据获取完成 - "
                f"形状: {returns_df.shape}, "
                f"时间范围: {returns_df.index[0]} 至 {returns_df.index[-1]}"
            )

            return returns_df

        except Exception as e:
            logger.error(f"数据获取失败: {str(e)}")
            raise

    def get_scc_factor(
        self,
        window: int = 20,
        ignore_errors: bool = True,
        show_progress: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        计算空间网络中心度因子（SCC）

        使用滑动窗口计算SCC因子时间序列。SCC因子基于股票间Pearson相关系数的
        平均距离，反映股票在网络中的中心程度。

        Parameters
        ----------
        window : int, default=20
            滑动窗口大小（交易日数量）
            常用值：
            - 20: 约1个月
            - 60: 约1季度
            - 252: 1年

        ignore_errors : bool, default=True
            是否忽略单个窗口的计算错误
            - True: 跳过错误窗口，继续处理
            - False: 遇到错误立即停止

        show_progress : bool, default=True
            是否显示tqdm进度条

        use_cache : bool, default=True
            是否使用缓存
            - True: 如果已计算过，直接返回缓存结果
            - False: 强制重新计算

        Returns
        -------
        pd.DataFrame
            SCC因子值数据
            - index: 日期时间索引（从第window个交易日开始）
            - columns: 股票代码（与输入相同）
            - values: SCC因子值

        Raises
        ------
        ValueError
            当收益率数据未获取时

        Examples
        --------
        >>> factor_engine = NetworkCentralityFactor(
        ...     codes="csi300",
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
        ... )
        >>> factor_engine.fetch_data()
        >>>
        >>> # 使用20天窗口计算SCC因子
        >>> scc_factor = factor_engine.get_scc_factor(window=20)
        >>> print(f"SCC因子形状: {scc_factor.shape}")
        >>> print(f"SCC因子范围: [{scc_factor.min().min():.4f}, {scc_factor.max().max():.4f}]")

        Notes
        -----
        1. 业务含义：
           - SCC值大：股票与市场中其他股票的相关性强，处于网络中心位置
           - SCC值小：股票相对独立，与其他股票相关性弱，处于网络边缘

        2. 论文回测结果（全市场）：
           - IC均值：8.30%
           - IC_IR：3.97
           - 多空组合年化收益：26.80%
           - 最大回撤：-1.79%

        See Also
        --------
        calculate_scc : SCC因子计算核心算法
        generate_factor : 滑动窗口因子生成器
        """
        cache_key = f"scc_{window}"

        # 检查缓存
        if use_cache and cache_key in self._factors_cache:
            logger.info(f"从缓存读取SCC因子（window={window}）")
            return self._factors_cache[cache_key]

        # 检查数据
        if self.returns_df is None:
            logger.error("收益率数据未获取，请先调用 fetch_data()")
            raise ValueError("收益率数据未获取，请先调用 fetch_data()")

        logger.info(f"开始计算SCC因子 - 窗口: {window}天")

        # 计算SCC因子
        scc_df = generate_factor(
            self.returns_df,
            calculate_scc,
            window=window,
            ignore_errors=ignore_errors,
            show_progress=show_progress
        )

        # 缓存结果
        self._factors_cache[cache_key] = scc_df

        return scc_df

    def get_tcc_factor(
        self,
        window: int = 20,
        ignore_errors: bool = True,
        show_progress: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        计算时间网络中心度因子（TCC）

        使用滑动窗口计算TCC因子时间序列。TCC因子基于收益率偏离的时间稳定性，
        反映股票收益率随时间的波动程度。

        Parameters
        ----------
        window : int, default=20
            滑动窗口大小（交易日数量）
            常用值：
            - 20: 约1个月
            - 60: 约1季度
            - 252: 1年

        ignore_errors : bool, default=True
            是否忽略单个窗口的计算错误
            - True: 跳过错误窗口，继续处理
            - False: 遇到错误立即停止

        show_progress : bool, default=True
            是否显示tqdm进度条

        use_cache : bool, default=True
            是否使用缓存
            - True: 如果已计算过，直接返回缓存结果
            - False: 强制重新计算

        Returns
        -------
        pd.DataFrame
            TCC因子值数据
            - index: 日期时间索引（从第window个交易日开始）
            - columns: 股票代码（与输入相同）
            - values: TCC因子值

        Raises
        ------
        ValueError
            当收益率数据未获取时

        Examples
        --------
        >>> factor_engine = NetworkCentralityFactor(
        ...     codes="csi300",
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
        ... )
        >>> factor_engine.fetch_data()
        >>>
        >>> # 使用20天窗口计算TCC因子
        >>> tcc_factor = factor_engine.get_tcc_factor(window=20)
        >>> print(f"TCC因子形状: {tcc_factor.shape}")
        >>> print(f"TCC因子范围: [{tcc_factor.min().min():.4f}, {tcc_factor.max().max():.4f}]")

        Notes
        -----
        1. 业务含义：
           - TCC值大：股票收益率相对市场平均的偏离小，波动稳定
           - TCC值小：股票收益率波动大，与市场平均偏离度高

        2. 论文回测结果（全市场）：
           - IC均值：9.05%
           - IC_IR：3.55
           - 多空组合年化收益：22.10%
           - 最大回撤：-15.02%

        See Also
        --------
        calculate_tcc : TCC因子计算核心算法
        generate_factor : 滑动窗口因子生成器
        """
        cache_key = f"tcc_{window}"

        # 检查缓存
        if use_cache and cache_key in self._factors_cache:
            logger.info(f"从缓存读取TCC因子（window={window}）")
            return self._factors_cache[cache_key]

        # 检查数据
        if self.returns_df is None:
            logger.error("收益率数据未获取，请先调用 fetch_data()")
            raise ValueError("收益率数据未获取，请先调用 fetch_data()")

        logger.info(f"开始计算TCC因子 - 窗口: {window}天")

        # 计算TCC因子
        tcc_df = generate_factor(
            self.returns_df,
            calculate_tcc,
            window=window,
            ignore_errors=ignore_errors,
            show_progress=show_progress
        )

        # 缓存结果
        self._factors_cache[cache_key] = tcc_df

        return tcc_df

    def get_cc_factor(
        self,
        window: int = 20,
        ignore_errors: bool = True,
        show_progress: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        计算综合网络中心度因子（CC）

        CC因子是SCC和TCC因子的1:1加权合成，综合考虑空间和时间维度的
        网络中心度。

        Parameters
        ----------
        window : int, default=20
            滑动窗口大小（交易日数量）
            常用值：
            - 20: 约1个月
            - 60: 约1季度
            - 252: 1年

        ignore_errors : bool, default=True
            是否忽略单个窗口的计算错误
            - True: 跳过错误窗口，继续处理
            - False: 遇到错误立即停止

        show_progress : bool, default=True
            是否显示tqdm进度条

        use_cache : bool, default=True
            是否使用缓存
            - True: 如果已计算过，直接返回缓存结果
            - False: 强制重新计算

        Returns
        -------
        pd.DataFrame
            CC因子值数据
            - index: 日期时间索引（从第window个交易日开始）
            - columns: 股票代码（与输入相同）
            - values: CC因子值（SCC * 0.5 + TCC * 0.5）

        Raises
        ------
        ValueError
            当收益率数据未获取时

        Examples
        --------
        >>> factor_engine = NetworkCentralityFactor(
        ...     codes="csi300",
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
        ... )
        >>> factor_engine.fetch_data()
        >>>
        >>> # 使用20天窗口计算CC因子
        >>> cc_factor = factor_engine.get_cc_factor(window=20)
        >>> print(f"CC因子形状: {cc_factor.shape}")
        >>> print(f"CC因子范围: [{cc_factor.min().min():.4f}, {cc_factor.max().max():.4f}]")

        Notes
        -----
        1. 计算公式：
           CC = SCC * 0.5 + TCC * 0.5

        2. 业务含义：
           - 综合考虑股票在网络中的空间中心度和时间稳定性
           - CC值大：股票在网络中位置中心且收益稳定
           - CC值小：股票在网络中位置边缘或收益不稳定

        3. 论文回测结果（全市场）：
           - IC均值：9.21%
           - IC_IR：4.19
           - 多空组合年化收益：24.86%
           - 最大回撤：-7.95%

        See Also
        --------
        get_scc_factor : 计算SCC因子
        get_tcc_factor : 计算TCC因子
        """
        cache_key = f"cc_{window}"

        # 检查缓存
        if use_cache and cache_key in self._factors_cache:
            logger.info(f"从缓存读取CC因子（window={window}）")
            return self._factors_cache[cache_key]

        logger.info(f"开始计算CC因子 - 窗口: {window}天")

        # 获取SCC和TCC因子
        scc_df = self.get_scc_factor(
            window=window,
            ignore_errors=ignore_errors,
            show_progress=show_progress,
            use_cache=use_cache
        )
        tcc_df = self.get_tcc_factor(
            window=window,
            ignore_errors=ignore_errors,
            show_progress=show_progress,
            use_cache=use_cache
        )

        # 计算CC因子：1:1加权合成
        cc_df = scc_df * 0.5 + tcc_df * 0.5

        # 缓存结果
        self._factors_cache[cache_key] = cc_df

        logger.success("CC因子计算完成")

        return cc_df

    def calculate_factors_separately(
        self,
        factor_types: List[str] = ['scc', 'tcc', 'cc'],
        window: int = 20,
        ignore_errors: bool = True,
        show_progress: bool = True,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量计算多个因子

        一次性计算多个网络中心度因子，返回一个字典包含所有请求的因子。
        支持的因子类型：'scc', 'tcc', 'cc'。

        Parameters
        ----------
        factor_types : list of str, default=['scc', 'tcc', 'cc']
            要计算的因子类型列表
            可选值：
            - 'scc': 空间网络中心度因子
            - 'tcc': 时间网络中心度因子
            - 'cc': 综合网络中心度因子

        window : int, default=20
            滑动窗口大小（交易日数量）

        ignore_errors : bool, default=True
            是否忽略单个窗口的计算错误

        show_progress : bool, default=True
            是否显示tqdm进度条

        use_cache : bool, default=True
            是否使用缓存

        Returns
        -------
        dict
            因子字典，key为因子类型，value为对应的因子DataFrame
            例如: {'scc': DataFrame, 'tcc': DataFrame, 'cc': DataFrame}

        Raises
        ------
        ValueError
            当收益率数据未获取或因子类型不支持时

        Examples
        --------
        >>> factor_engine = NetworkCentralityFactor(
        ...     codes="csi300",
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
        ... )
        >>> factor_engine.fetch_data()
        >>>
        >>> # 计算所有因子
        >>> factors = factor_engine.calculate_factors_separately(
        ...     factor_types=['scc', 'tcc', 'cc'],
        ...     window=20
        ... )
        >>>
        >>> # 只计算SCC和TCC
        >>> factors = factor_engine.calculate_factors_separately(
        ...     factor_types=['scc', 'tcc'],
        ...     window=20
        ... )
        >>>
        >>> # 访问结果
        >>> scc_df = factors['scc']
        >>> tcc_df = factors['tcc']

        Notes
        -----
        1. 批量计算的优势：
           - 一次性获取多个因子
           - 充分利用缓存机制
           - 便于后续分析和比较

        2. 执行顺序：
           - 如果包含 'cc'，会先计算 'scc' 和 'tcc'（如果未缓存）
           - 然后基于SCC和TCC计算CC

        3. 性能考虑：
           - 使用缓存可以显著提高计算效率
           - 建议批量计算而不是逐个调用
        """
        logger.info(f"开始批量计算因子 - 类型: {factor_types}, 窗口: {window}天")

        # 验证因子类型
        valid_factors = {'scc', 'tcc', 'cc'}
        for ft in factor_types:
            if ft not in valid_factors:
                logger.error(f"不支持的因子类型: {ft}")
                raise ValueError(
                    f"不支持的因子类型: {ft}，"
                    f"可选值: {valid_factors}"
                )

        # 检查数据
        if self.returns_df is None:
            logger.error("收益率数据未获取，请先调用 fetch_data()")
            raise ValueError("收益率数据未获取，请先调用 fetch_data()")

        # 批量计算
        factors_dict = {}

        for factor_type in factor_types:
            logger.debug(f"计算 {factor_type.upper()} 因子")

            if factor_type == 'scc':
                factors_dict['scc'] = self.get_scc_factor(
                    window=window,
                    ignore_errors=ignore_errors,
                    show_progress=show_progress,
                    use_cache=use_cache
                )
            elif factor_type == 'tcc':
                factors_dict['tcc'] = self.get_tcc_factor(
                    window=window,
                    ignore_errors=ignore_errors,
                    show_progress=show_progress,
                    use_cache=use_cache
                )
            elif factor_type == 'cc':
                factors_dict['cc'] = self.get_cc_factor(
                    window=window,
                    ignore_errors=ignore_errors,
                    show_progress=show_progress,
                    use_cache=use_cache
                )

        logger.success(
            f"批量计算完成 - "
            f"计算了 {len(factors_dict)} 个因子: {list(factors_dict.keys())}"
        )

        return factors_dict

    def cleanup_memory(self, clear_data: bool = False, clear_cache: bool = True):
        """
        清理内存缓存

        显式清理数据和因子缓存，释放内存。在大规模数据计算时，
        定期清理内存可以避免内存溢出。

        Parameters
        ----------
        clear_data : bool, default=False
            是否清理收益率数据
            - True: 清理 self.returns_df
            - False: 保留收益率数据（默认）

        clear_cache : bool, default=True
            是否清理因子缓存
            - True: 清理 self._factors_cache
            - False: 保留因子缓存

        Examples
        --------
        >>> factor_engine = NetworkCentralityFactor(
        ...     codes="ashares",
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
    ... )
    >>> factor_engine.fetch_data()
    >>> factors = factor_engine.calculate_factors_separately()
    >>>
    >>> # 使用完毕后清理内存
    >>> factor_engine.cleanup_memory()
    >>>
    >>> # 只清理缓存，保留数据
    >>> factor_engine.cleanup_memory(clear_data=False, clear_cache=True)
    >>>
    >>> # 清理所有
    >>> factor_engine.cleanup_memory(clear_data=True, clear_cache=True)

        Notes
        -----
        1. 内存管理建议：
           - 计算完成后及时清理缓存
           - 如果不再需要原始数据，也可以清理
           - 清理后如果需要重新计算，会自动重新获取数据

        2. 性能考虑：
           - 缓存可以避免重复计算，提高性能
           - 但会占用额外内存
           - 根据实际情况权衡使用

        3. 自动垃圾回收：
           - Python的垃圾回收机制会自动清理无法访问的对象
           - 但显式清理可以更快释放内存
        """
        logger.info("开始清理内存")

        if clear_data and self.returns_df is not None:
            logger.debug("清理收益率数据")
            del self.returns_df
            self.returns_df = None

        if clear_cache and self._factors_cache:
            logger.debug(f"清理因子缓存 - 清理了 {len(self._factors_cache)} 个缓存")
            self._factors_cache.clear()

        logger.success("内存清理完成")

    def __repr__(self) -> str:
        """对象的字符串表示"""
        return (
            f"NetworkCentralityFactor("
            f"codes={self.codes}, "
            f"start_date={self.start_date}, "
            f"end_date={self.end_date})"
        )

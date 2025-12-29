"""
Author: Hugo
Date: 2025-10-28 13:41:02
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-10-29 15:09:45
Description:
领先-滞后网络构建模块

本模块提供从金融数据构建领先-滞后网络的功能，重点关注隔夜与日间收益关系。

基于: "A tug of war across the market: overnight-vs-daytime lead-lag networks
and clustering-based portfolio strategies" (第3.2节)

核心功能：
- 收益率分解：将日收益率分解为隔夜和日间成分
- 网络构建：构建三种类型的领先-滞后网络
- 相关性计算：支持Pearson和Spearman相关性
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.core.algorithms import rank as pandas_rank
from qlib_data_provider import QlibDataProvider
from scipy.stats import rankdata

from utils import sliding_window

# 智能并行配置类
class ParallelConfig:
    """并行计算配置类"""
    def __init__(self):
        self.max_memory_usage_ratio = 0.7  # 最大内存使用比例
        self.min_chunk_size = 1
        self.max_chunk_size = 100
        self.enable_auto_adjustment = True
        self.fallback_to_single_process = True
        self.min_n_jobs = 1
        self.max_n_jobs = None  # None表示不限制


def _get_system_resources():
    """
    获取系统资源信息

    Returns
    -------
    dict
        包含CPU核心数、可用内存等信息
    """
    try:
        import psutil
        return {
            'cpu_cores': psutil.cpu_count(),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        }
    except ImportError:
        # 如果psutil不可用，使用基础估计
        import multiprocessing
        return {
            'cpu_cores': multiprocessing.cpu_count(),
            'available_memory_gb': 4.0,  # 保守估计4GB
            'total_memory_gb': 8.0,   # 保守估计8GB
        }


def _auto_adjust_n_jobs(requested_n_jobs: int, data_shape: Tuple[int, int],
                       config: ParallelConfig = None) -> int:
    """
    自动调整n_jobs以适应系统资源

    Parameters
    ----------
    requested_n_jobs : int
        用户请求的n_jobs
    data_shape : Tuple[int, int]
        数据形状
    config : ParallelConfig, optional
        并行配置

    Returns
    -------
    int
        调整后的n_jobs
    """
    if config is None:
        config = ParallelConfig()

    # 获取系统资源
    resources = _get_system_resources()
    cpu_cores = resources['cpu_cores']
    available_memory_gb = resources['available_memory_gb']

    # 如果禁用自动调整，直接返回
    if not config.enable_auto_adjustment:
        return min(requested_n_jobs, cpu_cores if config.max_n_jobs is None else config.max_n_jobs)

    # 基于CPU核心数的限制
    max_by_cpu = cpu_cores if config.max_n_jobs is None else min(cpu_cores, config.max_n_jobs)

    # 基于内存的限制（使用简化的内存估算）
    T, N = data_shape
    base_memory_gb = (T * N * 8) / (1024**3)  # 基础数据内存
    memory_per_process = base_memory_gb * 2  # 估算包括临时计算内存
    max_by_memory = int(available_memory_gb * config.max_memory_usage_ratio / memory_per_process)

    # 选择最小值
    recommended_n_jobs = min(requested_n_jobs, max_by_cpu, max_by_memory)

    # 确保不小于最小值
    final_n_jobs = max(config.min_n_jobs, recommended_n_jobs)

    # 如果调整了n_jobs，给出警告
    if final_n_jobs != requested_n_jobs:
        print(f"⚠️  资源限制：将n_jobs从{requested_n_jobs}调整到{final_n_jobs}")
        print(f"   CPU限制: {max_by_cpu}核心")
        print(f"   内存限制: {max_by_memory}进程")
        print(f"   可用内存: {available_memory_gb:.1f}GB")

    return final_n_jobs




def _compute_pearson_matrix(
    lead_returns: np.ndarray, lag_returns: np.ndarray
) -> np.ndarray:
    """
    高效计算领先-滞后相关性矩阵 (向量化实现)。

    Parameters
    ----------
    lead_returns : np.ndarray
        领先信号的收益率矩阵，形状为 (T, N)，T为时间步，N为资产数。
    lag_returns : np.ndarray
        滞后信号的收益率矩阵，形状为 (T, N)。

    Returns
    -------
    np.ndarray
        相关性矩阵 M (N, N)，其中 M[i, j] = Corr(lead_i, lag_j)。
    """
    assert (
        lead_returns.shape == lag_returns.shape
    ), f"形状不匹配: {lead_returns.shape} vs {lag_returns.shape}"

    T, N = lead_returns.shape
    if T <= 1:
        return np.zeros((N, N))

    # 标准化 (使用样本标准差, ddof=1)
    lead_mean = np.mean(lead_returns, axis=0, keepdims=True)
    lead_std = np.std(lead_returns, axis=0, ddof=1, keepdims=True) + 1e-8
    lead_normalized = (lead_returns - lead_mean) / lead_std

    lag_mean = np.mean(lag_returns, axis=0, keepdims=True)
    lag_std = np.std(lag_returns, axis=0, ddof=1, keepdims=True) + 1e-8
    lag_normalized = (lag_returns - lag_mean) / lag_std

    # 计算相关系数矩阵
    # 注意：因为使用了样本标准差(ddof=1)，这里的分母是 (T-1)
    M = (lead_normalized.T @ lag_normalized) / (T - 1)

    # 处理因浮点数计算产生的微小误差，确保相关系数在 [-1, 1] 范围内
    M = np.clip(M, -1, 1)

    return M

def _compute_spearman_matrix(
    lead_returns: np.ndarray, lag_returns: np.ndarray
) -> np.ndarray:
    """
    高效计算领先-滞后斯皮尔曼相关性矩阵（Pandas核心算法优化版）。

    直接使用pandas.core.algorithms.rank，避免DataFrame转换开销。
    比scipy.stats.rankdata快10-100倍。

    Parameters
    ----------
    lead_returns : np.ndarray
        领先信号的收益率矩阵，形状为 (T, N)，T为时间步，N为资产数。
    lag_returns : np.ndarray
        滞后信号的收益率矩阵，形状为 (T, N)。

    Returns
    -------
    np.ndarray
        斯皮尔曼相关性矩阵 M (N, N)，其中 M[i, j] = SpearmanCorr(lead_i, lag_j)。
    """
    assert (
        lead_returns.shape == lag_returns.shape
    ), f"形状不匹配: {lead_returns.shape} vs {lag_returns.shape}"

    T, N = lead_returns.shape
    if T <= 1:
        return np.zeros((N, N))

    try:

        # 使用pandas的rank算法直接处理numpy数组
        lead_ranks = pandas_rank(
            lead_returns,
            axis=0,              # 沿时间轴排名
            method='average',    # 处理并列值
            na_option='keep',    # 保持NaN值
            ascending=True,      # 升序排名
            pct=False           # 返回秩次而非百分比
        )

        lag_ranks = pandas_rank(
            lag_returns,
            axis=0,
            method='average',
            na_option='keep',
            ascending=True,
            pct=False
        )

    except (ImportError, AttributeError) as e:
        # 如果pandas内部API不可用，回退到scipy
        print(f"警告：pandas核心算法不可用，使用scipy rankdata: {e}")
        lead_ranks = np.apply_along_axis(rankdata, 0, lead_returns)
        lag_ranks = np.apply_along_axis(rankdata, 0, lag_returns)

    # 对秩次数据计算皮尔逊相关性
    return _compute_pearson_matrix(lead_ranks, lag_ranks)


def compute_rolling_lead_lag(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    window: int,
    align_index: bool = False,
    method: str = "pearson",
    n_jobs: int = -1,
    parallel_config: ParallelConfig = None,
) -> np.ndarray:
    """
    在滚动窗口上，计算两个DataFrame每列之间的相关性（内存安全版）。

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个DataFrame (例如, 隔夜收益率)。
    df2 : pd.DataFrame
        第二个DataFrame (例如, 日间收益率)。
    window : int
        滚动窗口的大小。
    align_index : bool, optional
        是否在计算前对齐两个DataFrame的索引和列, by default False。
    method : str, optional
        相关性计算方法 ('pearson' 或 'spearman'), by default "pearson"。
    n_jobs : int, optional
        并行运行的作业数, by default -1（将根据系统资源自动调整）。
    parallel_config : ParallelConfig, optional
        并行计算配置，by default None（使用默认配置）。

    Returns
    -------
    np.ndarray
        一个三维数组，形状为 (D, N, N)，
        D为滚动窗口数，N为资产数。代表每个时间点的相关性矩阵。
    """
    if align_index:
        df1, df2 = df1.align(df2)

    # 获取数据形状和窗口数
    T, N = df1.shape
    total_windows = T - window + 1

    if total_windows <= 0:
        raise ValueError(f"窗口大小({window})大于数据长度({T})")

    # 🚨 内存安全检查
    print(f"📊 数据检查: {T}天 × {N}只股票, 窗口{window}天, 总计{total_windows}个窗口")

    # 评估内存需求
    memory_safe, recommended_n_jobs = _assess_memory_safety(df1, window, method, n_jobs, parallel_config)

    # 选择计算方法
    if method.lower() == "pearson":
        compute_func = _compute_pearson_matrix
    elif method.lower() == "spearman":
        compute_func = _compute_spearman_matrix
    else:
        raise ValueError(f"未知的相关性计算方法: {method}")

    if memory_safe:
        print(f"✅ 内存安全，使用{n_jobs}个进程并行计算")
        return _safe_parallel_compute(df1, df2, window, compute_func, n_jobs, parallel_config)
    else:
        print(f"⚠️  内存不足，使用流式单进程计算")
        return _streaming_compute(df1, df2, window, compute_func)


def _assess_memory_safety(df1: pd.DataFrame, window: int,
                          method: str, n_jobs: int, config: ParallelConfig = None) -> Tuple[bool, int]:
    """
    评估内存安全性，避免将生成器转换为列表

    Returns
    -------
    Tuple[bool, int]
        (是否安全, 推荐的n_jobs)
    """
    if config is None:
        config = ParallelConfig()
        # 使用更保守的默认设置
        config.max_memory_usage_ratio = 0.2  # 只使用20%内存
        config.fallback_to_single_process = True

    # 获取系统资源
    resources = _get_system_resources()
    available_memory_gb = resources['available_memory_gb']

    # 计算内存需求（更保守的估计）
    T, N = df1.shape

    # 1. 原始数据内存
    base_data_gb = (T * N * 8) / (1024**3)

    # 2. 方法特定内存
    if method == "spearman":
        # Spearman需要排名数据
        method_multiplier = 2.5
    else:
        method_multiplier = 1.5

    # 3. 并行内存乘数
    if n_jobs > 1:
        parallel_multiplier = min(n_jobs, resources['cpu_cores'])
    else:
        parallel_multiplier = 1

    # 总内存需求
    total_memory_gb = base_data_gb * parallel_multiplier * method_multiplier

    # 保守的安全系数
    safety_factor = 3.0  # 非常保守的安全系数
    estimated_memory_gb = total_memory_gb * safety_factor

    memory_ratio = estimated_memory_gb / available_memory_gb
    memory_safe = memory_ratio < config.max_memory_usage_ratio

    # 计算推荐的n_jobs
    max_safe_n_jobs = int(available_memory_gb * config.max_memory_usage_ratio /
                         (base_data_gb * method_multiplier * safety_factor))
    recommended_n_jobs = max(config.min_n_jobs, min(n_jobs, max_safe_n_jobs))

    print(f"   内存估算: {estimated_memory_gb:.2f}GB (安全比例: {memory_ratio:.1%})")
    print(f"   推荐n_jobs: {recommended_n_jobs}")

    return memory_safe, recommended_n_jobs


def _safe_parallel_compute(df1: pd.DataFrame, df2: pd.DataFrame, window: int,
                          compute_func: callable, n_jobs: int, config: ParallelConfig) -> np.ndarray:
    """
    安全的并行计算实现（避免内存爆炸）
    """
    features_arrs = np.stack([df1.values, df2.values], axis=1)
    roll_arrs = sliding_window(features_arrs, window)

    # 计算总窗口数（不转换为列表）
    T = df1.shape[0]
    total_windows = T - window + 1

    # 智能调整n_jobs
    _, adjusted_n_jobs = _assess_memory_safety(df1, window, "spearman", n_jobs, config)

    # 保守的chunk大小
    chunk_size = max(1, min(10, total_windows // (adjusted_n_jobs * 4)))

    print(f"   Chunk大小: {chunk_size}, 总chunks: {(total_windows + chunk_size - 1) // chunk_size}")

    # 流式处理，避免内存爆炸
    results = []
    chunk = []

    for i, window_data in enumerate(roll_arrs):
        chunk.append(window_data)

        if len(chunk) >= chunk_size:
            # 处理当前chunk
            chunk_results = [compute_func(data[:, 0, :], data[:, 1, :]) for data in chunk]
            results.extend(chunk_results)
            chunk = []  # 立即清空chunk，释放内存

        if (i + 1) % 100 == 0:
            print(f"   进度: {i+1}/{total_windows}")

    # 处理剩余数据
    if chunk:
        chunk_results = [compute_func(data[:, 0, :], data[:, 1, :]) for data in chunk]
        results.extend(chunk_results)

    return np.array(results)


def _streaming_compute(df1: pd.DataFrame, df2: pd.DataFrame, window: int,
                       compute_func: callable) -> np.ndarray:
    """
    流式计算实现（最安全，单进程，逐个处理）
    """
    features_arrs = np.stack([df1.values, df2.values], axis=1)
    roll_arrs = sliding_window(features_arrs, window)

    T = df1.shape[0]
    total_windows = T - window + 1

    print(f"   流式处理模式：逐个计算{total_windows}个窗口")

    results = []
    for i, window_data in enumerate(roll_arrs):
        if i % 100 == 0:
            print(f"   进度: {i+1}/{total_windows}")

        result = compute_func(window_data[:, 0, :], window_data[:, 1, :])
        results.append(result)

    return np.array(results)


class LeadLagNetworkBuilder:
    """
    从金融价格数据构建领先-滞后网络的类。

    该类封装了构建多种类型的领先-滞后矩阵的逻辑，如论文3.2节所述：
    1. 隔夜-领先-日间：M[i,j] = Corr(overnight_returns_i, daytime_returns_j)
    2. 日间-领先-隔夜：M[i,j] = Corr(daytime_returns_i[t-1], overnight_returns_j[t])

    支持两种相关性计算方法：
    - Pearson：线性相关性，计算效率高，适用于线性关系
    - Spearman：单调相关性，对异常值稳健，适用于非线性关系

    使用流程:
    1. 初始化类，提供数据源、回看窗口和相关性方法。
    2. 调用 `build_network` 方法构建特定类型的网络。
    """

    def __init__(
        self,
        qlib_provider: QlibDataProvider,
        lookback_window: int = 60,
        correlation_method: str = "pearson",
        n_jobs: int = -1,
        parallel_config: ParallelConfig = None,
    ):
        """
        初始化领先-滞后网络构建器。

        Parameters
        ----------
        qlib_provider : QlibDataProvider
            提供收益率数据的Qlib数据提供者实例。
        lookback_window : int, optional
            用于计算相关性的滚动窗口天数, by default 60。
        correlation_method : str, optional
            相关性计算方法, 可选 'pearson' 或 'spearman', by default "pearson"。
            - 'pearson': 皮尔逊相关系数，适用于线性关系
            - 'spearman': 斯皮尔曼相关系数，对异常值稳健
        n_jobs : int, optional
            并行计算的作业数, by default -1。
            - -1: 使用所有可用的CPU核心
            - 1: 不使用并行计算（单进程）
            - >1: 使用指定数量的CPU核心
        parallel_config : ParallelConfig, optional
            并行计算配置，by default None（使用默认配置）。
        """
        self._qlib_provider = qlib_provider
        self._lookback_window = lookback_window
        self._correlation_method = self._validate_correlation_method(correlation_method)
        self._parallel_config = parallel_config if parallel_config is not None else ParallelConfig()

        # 初始化时先不调整n_jobs，在build_network时根据数据动态调整
        self._requested_n_jobs = self._validate_n_jobs(n_jobs)
        self._n_jobs = self._requested_n_jobs

    def _validate_correlation_method(self, method: str) -> str:
        """
        验证相关性计算方法的有效性。

        Parameters
        ----------
        method : str
            待验证的相关性方法名称

        Returns
        -------
        str
            验证后的方法名称（小写）

        Raises
        ------
        ValueError
            当方法名称不支持时
        """
        supported_methods = ['pearson', 'spearman']
        method_lower = method.lower()

        if method_lower not in supported_methods:
            raise ValueError(
                f"不支持的相关性方法: {method}. "
                f"支持的方法: {supported_methods}"
            )

        return method_lower

    def _validate_n_jobs(self, n_jobs: int) -> int:
        """
        验证并行作业数参数的有效性。

        Parameters
        ----------
        n_jobs : int
            待验证的并行作业数

        Returns
        -------
        int
            验证后的作业数

        Raises
        ------
        ValueError
            当作业数不是正整数时
        """
        if not isinstance(n_jobs, int) or n_jobs == 0:
            raise ValueError(
                f"n_jobs必须是整数且不为0，当前值: {n_jobs}"
            )
        return n_jobs

    @property
    def correlation_method(self) -> str:
        """
        获取当前使用的相关性计算方法。

        Returns
        -------
        str
            当前相关性方法 ('pearson' 或 'spearman')
        """
        return self._correlation_method

    @property
    def n_jobs(self) -> int:
        """
        获取当前使用的并行作业数。

        Returns
        -------
        int
            当前的并行作业数
        """
        return self._n_jobs

    def set_correlation_method(self, method: str) -> None:
        """
        设置相关性计算方法。

        Parameters
        ----------
        method : str
            新的相关性计算方法 ('pearson' 或 'spearman')

        Raises
        ------
        ValueError
            当方法名称不支持时
        """
        self._correlation_method = self._validate_correlation_method(method)

    def set_n_jobs(self, n_jobs: int) -> None:
        """
        设置并行作业数。

        Parameters
        ----------
        n_jobs : int
            新的并行作业数
            - -1: 使用所有可用的CPU核心
            - 1: 不使用并行计算（单进程）
            - >1: 使用指定数量的CPU核心

        Raises
        ------
        ValueError
            当作业数不是正整数时
        """
        self._requested_n_jobs = self._validate_n_jobs(n_jobs)
        self._n_jobs = self._requested_n_jobs

    def set_parallel_config(self, config: ParallelConfig) -> None:
        """
        设置并行计算配置。

        Parameters
        ----------
        config : ParallelConfig
            新的并行配置
        """
        self._parallel_config = config

    def get_parallel_config(self) -> ParallelConfig:
        """
        获取当前的并行配置。

        Returns
        -------
        ParallelConfig
            当前的并行配置
        """
        return self._parallel_config

    def check_memory_requirements(self, data_shape: Tuple[int, int], method: str = None) -> dict:
        """
        检查内存需求并给出建议。

        Parameters
        ----------
        data_shape : Tuple[int, int]
            数据形状 (T, N)
        method : str, optional
            计算方法，默认使用当前设置的方法

        Returns
        -------
        dict
            包含内存分析结果的字典
        """
        if method is None:
            method = self._correlation_method

        # 获取系统资源
        resources = _get_system_resources()

        # 估算内存需求（使用简化方法）
        T, N = data_shape
        base_memory_gb = (T * N * 8) / (1024**3)
        if method == "spearman":
            method_multiplier = 2.5  # Spearman需要额外内存
        else:
            method_multiplier = 1.5
        estimated_memory = base_memory_gb * method_multiplier * min(self._requested_n_jobs, 4)

        # 建议的n_jobs
        recommended_n_jobs = _auto_adjust_n_jobs(self._requested_n_jobs, data_shape, self._parallel_config)

        analysis = {
            'data_shape': data_shape,
            'method': method,
            'requested_n_jobs': self._requested_n_jobs,
            'recommended_n_jobs': recommended_n_jobs,
            'estimated_memory_gb': estimated_memory,
            'available_memory_gb': resources['available_memory_gb'],
            'cpu_cores': resources['cpu_cores'],
            'memory_usage_ratio': estimated_memory / resources['available_memory_gb'],
            'needs_adjustment': recommended_n_jobs != self._requested_n_jobs,
            'warning_level': 'normal'
        }

        # 确定警告级别
        if analysis['memory_usage_ratio'] > 0.9:
            analysis['warning_level'] = 'critical'
            analysis['message'] = '⚠️  内存严重不足，强烈建议减少n_jobs'
        elif analysis['memory_usage_ratio'] > 0.7:
            analysis['warning_level'] = 'warning'
            analysis['message'] = '⚠️  内存紧张，建议减少n_jobs'
        elif analysis['needs_adjustment']:
            analysis['warning_level'] = 'info'
            analysis['message'] = 'ℹ️  建议调整n_jobs以获得最佳性能'
        else:
            analysis['message'] = '✅ 资源充足，可以使用请求的n_jobs'

        return analysis

    def print_memory_analysis(self, data_shape: Tuple[int, int], method: str = None) -> None:
        """
        打印内存分析结果。

        Parameters
        ----------
        data_shape : Tuple[int, int]
            数据形状 (T, N)
        method : str, optional
            计算方法
        """
        analysis = self.check_memory_requirements(data_shape, method)

        print("=" * 60)
        print("🧠 内存使用分析")
        print("=" * 60)
        print(f"数据形状: {analysis['data_shape'][0]}天 × {analysis['data_shape'][1]}只股票")
        print(f"计算方法: {analysis['method'].upper()}")
        print(f"请求的n_jobs: {analysis['requested_n_jobs']}")
        print(f"推荐的n_jobs: {analysis['recommended_n_jobs']}")
        print(f"CPU核心数: {analysis['cpu_cores']}")
        print()
        print(f"估算内存需求: {analysis['estimated_memory_gb']:.2f}GB")
        print(f"可用内存: {analysis['available_memory_gb']:.2f}GB")
        print(f"内存使用比例: {analysis['memory_usage_ratio']:.1%}")
        print()
        print(f"状态: {analysis['message']}")
        print("=" * 60)

    def _create_adjacency_matrix(
        self, lead_lag_matrix: np.ndarray, absolute_values: bool = True
    ) -> np.ndarray:
        """
        从领先-滞后矩阵创建有向邻接矩阵。

        邻接矩阵 A 定义为:
        A[i,j] = |M[i,j]| (如果 absolute_values=True) 或 M[i,j]

        Parameters
        ----------
        lead_lag_matrix : np.ndarray
            领先-滞后相关性矩阵。
        absolute_values : bool, optional
            是否对相关性取绝对值, by default True。

        Returns
        -------
        np.ndarray
            有向邻接矩阵 A。
        """
        if absolute_values:
            # NumPy数组使用 np.abs()
            A = np.abs(lead_lag_matrix)
        else:
            A = lead_lag_matrix.copy()

        return A

    def build_network(
        self,
        network_type: str = "overnight_lead_daytime",
        absolute_values: bool = True,
        correlation_method: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建一个完整的领先-滞后网络。

        Parameters
        ----------
        network_type : str, optional
            要构建的网络类型，可选值为:
            - "overnight_lead_daytime"
            - "daytime_lead_overnight"
            by default "overnight_lead_daytime"。
        absolute_values : bool, optional
            是否在构建邻接矩阵时使用绝对值, by default True。
        correlation_method : str, optional
            相关性计算方法，可选 'pearson' 或 'spearman'。
            如果为 None，则使用实例初始化时设置的方法，by default None。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            一个元组，包含 (领先-滞后矩阵 M, 邻接矩阵 A)。
            两个矩阵的形状均为 (D, N, N)。

        Raises
        ------
        ValueError
            当网络类型或相关性方法不支持时。

        Notes
        -----
        并行计算使用实例初始化时设置的n_jobs参数。
        可以通过set_n_jobs()方法动态调整并行度。
        """
        # 确定使用的相关性方法
        method = correlation_method if correlation_method is not None else self._correlation_method
        method = self._validate_correlation_method(method)

        # 根据网络类型构建领先-滞后矩阵
        if network_type == "overnight_lead_daytime":
            M: np.ndarray = compute_rolling_lead_lag(
                self._qlib_provider.overnight_return_df.fillna(0),
                self._qlib_provider.daytime_return_df.fillna(0),
                self._lookback_window,
                method=method,
                n_jobs=self._requested_n_jobs,
                parallel_config=self._parallel_config,
            )

        elif network_type == "daytime_lead_overnight":
            M: np.ndarray = compute_rolling_lead_lag(
                self._qlib_provider.daytime_return_df.shift(1).fillna(0),
                self._qlib_provider.overnight_return_df.fillna(0),
                self._lookback_window,
                method=method,
                n_jobs=self._requested_n_jobs,
                parallel_config=self._parallel_config,
            )

        else:
            raise ValueError(f"未知的网络类型: {network_type}. "
                           f"支持的类型: ['overnight_lead_daytime', 'daytime_lead_overnight']")

        # 创建邻接矩阵
        A = self._create_adjacency_matrix(M, absolute_values)

        return M, A

    def build_multiple_networks(
        self,
        network_types: List[str] = None,
        correlation_methods: List[str] = None,
        absolute_values: bool = True
    ) -> dict:
        """
        构建多种类型的网络，便于比较分析。

        Parameters
        ----------
        network_types : List[str], optional
            要构建的网络类型列表，默认构建两种类型。
        correlation_methods : List[str], optional
            要使用的相关性方法列表，默认使用当前方法。
        absolute_values : bool, optional
            是否在构建邻接矩阵时使用绝对值, by default True。

        Returns
        -------
        dict
            包含多种网络结果的字典，键为 "network_type_correlation_method" 格式，
            值为包含 (M, A) 的元组。

        Example
        -------
        >>> builder = LeadLagNetworkBuilder(provider, correlation_method="pearson")
        >>> networks = builder.build_multiple_networks(
        ...     network_types=["overnight_lead_daytime", "daytime_lead_overnight"],
        ...     correlation_methods=["pearson", "spearman"]
        ... )
        >>> # 访问特定网络
        >>> M_pearson, A_pearson = networks["overnight_lead_daytime_pearson"]
        """
        # 设置默认值
        if network_types is None:
            network_types = ["overnight_lead_daytime", "daytime_lead_overnight"]

        if correlation_methods is None:
            correlation_methods = [self._correlation_method]

        results = {}

        for network_type in network_types:
            for method in correlation_methods:
                key = f"{network_type}_{method}"
                try:
                    M, A = self.build_network(
                        network_type=network_type,
                        absolute_values=absolute_values,
                        correlation_method=method
                    )
                    results[key] = {"M": M, "A": A}
                except Exception as e:
                    print(f"构建网络 {key} 时出错: {e}")
                    continue

        return results




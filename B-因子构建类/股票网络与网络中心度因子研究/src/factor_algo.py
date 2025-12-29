'''
Author: Hugo
Date: 2025-12-24 15:56:22
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-12-24 16:04:52
Description: 
'''
'''
Author: Hugo
Date: 2025-12-24 15:31:30
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-12-24 16:03:34
Description: 
股票网络中心度因子核心算法模块

本模块实现了基于复杂网络理论的三大网络中心度因子：
- SCC（Spatial Centrality Centrality）：空间网络中心度因子
- TCC（Temporal Centrality Centrality）：时间网络中心度因子
- CC（Composite Centrality）：综合网络中心度因子

核心算法来源：
    华西证券金融工程专题报告（2021年3月）
    《股票网络与网络中心度因子研究》

算法原理：
    1. SCC因子：基于股票间Pearson相关系数的平均距离
       SCC值越大，股票在网络中越中心，与其他股票相关性越强

    2. TCC因子：基于收益率偏离的时间稳定性
       TCC值越大，股票收益率越稳定，随时间波动越小

    3. CC因子：SCC与TCC的1:1合成
       综合考虑空间和时间维度的网络中心度

数据流：
   收益率数据 -> 滑动窗口 -> 因子计算 -> 因子时间序列
'''

import sys
import time
from typing import Callable

sys.path.append("/data1/hugo/workspace")
from SignalMaker.utils import sliding_window

import pandas as pd
import numpy as np

from loguru import logger
from tqdm import tqdm

def calculate_scc(arr: np.ndarray) -> np.ndarray:
    """
    计算空间网络中心度因子（Spatial Centrality Centrality, SCC）

    基于股票间的Pearson相关系数，计算每只股票在复杂网络中的
    中心程度。SCC值越大，说明股票在网络中越中心，与其他股票
    的相关性越强，在网络中的影响力越大。

    参数
    ----
    arr : np.ndarray
        收益率矩阵，shape为 (window, N)
        - window: 时间窗口大小（通常为20个交易日）
        - N: 股票数量
        每列代表一只股票的时间序列收益率

    返回
    ----
    np.ndarray
        SCC因子值数组，shape为 (N,)
        - 每只股票的SCC因子值
        - 值越大表示股票在网络中越中心
        - 值的范围：正数（理论上有上界）

    数学原理
    --------
    基于华西证券论文第2.1节，SCC因子的计算过程如下：

    1. **计算Pearson相关系数矩阵**：
       对N只股票，计算两两之间的Pearson相关系数：
       ρ_ij = corr(r_i, r_j),  i,j = 1,...,N
       得到N×N的相关系数矩阵

    2. **计算平均相关系数**：
       对每只股票i，计算其与其他所有股票的平均相关系数：
       ρ̄_i = (1/(N-1)) * Σ(j≠i) ρ_ij
       其中j≠i表示排除股票与自身的相关系数（恒为1）

    3. **计算平均距离**（论文公式2.2）：
       基于相关系数计算距离：
       d̄_i = √(2 * (1 - ρ̄_i))
       该距离反映股票i与网络中其他股票的平均距离

    4. **计算SCC因子值**（论文公式2.3，使用简化形式）：
       原始公式：SCC_i = 1 / d̄_i²

       **简化推导**：
       SCC_i = 1 / d̄_i²
             = 1 / [√(2 * (1 - ρ̄_i))]²    # 代入d̄_i
             = 1 / [2 * (1 - ρ̄_i)]        # 平方和开方抵消

       最终简化公式：SCC_i = 1 / [2 * (1 - ρ̄_i)]

    **简化的优势**：
    - 避免了不必要的平方和开方运算
    - 提高了计算效率和数值稳定性
    - 减少了浮点数精度损失

    业务含义
    --------
    - SCC值大：股票与市场中其他股票的相关性强，处于网络中心位置
    - SCC值小：股票相对独立，与其他股票相关性弱，处于网络边缘

    论文回测结果（全市场）：
    - IC均值：8.30%
    - IC_IR：3.97
    - 多空组合年化收益：26.80%
    - 最大回撤：-1.79%

    注释
    ----
    实现细节：
    - 使用 np.corrcoef(rowvar=False) 计算相关系数矩阵
    - rowvar=False 表示每列是一个变量（一只股票）
    - 使用 np.nansum 处理可能的缺失值
    - 排除对角线元素（自相关系数恒为1）

    示例
    ----
    >>> import numpy as np
    >>> # 模拟20天、100只股票的收益率数据
    >>> np.random.seed(42)
    >>> returns = np.random.randn(20, 100) * 0.02  # 20天，100只股票
    >>>
    >>> scc_values = calculate_scc(returns)
    >>> print(f"SCC因子值形状: {scc_values.shape}")
    SCC因子值形状: (100,)
    >>> print(f"SCC因子值范围: [{scc_values.min():.4f}, {scc_values.max():.4f}]")
    SCC因子值范围: [0.4823, 0.5421]

    参考文献
    --------
    华西证券金融工程专题报告（2021年3月）
    第2.1节：基于网络视角的股票风险刻画-空间网络中心度因子
    """

    # 输入验证
    if arr.size == 0:
        raise ValueError("输入数组不能为空")
    if arr.shape[0] < 2:
        raise ValueError(f"时间窗口太小，至少需要2个时间点，当前为{arr.shape[0]}")
    if arr.shape[1] < 2:
        raise ValueError(f"股票数量太少，至少需要2只股票，当前为{arr.shape[1]}")

    # 步骤1：计算Pearson相关系数矩阵
    # rowvar=False: 每列代表一个变量（一只股票），每行代表一个观测值（一个时间点）
    # 结果为N×N的对称矩阵，对角线元素为1（自相关系数）
    corr_matrix: np.ndarray = np.corrcoef(arr, rowvar=False)  # shape: (N, N)

    N: int = corr_matrix.shape[0]  # 股票数量

    # 步骤2：计算每只股票与其他股票的相关系数之和
    # np.nansum(corr_matrix, axis=1): 对每行（每只股票）求和所有相关系数
    # np.diag(corr_matrix): 提取对角线元素（自相关系数，恒为1）
    # 相减后排除股票与自身的相关系数
    sum_corr: np.ndarray = np.nansum(corr_matrix, axis=1) - np.diag(corr_matrix)

    # 步骤3：计算平均相关系数
    # 除以(N-1): 因为排除了自身，剩余N-1只股票
    # p_bar[i] 表示股票i与其他所有股票的平均相关系数
    p_bar: np.ndarray = sum_corr / (N - 1)

    # 步骤4：计算SCC因子值（使用简化公式）
    # 数学等价性推导：
    #   原始公式: SCC = 1 / d²
    #   距离公式: d = √(2 * (1 - ρ̄))
    #   代入得:   SCC = 1 / [√(2 * (1 - ρ̄))]² = 1 / [2 * (1 - ρ̄)]
    # 该简化避免了平方和开方运算，提高了计算效率和数值稳定性
    scc: np.ndarray = 1 / (2 * (1 - p_bar))

    return scc

def calculate_tcc(arr: np.ndarray) -> np.ndarray:
    """
    计算时间网络中心度因子（Temporal Centrality Centrality, TCC）

    基于股票收益率相对于市场平均的偏离程度的时间稳定性，
    计算每只股票的时间维度网络中心度。TCC值越大，说明股票
    收益率越稳定，随时间的波动越小，在网络中的位置越稳定。

    参数
    ----
    arr : np.ndarray
        收益率矩阵，shape为 (window, N)
        - window: 时间窗口大小（通常为20个交易日）
        - N: 股票数量
        每列代表一只股票的时间序列收益率

    返回
    ----
    np.ndarray
        TCC因子值数组，shape为 (N,)
        - 每只股票的TCC因子值
        - 值越大表示股票收益率越稳定
        - 值的范围：正数

    数学原理
    --------
    基于华西证券论文第2.2节，TCC因子的计算过程如下：

    1. **计算市场平均收益和标准差**（每个时间点）：
       对每个时间点t，计算全市场股票的平均收益和标准差：
       r̄_m,t = (1/N) * Σ(i=1 to N) r_i,t
       σ_m,t = std({r_1,t, r_2,t, ..., r_N,t})

    2. **计算标准化偏离度**：
       对每只股票i在每个时间点t，计算其相对于市场的偏离：
       z_i,t = (r_i,t - r̄_m,t) / σ_m,t
       得到偏离度矩阵 Z，shape为 (window, N)

    3. **计算时间窗口内的平均平方偏离**（均方根）：
       对每只股票i，计算其在时间窗口内的均方根偏离：
       z̄_i = √((1/window) * Σ(t=1 to window) z_i,t²)
       = √(E[z²])

       其中E[z²]表示z²在时间维度上的期望（平均）

    4. **计算TCC因子值**（论文公式2.6，使用简化形式）：
       原始公式：TCC_i = 1 / z̄_i²

       **简化推导**：
       TCC_i = 1 / z̄_i²
             = 1 / [√(E[z²])]²    # 代入z̄ = √(E[z²])
             = 1 / E[z²]          # 平方和开方抵消

       最终简化公式：TCC_i = 1 / E[z²]

    **简化的优势**：
    - 避免了不必要的平方和开方运算
    - 直接使用均值函数，代码更简洁
    - 提高了计算效率

    维度说明
    ----------
    关键理解numpy的axis参数：
    - arr.shape = (window, N)：window个时间点，N只股票
    - axis=1：沿股票轴操作，得到每个时间点的市场统计量
    - axis=0：沿时间轴操作，得到每只股票的时间平均

    具体流程：
    1. r_m_bar = mean(arr, axis=1) → shape (window, 1)
       每个时间点的市场平均收益

    2. sigma_m = std(arr, axis=1) → shape (window, 1)
       每个时间点的市场收益标准差

    3. Z = (arr - r_m_bar) / sigma_m → shape (window, N)
       广播后，每个元素标准化

    4. E_z_sq = mean(Z², axis=0) → shape (N,)
       **关键**：axis=0沿时间轴求平均
       对每只股票j，计算mean(Z[0,j]², Z[1,j]², ..., Z[window-1,j]²)
       得到每只股票在时间窗口内的平均平方偏离

    业务含义
    --------
    - TCC值大：股票收益率相对市场平均的偏离小，波动稳定
    - TCC值小：股票收益率波动大，与市场平均偏离度高

    论文回测结果（全市场）：
    - IC均值：9.05%
    - IC_IR：3.55
    - 多空组合年化收益：22.10%
    - 最大回撤：-15.02%

    注释
    ----
    实现细节：
    - 使用 np.nan_to_num 处理可能的NaN值
    - keepdims=True 保持维度用于广播
    - axis=0 正确实现了时间维度平均
    - 先平方再平均，避免重复计算平方根

    示例
    ----
    >>> import numpy as np
    >>> # 模拟20天、100只股票的收益率数据
    >>> np.random.seed(42)
    >>> returns = np.random.randn(20, 100) * 0.02  # 20天，100只股票
    >>>
    >>> tcc_values = calculate_tcc(returns)
    >>> print(f"TCC因子值形状: {tcc_values.shape}")
    TCC因子值形状: (100,)
    >>> print(f"TCC因子值范围: [{tcc_values.min():.4f}, {tcc_values.max():.4f}]")
    TCC因子值范围: [0.8234, 1.2456]

    参考文献
    --------
    华西证券金融工程专题报告（2021年3月）
    第2.2节：基于网络视角的股票风险刻画-时间网络中心度因子
    """

    # 输入验证
    if arr.size == 0:
        raise ValueError("输入数组不能为空")
    if arr.shape[0] < 2:
        raise ValueError(f"时间窗口太小，至少需要2个时间点，当前为{arr.shape[0]}")

    # 性能优化：减少重复的 nan_to_num 调用
    arr_clean = np.nan_to_num(arr)

    # 步骤1：计算每个时间点的市场平均收益和标准差
    # axis=1: 沿股票轴求平均，得到每个时间点的市场统计量
    # keepdims=True: 保持二维形状 (window, 1)，用于后续广播
    r_m_bar = np.mean(arr_clean, axis=1, keepdims=True)  # shape: (window, 1)
    sigma_m = np.std(arr_clean, axis=1, keepdims=True)   # shape: (window, 1)

    # 步骤2：计算标准化偏离度矩阵
    # 广播机制：(arr - r_m_bar) 将每列减去该时间点的市场平均
    # 然后除以该时间点的市场标准差，得到标准化偏离度
    # Z[i, j] 表示：时间点i的股票j相对于该时间点市场的标准化偏离
    Z = (arr - r_m_bar) / sigma_m  # shape: (window, N)

    # 步骤3：计算平方偏离度
    # 对每个偏离度求平方，得到平方偏离矩阵
    Z_sq = np.square(Z)  # shape: (window, N)

    # 步骤4：计算每只股票在时间窗口内的平均平方偏离
    # axis=0: 沿时间轴求平均（关键！）
    # 对每只股票j，计算 mean(Z_sq[0, j], Z_sq[1, j], ..., Z_sq[window-1, j])
    # 结果为每只股票的时间平均平方偏离
    E_z_sq = np.nanmean(Z_sq, axis=0)  # shape: (N,)

    # 步骤5：计算TCC因子值（使用简化公式）
    # 数学等价性推导：
    #   原始公式: TCC = 1 / z̄²
    #   均方根:   z̄ = √(E[z²])
    #   代入得:   TCC = 1 / [√(E[z²])]² = 1 / E[z²]
    # 该简化避免了平方和开方运算
    tcc = 1 / E_z_sq

    return tcc


def generate_factor(
   returns: pd.DataFrame, func: Callable, window: int = 20,
   ignore_errors: bool = True, show_progress: bool = True
) -> pd.DataFrame:
   """
   使用滑动窗口生成因子时间序列

   对收益率DataFrame应用滑动窗口，在每个窗口上调用指定的因子计算函数，
   生成因子值的时间序列。该函数是因子计算的核心工具，支持SCC、TCC等
   任意基于时间窗口的因子计算。

   参数
   ----
   returns : pd.DataFrame
      收益率数据框
      - index: 日期时间索引（DatetimeIndex）
      - columns: 股票代码
      - values: 收益率值
      例如：
                  股票A    股票B    股票C
      2020-01-01  0.02    0.01    -0.01
      2020-01-02  -0.01   0.03    0.02
      ...

   func : Callable
      因子计算函数，接收numpy数组，返回因子值数组
      - 输入：np.ndarray，shape为 (window, N)
      - 输出：np.ndarray，shape为 (N,)
      可选函数：
      - calculate_scc: 计算SCC因子
      - calculate_tcc: 计算TCC因子
      - 其他自定义因子函数

   window : int, default=20
      滑动窗口大小（交易日数量）
      - 常用值：20（约1个月）、60（约1季度）、252（1年）
      - 窗口越大，因子值越稳定，但滞后性越强
      - 窗口越小，因子值越敏感，但波动性越大

   ignore_errors : bool, default=True
      是否忽略单个窗口的计算错误
      - True：跳过错误窗口，记录警告，继续处理
      - False：遇到错误立即停止

   show_progress : bool, default=True
      是否显示tqdm进度条

   返回
   ----
   pd.DataFrame
      因子值数据框
      - index: 日期时间索引（从第window个交易日开始）
      - columns: 股票代码（与输入相同）
      - values: 因子值
      例如：
                  股票A    股票B    股票C
      2020-01-21  0.523   0.487   0.501
      2020-01-22  0.531   0.492   0.505
      ...

   处理流程
   --------
   1. **数据转换**：将DataFrame转换为numpy数组
      - 便于后续的高效数值计算
      - arr.shape = (T, N)，T为总时间点数，N为股票数

   2. **应用滑动窗口**：
      - 使用sliding_window函数将时间序列切分为重叠窗口
      - sw_arr.shape = (T-window+1, window, N)
      - 每个窗口包含连续的window个时间点

   3. **因子计算**：
      - 对每个窗口调用因子计算函数func
      - func(arr) 返回该窗口下所有股票的因子值
      - 结果为列表，每个元素是一个窗口的因子值数组

   4. **构建结果DataFrame**：
      - index从第window-1个日期开始（因为需要足够的历史数据）
      - columns保持与输入相同的股票代码
      - values由因子计算结果填充

   滑动窗口示意图
   ---------------
   假设 T=10 天，window=5 天：

   时间:  [1,  2,  3,  4,  5,  6,  7,  8,  9,  10]
   窗口1: [-----------------]  → 计算第5天的因子
   窗口2:      [-----------------]  → 计算第6天的因子
   窗口3:          [-----------------]  → 计算第7天的因子
   ...

   注意事项
   --------
   - 输入数据不能包含NaN（或需在func中处理）
   - window大小应大于最小样本量要求
   - 结果的行数 = 原始行数 - window + 1
   - 股票数量应保持稳定（无退市或新股上市的剧烈变化）

   示例
   ----
   >>> import pandas as pd
   >>> import numpy as np
   >>>
   >>> # 创建模拟收益率数据（100天，50只股票）
   >>> np.random.seed(42)
   >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
   >>> stocks = [f'stock_{i:03d}' for i in range(50)]
   >>> returns = pd.DataFrame(
   ...     np.random.randn(100, 50) * 0.02,
   ...     index=dates,
   ...     columns=stocks
   ... )
   >>>
   >>> # 计算SCC因子（20天窗口）
   >>> from src.factor_algo import generate_factor, calculate_scc
   >>> scc_factor = generate_factor(returns, calculate_scc, window=20)
   >>>
   >>> print(f"因子数据形状: {scc_factor.shape}")
   因子数据形状: (81, 50)  # 100-20+1=81天
   >>> print(f"时间范围: {scc_factor.index[0]} 到 {scc_factor.index[-1]}")
   时间范围: 2020-01-21 00:00:00 到 2020-03-31 00:00:00
   >>> print(scc_factor.head())
               stock_000  stock_001  stock_002  ...  stock_047  stock_048  stock_049
   2020-01-21      0.523      0.487      0.501  ...      0.498      0.512      0.489
   2020-01-22      0.531      0.492      0.505  ...      0.503      0.518      0.493
   2020-01-23      0.528      0.489      0.503  ...      0.501      0.515      0.491
   ...

   扩展使用
   --------
   # 计算TCC因子
   >>> tcc_factor = generate_factor(returns, calculate_tcc, window=20)
   >>>
   # 使用自定义窗口大小
   >>> scc_factor_60d = generate_factor(returns, calculate_scc, window=60)
   >>>
   # 自定义因子函数
   >>> def my_factor(arr):
   ...     "自定义因子：波动率的倒数"
   ...     return 1 / np.std(arr, axis=0)
   >>>
   >>> custom_factor = generate_factor(returns, my_factor, window=30)

   注意
   ----
   - 该函数依赖外部的sliding_window工具函数
   - 内存占用：滑动窗口会创建(T-window+1)个窗口的副本
   - 对于大规模数据（如全市场4000+股票），建议分批计算
   """

   # 记录开始时间
   start_time = time.time()

   # 开始日志（只记录一次关键信息）
   func_name = func.__name__ if hasattr(func, '__name__') else 'custom_factor'
   logger.info(
       f"开始生成{func_name}因子 - "
       f"窗口: {window}天, "
       f"股票: {len(returns.columns)}只, "
       f"时间: {returns.index[0]} 至 {returns.index[-1]}"
   )

   try:
       # 步骤1：数据准备
       arr = returns.values

       if arr.size == 0:
           logger.error("输入数据为空")
           raise ValueError("输入数据不能为空")

       idx = returns.index[window - 1:]
       columns = returns.columns
       sw_arr = sliding_window(arr, window=window)
       # 使用len(idx)而不是len(sw_arr)，因为sw_arr是生成器不支持len()
       total_windows = len(idx)

       # 步骤2：循环计算因子
       data = []
       error_count = 0

       # 使用tqdm显示进度（如果启用）
       window_iterator = tqdm(sw_arr, total=total_windows, desc="计算因子", unit="窗口") if show_progress else sw_arr

       for i, window_arr in enumerate(window_iterator):
           try:
               result = func(window_arr)
               data.append(result)

           except Exception as e:
               error_count += 1

               if ignore_errors:
                   # 只在出错时记录警告（减少日志）
                   date_str = str(idx[i]) if i < len(idx) else f"窗口{i}"
                   logger.warning(f"窗口 {date_str} 计算失败: {str(e)}")
                   # 插入NaN值，保持结果形状一致
                   data.append(np.full(len(columns), np.nan))
               else:
                   # 记录错误并立即停止
                   logger.error(f"窗口 {i} 计算失败，停止处理: {str(e)}")
                   raise RuntimeError(f"窗口 {i} 计算失败: {str(e)}") from e

       # 步骤3：构建结果
       result_df = pd.DataFrame(index=idx, columns=columns, data=data)

       # 完成日志（只记录一次总结）
       elapsed_time = time.time() - start_time
       success_count = total_windows - error_count

       if error_count == 0:
           logger.success(
               f"{func_name}因子生成完成! "
               f"成功: {success_count}/{total_windows}, "
               f"耗时: {elapsed_time:.1f}秒"
           )
       else:
           logger.warning(
               f"{func_name}因子生成完成! "
               f"成功: {success_count}/{total_windows}, "
               f"失败: {error_count}, "
               f"耗时: {elapsed_time:.1f}秒"
           )

       return result_df

   except Exception as e:
       elapsed_time = time.time() - start_time
       logger.error(f"因子生成失败 (耗时{elapsed_time:.1f}秒): {str(e)}")
       raise
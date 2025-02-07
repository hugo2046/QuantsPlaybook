"""
Author: Hugo
Date: 2024-07-12 13:39:35
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-08-07 10:50:47
Description: 

在向量化构造信号时视角最好用于在T日,避免使用到shift(-N)这种引用到未来数据的情况,信号多了容易忽略,
使用后尽量在结果处使用shift(2)将视角还原到T日,所以最好使用rolling的形式构造一些信号。

使用说明:https://github.com/hugo2046/QuantsPlaybook/blob/master/C-%E6%8B%A9%E6%97%B6%E7%B1%BB/%E5%9F%BA%E4%BA%8E%E9%B3%84%E9%B1%BC%E7%BA%BF%E7%9A%84%E6%8C%87%E6%95%B0%E6%8B%A9%E6%97%B6%E5%8F%8A%E8%BD%AE%E5%8A%A8%E7%AD%96%E7%95%A5/zs_timing_strategy.ipynb
"""

from typing import Tuple, Union, Generator

import numpy as np
import pandas as pd
from talib import MACD, SMA

from .utils import sliding_window

__all__ = [
    "calculate_alligator_indicator",
    "alignment_signal",
    "alligator_classify_rows",
    "get_alligator_signal",
    "calculate_ao",
    "check_continuation_up_or_down",
    "get_ao_indicator_signal",
    "check_classily_top_fractal",
    "check_classily_bottom_fractal",
    "get_fractal_classily",
    "get_fractal_signal",
    "macd_classify_cols",
    "get_macd_signal",
    "get_north_money_signal"
]


def get_shift(arr: np.ndarray, periods: int, axis: int = 0) -> np.ndarray:
    """
    获取滞后矩阵或数组。

    :param arr: 输入的数组或矩阵。
    :type arr: np.ndarray
    :param periods: 滞后步长。
    :type periods: int
    :param axis: 滞后操作的轴，0 表示行，1 表示列。
    :type axis: int
    :return: 滞后后的数组或矩阵。
    :rtype: np.ndarray
    :raises ValueError: 如果滞后步长大于数组长度。
    """
    if arr.shape[axis] < periods:
        raise ValueError("滞后步长大于数组长度。")
    elif periods == 0:
        return arr

    tmp: np.ndarray = np.roll(arr, periods, axis=axis)
    if axis == 0:
        filler = np.nan * np.ones(periods)
    else:
        filler = np.nan * np.ones((periods, arr.shape[1]))

    if periods > 0:
        if axis == 0:
            tmp[:periods] = filler
        else:
            tmp[:, :periods] = filler
    else:
        if axis == 0:
            tmp[periods:] = filler
        else:
            tmp[:, periods:] = filler

    return tmp

#######################################################################################################
#                               鳄鱼线指标计算
#######################################################################################################


def calculate_alligator_indicator(
    close_arr: Union[pd.Series, np.ndarray],
    periods: Tuple[int] = None,
    lag: Tuple[int] = None,
) -> np.ndarray:
    """
    计算鳄鱼线指标。

    参数：
        close_arr (Union[pd.Series, np.ndarray]): 收盘价序列，可以是Pandas Series或NumPy数组。
        periods (Tuple[int]): 需要计算的周期列表。
        lag (Tuple[int]): 滞后周期列表。

    返回：
        np.ndarray: 鳄鱼线指标的计算结果。
                    [下颚线，牙齿线，上唇线]
    Raises:
        ValueError: 如果输入的数据长度小于最大周期。
        ValueError: 如果输入的周期不是列表或元组。
    """
    if periods is None:
        periods: Tuple[int] = (13, 8, 5)

    if lag is None:
        lag: Tuple[int] = (8, 5, 3)

    if isinstance(close_arr, pd.Series):
        close_arr: np.ndarray = close_arr.values

    max_size: int = max(periods)
    if close_arr.shape[0] < max_size:
        raise ValueError("输入的数据长度小于最大周期。")

    if not isinstance(periods, (list, tuple)):
        raise ValueError("输入的周期不是列表或元组。")

    close_arr: np.ndarray = close_arr.astype(np.float64)
    # 计算鳄鱼线
    alligator_arr: np.ndarray = np.array(
        [get_shift(SMA(close_arr, i), j) for i, j in zip(periods, lag)]
    ).T

    return alligator_arr


def alignment_signal(arr: np.ndarray, alignment_type="bullish") -> np.ndarray:
    """
    根据给定的排列类型，判断数组是否满足排列条件，并返回触发信号。
    参数：
    arr (np.ndarray): 输入的数组。这里的数组类下标类似于[ma30,ma20,ma10,ma5]
    alignment_type (str, 可选): 排列类型，可选值为 'bullish' 或 'bearish'。默认为 'bullish'。
    返回：
    np.ndarray: 触发信号的布尔数组。
    异常：
    ValueError: 当 alignment_type 不是 'bullish' 或 'bearish' 时抛出异常。
    """
    if alignment_type == "bullish":
        # 多头排列：每列数值比右边的大
        is_aligned: np.ndarray = np.all(np.diff(arr, axis=1) > 0, axis=1)
    elif alignment_type == "bearish":
        # 空头排列：每列数值比右边的小
        is_aligned: np.ndarray = np.all(np.diff(arr, axis=1) < 0, axis=1)
    else:
        raise ValueError("alignment_type must be 'bullish' or 'bearish'")

    # 计算触发信号：今天形成排列且昨天不是
    signal: np.ndarray = np.zeros(len(is_aligned), dtype=bool)
    signal[1:] = is_aligned[1:] & ~is_aligned[:-1]

    return signal


# def alligator_classify_rows(alligator_arr: np.ndarray) -> np.ndarray:
#     """
#     根据鳄鱼线指标对行进行分类。

#     参数：
#         alligator_arr (np.ndarray): 鳄鱼线指标数组，形状为 (n, 3)，其中 n 为行数。

#     返回：
#         np.ndarray: 包含分类结果的数组，形状为 (n,)，其中 n 为行数。分类结果为 1、0 或 -1，
#         分别表示满足下颚线、牙齿线、上唇线的条件。

#     示例：
#         >>> alligator_arr = np.array([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
#         >>> alligator_classify_rows(alligator_arr)
#         array([1, -1, 0])

#     """
#     # 数组下标[下颚线，牙齿线，上唇线]
#     # 检查 x[0] < x[1] < x[2]
#     condition1 = np.all(np.diff(alligator_arr, axis=1) > 0, axis=1)
#     # 检查 x[0] > x[1] > x[2]
#     condition2 = np.all(np.diff(alligator_arr, axis=1) < 0, axis=1)

#     # 初始化结果数组，所有元素为 np.nan
#     result: np.ndarray = np.full(alligator_arr.shape[0], np.nan, dtype=float)
#     # 满足 condition1 的行标记为 1
#     result[condition1] = 1
#     # 满足 condition2 的行标记为 -1
#     result[condition2] = -1

#     return result


def alligator_classify_rows(alligator_arr: np.ndarray) -> np.ndarray:
    """
    根据鳄鱼线指标对行进行分类。

    参数：
        alligator_arr (np.ndarray): 鳄鱼线指标数组，形状为 (n, 3)，其中 n 为行数。

    返回：
        np.ndarray: 包含分类结果的数组，形状为 (n,)，其中 n 为行数。分类结果为 1、0 或 -1，
        分别表示满足下颚线、牙齿线、上唇线的条件。

    示例：
        >>> alligator_arr = np.array([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        >>> alligator_classify_rows(alligator_arr)
        array([1, -1, 0])

    """
    # 数组下标[下颚线，牙齿线，上唇线]
    # 检查 x[0] < x[1] < x[2]
    condition1 = alignment_signal(alligator_arr, "bullish")
    # 检查 x[0] > x[1] > x[2]
    condition2 = alignment_signal(alligator_arr, "bearish")

    # 初始化结果数组，所有元素为 np.nan
    result: np.ndarray = np.full(alligator_arr.shape[0], np.nan, dtype=float)
    # 满足 condition1 的行标记为 1
    result[condition1] = 1
    # 满足 condition2 的行标记为 -1
    result[condition2] = -1

    return result


def get_alligator_signal(
    close_df: Union[pd.DataFrame, pd.Series],
    periods: Tuple[int] = None,
    lag: Tuple[int] = None,
    keep_pre_status: bool = True,
) -> Union[pd.DataFrame, pd.Series]:
    """
    根据鳄鱼线指标生成交易信号。

    参数：
    close_df (Union[pd.DataFrame, pd.Series]): 收盘价数据，可以是DataFrame或Series。
    periods (Tuple[int], optional): 鳄鱼线指标的周期。默认为None。
    lag (Tuple[int], optional): 鳄鱼线指标的滞后周期。默认为None。
    keep_pre_status (bool, optional): 是否保持前一交易日的仓位。默认为True。
    返回：
    pd.DataFrame: 包含交易信号的DataFrame。

    Raises:
    ValueError: 如果输入的数据不是DataFrame或Series。
    """
    # 无信号（沉睡的鳄鱼）， 维持前一交易日的仓位
    # 故fillna(method='ffill')
    if isinstance(close_df, pd.DataFrame):

        signal: pd.DataFrame = close_df.apply(
            lambda x: alligator_classify_rows(
                calculate_alligator_indicator(x, periods, lag)
            ),
            raw=True,
        )

    elif isinstance(close_df, pd.Series):

        signal: pd.Series = pd.Series(
            alligator_classify_rows(
                calculate_alligator_indicator(close_df, periods, lag)
            ),
            index=close_df.index,
        )

    else:
        raise ValueError("输入的数据不是DataFrame或Series")

    if keep_pre_status:
        return signal.ffill().fillna(0)

    return signal


#######################################################################################################
#                               AO指标计算
#######################################################################################################


def calculate_ao(
    high_df: Union[pd.DataFrame, pd.Series],
    low_df: Union[pd.DataFrame, pd.Series],
    periods: Tuple[int] = (5, 34),
) -> pd.DataFrame:
    """
    计算鳄鱼线指标（AO）。

    参数：
        high_df (Union[pd.DataFrame, pd.Series]): 高价数据，可以是DataFrame或Series。
        low_df (Union[pd.DataFrame, pd.Series]): 低价数据，可以是DataFrame或Series。
        periods (Tuple[int], optional): 计算SMA的周期，默认为(5, 34)。

    返回：
        pd.DataFrame: 动量震荡指标（AO）。

    计算方式：
        在tradiview中，AO指标的计算方式是：
        AO = SMA(High + Low, 5) - SMA(High + Low, 34)
        但是研报中的计算方式是：AO = SMA(High - Low, 5) - SMA(High - Low, 34)
        参考链接：https://cn.tradingview.com/support/solutions/43000501826/
    """
    median_price: Union[pd.DataFrame, pd.Series] = (high_df - low_df) * 0.5

    return (
        median_price.rolling(periods[0]).mean()
        - median_price.rolling(periods[1]).mean()
    )


def check_continuation_up_or_down(arr: np.ndarray) -> int:
    """
    检查给定数组是否持续上涨或持续下跌。

    Parameters:
        arr (np.ndarray): 输入的一维数组。

    Returns:
        int: 如果数组持续上涨，返回1；如果数组持续下跌，返回-1；如果数组不是持续上涨或持续下跌，返回np.nan。
    """
    if np.all(np.diff(arr) > 0):
        return 1
    elif np.all(np.diff(arr) < 0):
        return -1
    else:
        return np.nan


def get_ao_indicator_signal(
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    window: int = 3,
    keep_pre_status: bool = True,
) -> pd.DataFrame:
    """
    AO 指标连续window个交易日上行则1；
    连续window个交易日下行则-1；

    参数：
        high_df (pd.DataFrame): 高价数据的DataFrame。
        low_df (pd.DataFrame): 低价数据的DataFrame。
        window (int): 滚动窗口的大小，默认为3。
        keep_pre_status (bool): 是否保持前一交易日的仓位。默认为True。

    返回：
        pd.DataFrame: 包含连续上涨和下跌信号的DataFrame。
    """
    ao_indicator: pd.DataFrame = calculate_ao(high_df, low_df)

    signal: pd.DataFrame = ao_indicator.rolling(window).apply(
        check_continuation_up_or_down, raw=True
    )
    if keep_pre_status:
        return signal.ffill().fillna(0)
    return signal


#######################################################################################################
#                                   分型指标计算
#######################################################################################################


def check_classily_top_fractal(high_arr: np.ndarray, low_arr: np.ndarray) -> int:
    """
    检查是否存在经典的顶部分型。

    参数：
    high_arr: numpy.ndarray
        包含高点数据的数组。
    low_arr: numpy.ndarray
        包含低点数据的数组。

    返回值：
    int
        如果存在经典的顶部分型，则返回1；否则返回0。

    异常：
    ValueError
        当输入的数据长度不一致时，抛出该异常。
    """
    # 无论输入长度
    if high_arr.shape[0] != low_arr.shape[0]:
        raise ValueError("输入的数据长度不一致。")

    # 检查 high_t1 < high_t2 > high_t3 条件
    condition_high = (high_arr[-3, :] < high_arr[-2, :]) & (
        high_arr[-2, :] > high_arr[-1, :]
    )

    # 检查 low_t1 < low_t2 > low_ 条件
    condition_low = (low_arr[-3, :] < low_arr[-2, :]) & (
        low_arr[-2, :] > low_arr[-1, :]
    )

    return (condition_high & condition_low) * 1


def check_classily_bottom_fractal(high_arr: np.ndarray, low_arr: np.ndarray) -> int:
    """
    检查是否存在经典底部分型。

    参数：
    high_arr : np.ndarray
        包含高点数据的数组。
    low_arr : np.ndarray
        包含低点数据的数组。

    返回值：
    int
        如果存在经典底部分型，返回-1；否则返回0。

    Raises:
    ValueError
        如果输入的数据长度不一致。

    示例：
    >>> high_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> low_arr = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    >>> check_classily_bottom_fractal(high_arr, low_arr)
    -1
    """
    if high_arr.shape[0] != low_arr.shape[0]:
        raise ValueError("输入的数据长度不一致。")

    # 检查 low_t1 > low_t2 < low_t3 条件
    condition_low = (low_arr[-3, :] > low_arr[-2, :]) & (
        low_arr[-2, :] < low_arr[-1, :]
    )
    # 检查 high_t1 > high_t2 < high_t3 条件
    condition_high = (high_arr[-3, :] > high_arr[-2, :]) & (
        high_arr[-2, :] < high_arr[-1, :]
    )

    # 同时满足上述两个条件
    return (condition_low & condition_high) * -1


def get_fractal_classily(
    high_df: pd.DataFrame, low_df: pd.DataFrame, window: int = 3
) -> pd.DataFrame:
    """
    根据高点和低点数据计算分型信号。

    参数：
    - high_df (pd.DataFrame): 包含高点数据的DataFrame。
    - low_df (pd.DataFrame): 包含低点数据的DataFrame。

    返回：
    - pd.DataFrame: 包含分型信号的DataFrame，其中1表示顶分型，-1表示底分型。

    异常：
    - ValueError: 如果输入的数据不是DataFrame。

    示例：
    ```python
    high_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    low_df = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
    result = get_fractal(high_df, low_df)
    print(result)
    ```
    """
    if not isinstance(high_df, pd.DataFrame) or not isinstance(low_df, pd.DataFrame):
        raise ValueError("输入的数据不是DataFrame")

    high_df, low_df = high_df.align(low_df)
    data: np.ndarray = np.stack((high_df.values, low_df.values), axis=2)
    datas: Generator = sliding_window(data, window)
    arr = np.array(
        [
            check_classily_top_fractal(arr[:, :, 0], arr[:, :, 1])
            + check_classily_bottom_fractal(arr[:, :, 0], arr[:, :, 1])
            for arr in datas
        ]
    )

    # 如果为1则为顶分型，如果为-1则为底分型
    return pd.DataFrame(arr, index=high_df.index[2:], columns=high_df.columns)


def get_fractal_signal(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    keep_pre_status: bool = True,
) -> pd.DataFrame:
    """
    根据鳄鱼线指标生成交易信号。

    参数：
        close_df (pd.DataFrame): 指数收盘价的数据框。
        high_df (pd.DataFrame): 包含高点数据的DataFrame。
        low_df (pd.DataFrame): 包含低点数据的DataFrame。

    返回：
        pd.DataFrame: 包含交易信号的数据框。如果指数收盘价向上突破最近上分形的最高价，则信号为看多，仓位为满仓；
        如果指数收盘价向下突破最近下分形的最低价，则信号为看空，仓位为空仓。

    """

    fractal_df: pd.DataFrame = get_fractal_classily(high_df, low_df)

    up_ser: pd.DataFrame = (close_df > close_df.shift(3)) * 1
    down_ser: pd.DataFrame = (close_df < close_df.shift(3)) * -1

    signal: pd.DataFrame = ((up_ser + fractal_df.shift(1)) == 2) * 1 + (
        (down_ser + fractal_df.shift(1)) == -2
    ) * -1
    if keep_pre_status:
        return signal.ffill().fillna(0)
    return signal


def evaluate_signals(row) -> int:
    """
    根据给定的信号行评估结果。

    参数:
    - row: 一个包含 Alligator, AO, 和 Fractal 信号的序列或列表。

    返回:
    - 1, 如果 Alligator 等于 1 且 AO 或 Fractal 至少有一个等于 1。
    - -1, 如果 Alligator、AO 或 Fractal 中任意一个等于 -1。
    - 0, 其他情况。
    """
    # 检查 Alligator 等于 1 且 AO 或 Fractal 至少有一个等于 1 的情况
    if row[0] == 1 and (row[1] == 1 or row[2] == 1):
        return 1
    # 检查 Alligator、AO 或 Fractal 中任意一个等于 -1 的情况
    elif row[0] == -1 or row[1] == -1 or row[2] == -1:
        return -1
    # 如果以上条件都不满足，返回 0
    else:
        return 0


#######################################################################################################
#                                   MACD指标计算
#######################################################################################################


def macd_classify_cols(
    dif: Union[pd.Series, np.ndarray],
    dea: Union[pd.Series, np.ndarray],
    hist: Union[pd.Series, np.ndarray],
) -> Union[pd.Series, np.ndarray]:
    """
    根据 MACD 指标的 dif、dea 和 hist 值，对数据进行分类。

    参数：
        dif (Union[pd.Series, np.ndarray]): dif 值，可以是 pandas Series 或 numpy 数组。
        dea (Union[pd.Series, np.ndarray]): dea 值，可以是 pandas Series 或 numpy 数组。
        hist (Union[pd.Series, np.ndarray]): hist 值，可以是 pandas Series 或 numpy 数组。

    返回：
        Union[pd.Series, np.ndarray]: 分类结果，可以是 pandas Series 或 numpy 数组。看多的分类结果为 1，看空的分类结果为 -1，其他情况为 0。
    """

    # DIF（快线）上穿 DEA（慢线），同时能量柱由绿转红
    DIF_cross_DEA: Union[pd.Series, np.ndarray] = (dif > dea) & (
        get_shift(dif, 1) < get_shift(dea, 1)
    )
    MACD_green_to_red: Union[pd.Series, np.ndarray] = (hist > 0) & (
        get_shift(hist, 1) < 0
    )
    bullish_zero_zone: Union[pd.Series, np.ndarray] = (
        (dif >= 0) & (dea >= 0) & (hist >= 0)
    )

    # 看多
    bullish: Union[pd.Series, np.ndarray] = (
        DIF_cross_DEA & MACD_green_to_red & bullish_zero_zone
    )

    # DIF（快线）下穿 DEA（慢线），同时能量柱由红转绿
    DEA_cross_DIF: Union[pd.Series, np.ndarray] = (dif < dea) & (
        get_shift(dif, 1) > get_shift(dea, 1)
    )
    MACD_red_to_green: Union[pd.Series, np.ndarray] = (hist < 0) & (
        get_shift(hist, 1) > 0
    )
    bearish_zero_zone: Union[pd.Series, np.ndarray] = (dif < 0) & (dea < 0) & (hist < 0)

    # 看空
    bearish: Union[pd.Series, np.ndarray] = (
        DEA_cross_DIF & MACD_red_to_green & bearish_zero_zone
    )

    return bullish.astype(int) - bearish.astype(int)


def get_macd_signal(
    close_df: pd.DataFrame, keep_pre_status: bool = True
) -> pd.DataFrame:
    """
    根据给定的收盘价数据，计算MACD信号。

    参数：
        - close_df (pd.DataFrame): 包含收盘价数据的DataFrame。
        - keep_pre_status (bool): 是否保留前一状态的信号。默认为True。

    返回：
        - signal (pd.DataFrame): 包含MACD信号的DataFrame。

    示例：
        >>> close_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})
        >>> get_macd_signal(close_df)
           A  B
        0  0  0
        1  1  1
        2  1  1
        3  1  1
        4  1  1
    """

    signal: pd.DataFrame = close_df.apply(
        lambda ser: macd_classify_cols(*MACD(ser)), raw=True
    )
    if keep_pre_status:
        return signal.replace(0, np.nan).ffill().fillna(0)
    return signal

#######################################################################################################
#                                   北向指标计算
#######################################################################################################


def get_north_money_signal(north_money: pd.DataFrame) -> pd.Series:
    """
    根据北向资金数据生成信号
    """
    bottom: pd.Series = north_money["north_money"].rolling(60).quantile(0.2)
    top: pd.Series = north_money["north_money"].rolling(60).quantile(0.8)

    north_signal: pd.Series = pd.Series(
        [np.nan] * len(north_money), index=north_money.index
    )

    north_signal[north_money["north_money"] > top] = 1
    north_signal[north_money["north_money"] < bottom] = -1
    return north_signal.ffill().fillna(0)
"""
Author: Hugo
Date: 2024-08-07 10:36:44
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-08-07 10:39:00
Description: 
"""
from typing import Tuple

import numpy as np
import pandas as pd



def trans_to_entries_exits(signal: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    将信号转换为进出点。

    参数：
        signal (pd.Series): 包含信号的Series对象。

    返回：
        Tuple[pd.Series, pd.Series]: 包含进入点和退出点的元组。

    示例：
        >>> signal = pd.Series([1, 0, 1, -1, 0, 1])
        >>> entries, exits = trans_to_entries_exits(signal)
        >>> entries
        0     True
        1    False
        2     True
        3    False
        4    False
        5     True
        dtype: bool
        >>> exits
        0    False
        1    False
        2    False
        3     True
        4    False
        5    False
        dtype: bool
    """
    if not isinstance(signal, pd.Series):
        raise ValueError("signal必须是pd.Series对象。")
    entries: pd.Series = signal.apply(lambda x: True if x == 1 else False)
    exits: pd.Series = signal.apply(lambda x: True if x == -1 else False)
    return entries, exits

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

"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-03-27 15:02:44
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-03-29 10:50:17
Description: pd rolling table
"""
from typing import Union

import numba as nb
import numpy as np
import pandas as pd

#################### 用于rolling ####################
# https://github.com/bsolomon1124/pyfinance/blob/c6fd88ba4fb5c9f083ebc3ff60960a1b4df76c55/pyfinance/utils.py#L713


@nb.njit
def rolling_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """获取滚动窗口

    Parameters
    ----------
    arr : np.ndarray
        数据
    window : int
        窗口

    Returns
    -------
    np.ndarray
        拆分后的列表
    """
    shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
    windows = np.empty(shape=shape)
    for i in range(shape[0]):
        windows[i] = arr[i : i + window]
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


def rolling_frame(
    df: Union[pd.DataFrame, pd.Series, np.array], window: int
) -> np.ndarray:
    """滚动df

    Parameters
    ----------
    df : Union[pd.DataFrame,pd.Series,np.array]
        数据
    wondows : int
        滚动窗口

    Returns
    -------
    np.ndarray
        结果
    """
    if window > df.shape[0]:
        raise ValueError(
            "Specified `window` length of {0} exceeds length of"
            " `a`, {1}.".format(window, df.shape[0])
        )

    if isinstance(df, (pd.DataFrame, pd.Series)):
        arr: np.array = df.values

    if arr.ndim == 1:
        arr = arr.copy().reshape(-1, 1)

    return rolling_windows(arr, window)

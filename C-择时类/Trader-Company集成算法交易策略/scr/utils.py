'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-08-15 09:13:32
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-08-26 20:35:26
Description:
'''

from typing import List

import numpy as np


def calc_least_squares(exog: np.ndarray,
                       endog: np.ndarray,
                       add_constant: bool = True) -> np.ndarray:

    if len(endog.shape) > 1:
        endog = endog.flatten()

    X: np.ndarray = np.c_[np.ones(len(exog)), exog] if add_constant else exog

    solution = np.linalg.lstsq(X, endog, rcond=None)[0]
    return solution[1:] if add_constant else solution


def rolling_window(data: np.ndarray, window: int) -> List:
    """获取滚动窗口期内的数据

    Args:
        data (np.ndarray): axis 0为日期 1为股票
        window (int): 窗口期

    Returns:
        np.ndarray
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    shape = (data.shape[0] - window + 1, window) + data.shape[1:]
    strides = (data.strides[0], ) + data.strides
    slice_arr = np.squeeze(
        np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides))

    if slice_arr.ndim == 1:
        slice_arr = np.atleast_2d(slice_arr)
    return slice_arr


def calculate_best_chunk_size(data_length: int, n_workers: int) -> int:

    chunk_size, extra = divmod(data_length, n_workers * 5)
    if extra:
        chunk_size += 1
    return chunk_size

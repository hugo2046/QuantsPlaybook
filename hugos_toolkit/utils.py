'''
Author: Hugo
Date: 2024-08-08 13:09:19
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-08-08 13:10:16
Description: 
'''
import numpy as np
from typing import Tuple,Generator


def sliding_window(arr: np.ndarray, window: int, step: int=1) -> Generator:
    """
    滑动窗口生成器函数，用于生成给定数组的滑动窗口。
    :param arr: 输入的数组
    :type arr: np.ndarray
    :param window: 窗口大小，表示多少天
    :type window: int
    :param step: 每天包含的时间点数
    :type step: int
    :return: 生成器对象，每次迭代返回一个滑动窗口
    :rtype: Generator
    """
    if step < 1:
        raise ValueError("step must be greater than 0")

    window_total: int = window * step
    num_windows: int = (arr.shape[0] - window_total + step) // step

    new_shape: Tuple = (
        num_windows,
        window_total,
    ) + arr.shape[1:]
    new_strides: Tuple = (arr.strides[0] * step,) + arr.strides

    sliding_arr = np.lib.stride_tricks.as_strided(
        arr, shape=new_shape, strides=new_strides
    )

    for window in sliding_arr:
        yield window
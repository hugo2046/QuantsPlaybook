'''
Author: Hugo
Date: 2024-10-25 13:16:26
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-06 16:46:18
Description: 通用工具函数
'''


from typing import Generator, Tuple
import numpy as np

# def sliding_window(arr: np.ndarray, window: int, step: int = 1) -> Generator:
#     """
#     滑动窗口生成器函数，用于生成给定数组的滑动窗口。
#     :param arr: 输入的数组
#     :type arr: np.ndarray
#     :param window: 窗口大小，表示多少天
#     :type window: int
#     :param step: 每天包含的时间点数
#     :type step: int
#     :return: 生成器对象，每次迭代返回一个滑动窗口
#     :rtype: Generator
#     """
#     if step < 1:
#         raise ValueError("step must be greater than 0")

#     window_total: int = window * step
#     num_windows: int = (arr.shape[0] - window_total + step) // step

#     new_shape: Tuple = (
#         num_windows,
#         window_total,
#     ) + arr.shape[1:]
#     new_strides: Tuple = (arr.strides[0] * step,) + arr.strides

#     sliding_arr = np.lib.stride_tricks.as_strided(
#         arr, shape=new_shape, strides=new_strides,riteable=False  # 防止意外修改原始数据
#     )

#     for window in sliding_arr:
#         yield window

class SlidingWindowError(Exception):
    """滑动窗口异常基类"""
    pass

class InputTooShortError(SlidingWindowError):
    """输入数据长度不足异常"""
    def __init__(self, n_samples, window):
        super().__init__(
            f"输入数据长度{n_samples}小于窗口长度{window}"
        )
        self.n_samples = n_samples
        self.window = window

class InvalidWindowError(SlidingWindowError):
    """无效窗口参数异常"""
    pass

def sliding_window(
    arr: np.ndarray, 
    window: int, 
    step: int = 1
) -> Generator[np.ndarray, None, None]:
    """生成高效内存视图的滑动窗口
    
    参数:
        arr: 输入数组，形状为(N, ...)
        window: 窗口长度（时间步数）
        step: 窗口滑动步长
    
    生成:
        形状为(window, ...)的窗口视图
    
    异常:
        InputTooShortError: 当输入数据长度小于窗口长度时
        InvalidWindowError: 当窗口参数无效时
    
    示例:
        >>> data = np.arange(5)
        >>> list(sliding_window(data, 3))
        [array([0,1,2]), array([1,2,3]), array([2,3,4])]
    """
    # 参数验证
    if arr.ndim == 0:
        raise ValueError("输入数组维度必须≥1")
    if window < 1:
        raise InvalidWindowError(f"无效窗口长度: {window} (必须≥1)")
    if step < 1:
        raise InvalidWindowError(f"无效步长: {step} (必须≥1)")
    
    n_samples = arr.shape[0]
    if n_samples < window:
        raise InputTooShortError(n_samples, window)
    
    # 计算窗口数量
    num_windows = (n_samples - window) // step + 1
    
    # 构造内存视图
    new_shape = (num_windows, window) + arr.shape[1:]
    new_strides = (arr.strides[0] * step, arr.strides[0]) + arr.strides[1:]
    
    try:
        sliding_view = np.lib.stride_tricks.as_strided(
            arr,
            shape=new_shape,
            strides=new_strides,
            writeable=False
        )
    except ValueError as e:
        raise SlidingWindowError("滑动窗口构造失败") from e
    
    yield from sliding_view
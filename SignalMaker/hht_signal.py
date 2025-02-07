"""
Author: Hugo
Date: 2024-12-13 10:28:42
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-12-13 13:16:23
Description: 
"""

import contextlib
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed,parallel
from PyEMD import EMD
from scipy.signal import hilbert
from tqdm import tqdm
from vmdpy import VMD

from .utils import sliding_window


def calculate_instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """
    计算信号的解析信号和瞬时相位。

    参数:
    signal (np.ndarray): 输入的信号。

    返回:
    np.ndarray: 瞬时相位。
    """
    analytic_signal: np.ndarray = hilbert(signal)

    return np.angle(analytic_signal)


def decompose_signal(
    signal: np.ndarray, method: str = "EMD", max_imf: int = 9
) -> np.ndarray:
    """
    分解信号，使用 EMD 或 VMD。

    参数:
    signal (np.ndarray): 输入的信号。
    method (str): 分解方法，'EMD' 或 'VMD'。
    max_imf (int): 最大的 IMF 数量。

    返回:
    np.ndarray: 分解后的 IMFs。
    """
    if method == "EMD":
        emd = EMD()
        imfs = emd.emd(signal, max_imf=max_imf)
    elif method == "VMD":
        alpha = 2000  # 惩罚因子
        tau = 0.3  # 噪声容忍度
        K = max_imf  # 模态数量
        DC = 0  # 是否包含直流分量
        init = 1  # 初始化方式
        tol = 1e-6  # 收敛容忍度
        imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    else:
        raise ValueError("Invalid method. Choose 'EMD' or 'VMD'.")
    return imfs



def get_ht_binary_signal(differenced: np.ndarray) -> int:
    """
    将平滑+差分后的数据做希尔伯特变换，再查看是否在阈值内，是的话则标记为1，否则为0表示开平仓标记。

    参数:
    differenced (np.ndarray): 输入的差分数据。

    返回:
    int: 如果瞬时相位在阈值范围内，则返回1，否则返回0。
    """

    instantaneous_phase: np.ndarray = calculate_instantaneous_phase(differenced)
    threshold: float = np.pi * 0.5
    return np.where(
        (instantaneous_phase >= -threshold) & (instantaneous_phase <= threshold), 1, 0
    )


def get_ht_signal(
    data: pd.DataFrame, ma_period: int = 60, ht_period: int = 30
) -> pd.DataFrame:
    """
    获取希尔伯特变换信号。

    当相位处于[-pi/2, pi/2]的范围时，价格处于上涨趋势，此时择时信号应为看多。
    由于信号为日频，故我们将成交价格设为次日收盘价。另外，由于价格是非稳态序列，
    希尔伯特变换适用于窄带平稳信号，因此需要对价格数据进行预处理，即平滑（去噪）和差分（去趋势）。

    参数:
    data (pd.DataFrame): 包含价格数据的DataFrame。
    ma_period (int): 移动平均周期，默认为60。
    ht_period (int): 希尔伯特变换周期，默认为30。

    返回:
    pd.DataFrame: 包含二进制信号的DataFrame。
    """
    
    data_: pd.DataFrame = data.copy()
    differenced: pd.Series = data["close"].rolling(ma_period).mean().diff().dropna()
    signal: pd.Series = differenced.rolling(ht_period).apply(
        lambda x: get_ht_binary_signal(x)[-1], raw=True
    )

    data_["binary_signal"] = signal

    return data_


def get_hht_signal(
    data: pd.DataFrame, hht_period: int = 60, imf_index: int = 2, max_imf: int = 9,method:str="EMD"
) -> pd.DataFrame:
    """
    通过 EMD 将价格序列分解为稳态单一频率的 IMFs，再取合适的 IMF 进行希尔伯特变换。
    接下来，我们直接将指数价格序列输入到 EMD 中，分解得到 IMFs，取代表中高频的 IMF3 作为输入进行希尔伯特变换。

    参数:
    data (pd.DataFrame): 包含价格数据的 DataFrame，必须包含 'close' 列。
    hht_period (int): 滚动窗口的周期长度，默认为 60。
    imf_index (int): 选择进行希尔伯特变换的 IMF 索引，默认为 2。下标从0开始
    max_imf (int, 可选): 最大的 IMF 数量，默认为 None。
    method (str, 可选): 分解方法，'EMD' 或 'VMD'，默认为 'EMD'。

    返回:
    pd.DataFrame: 返回包含原始数据和二进制信号的 DataFrame，新增列 'binary_signal' 表示二进制信号。
    """

    data_: pd.DataFrame = data.copy()
    # signal: pd.Series = (
    #     data["close"]
    #     .rolling(hht_period)
    #     .apply(lambda x: get_hht_binary_signal(x, imf_index, max_imf)[-1], raw=True)
    # )
    signal:pd.Series = parallel_apply(data["close"], hht_period, imf_index, max_imf,method)
    data_["binary_signal"] = signal
    return data_


def get_hht_binary_signal(
    close: np.ndarray, imf_index: int = 2, max_imf: int = None, method: str = "EMD"
) -> int:
    """
    获取HHT二进制信号。

    使用EMD分解后的序列做希尔伯特变换，返回指定IMF的二进制信号。

    参数:
    close (np.ndarray): 收盘价序列。
    imf_index (int, 可选): 要使用的IMF索引，默认为2。
    max_imf (int, 可选): 最大IMF数量，默认为None。
    method (str, 可选): 分解方法，'EMD' 或 'VMD'，默认为 'EMD'。

    返回:
    int: 指定IMF的二进制信号。
    """

    imfs: List[np.ndarray] = decompose_signal(
        close, method=method.upper(), max_imf=max_imf
    )

    # 确保IMF数量足够
    if len(imfs) <= imf_index:
        return [0]  # 或者返回其他默认值

    return get_ht_binary_signal(imfs[imf_index])

def get_last_value(func, *args, **kwargs):
    """
    调用部分函数并返回其结果的最后一个值。
    """
    result = func(*args, **kwargs)
    return result[-1]


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tqdm = tqdm_object

        def __call__(self, *args, **kwargs):
            self._tqdm.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def parallel_apply(
    close: pd.Series,window: int=60,imf_index:int=2,max_imf:int=None,method:str="EMD",n_jobs:int=30
) -> pd.Series:
    """
    将 HHT 二进制信号计算函数 apply 到每个滑动窗口上，并将结果concat起来。

    参数:
    close (pd.Series): 收盘价序列
    window (int): 滑动窗口的长度
    imf_index (int): 选择进行希尔伯特变换的 IMF 索引
    max_imf (int): 最大 IMF 数量
    method (str): 分解方法，'EMD' 或 'VMD'
    n_jobs (int): 并行计算的进程数

    返回:
    pd.Series: 包含 HHT 二进制信号的 Series
    """
    func = partial(get_hht_binary_signal, imf_index=imf_index, max_imf=max_imf, method=method)
    windows = list(sliding_window(close.values, window))
    
    # 使用tqdm_joblib包装进度条
    with tqdm_joblib(tqdm(total=len(windows), desc="Processing")):
        signal = Parallel(n_jobs=n_jobs)(
            delayed(get_last_value)(func, ser)
            for ser in windows
        )

    return pd.Series(signal, index=close.index[window - 1:])

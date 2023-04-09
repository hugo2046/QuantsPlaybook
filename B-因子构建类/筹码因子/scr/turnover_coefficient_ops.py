"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-03-27 15:02:44
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-03-29 10:50:17
Description: 历史换手率衰减筹码分布算子
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from qlib.data.ops import PairRolling

from .distribution_of_chips import calc_normalization_turnover
from .utils import rolling_frame


#################### 历史半衰期换手率筹码分布 ####################
def calc_rc(close_arr: np.ndarray) -> np.ndarray:
    """计算相对收益率"""
    if not isinstance(close_arr, np.ndarray):
        raise TypeError("close_arr must be np.ndarray")
    if close_arr.ndim != 1:
        close_arr = close_arr.flatten()
    return np.subtract(1, np.divide(close_arr, close_arr[-1]))


def calc_distribution_of_chips(
    turnover: np.ndarray, close: np.ndarray, N: int
) -> Tuple[np.float64]:
    """计算筹码分布指标"""
    weight: np.ndarray = calc_normalization_turnover(turnover)
    rc: np.ndarray = calc_rc(close)

    # ARC
    arc: np.float64 = np.multiply(weight, rc).sum()

    # VRC
    vrc: np.float64 = N / (N - 1) * (weight * np.square(rc - arc)).sum()

    # SRC
    src: np.float64 = (
        N / (N - 1) * (weight * np.power(rc - arc, 3)).sum() / np.power(vrc, 1.5)
    )

    # KRC
    krc: np.float64 = (
        N / (N - 1) * (weight * np.power(rc - arc, 4)).sum() / np.power(vrc, 2)
    )

    return arc, vrc, src, krc


def calc_roll_cyq(
    turnover: pd.Series, close: pd.Series, N: int, method: str
) -> pd.Series:
    """计算滚动筹码指标

    Args:
        turnover (pd.Series): 换手率
        close (pd.Series): 收盘价
        N (int): 窗口
        method (str): 所需计算指标
            1. ARC ARC>0表示筹码处于平均盈利状态;ARC<0表示筹码处于平均亏损状态
            2. VRC VRC特别大表示筹码特别的分散;VRC特别小表示筹码比较集中;
            3. SRC SRC>0表示筹码分布不对称且右偏,即有一部分投资者的收益特别高;
                   SRC<0表示筹码分布不对称且左偏,即有一部分投资者亏损特别严重;
            4. KRC KRC特别大表示筹码分布里面盈亏分化很强,要么盈利很多,要么亏损很多,
                   处于小赢小亏状态的筹码非常少;
                   KRC特别小则表现盈亏分化很小,筹码处于小赢小亏的状态;
    """
    method: str = method.upper()
    method_dic: Dict = {"ARC": 0, "VRC": 1, "SRC": 2, "KRC": 3}

    turnover_ls: np.ndarray = rolling_frame(turnover, N)
    close_ls: np.ndarray = rolling_frame(close, N)

    idx: pd.Index = turnover.index[N - 1 :]
    ls: List = [
        calc_distribution_of_chips(left, right, N)[method_dic[method]]
        for left, right in zip(turnover_ls, close_ls)
    ]
    return pd.Series(index=idx, dtype=np.float16, data=ls)


#################### 构建算子 ####################
class ARC(PairRolling):
    def __init__(self, feature_left, feature_right, N):

        super(ARC, self).__init__(feature_left, feature_right, N, "Arc")

    def _load_internal(self, instrument, start_index, end_index, *args):

        series_left: pd.Series = self.feature_left.load(
            instrument, start_index, end_index, *args
        )
        series_right: pd.Series = self.feature_right.load(
            instrument, start_index, end_index, *args
        )

        if (series_left.shape[0] < self.N) or (series_right.shape[0] < self.N):
            return pd.Series(dtype=np.float16)

        return calc_roll_cyq(series_left, series_right, self.N, "ARC")


class VRC(PairRolling):
    def __init__(self, feature_left, feature_right, N):

        super(VRC, self).__init__(feature_left, feature_right, N, "Vrc")

    def _load_internal(self, instrument, start_index, end_index, *args):

        series_left: pd.Series = self.feature_left.load(
            instrument, start_index, end_index, *args
        )
        series_right: pd.Series = self.feature_right.load(
            instrument, start_index, end_index, *args
        )
        if (series_left.shape[0] < self.N) or (series_right.shape[0] < self.N):
            return pd.Series(dtype=np.float16)
        return calc_roll_cyq(series_left, series_right, self.N, "VRC")


class SRC(PairRolling):
    def __init__(self, feature_left, feature_right, N):

        super(SRC, self).__init__(feature_left, feature_right, N, "Src")

    def _load_internal(self, instrument, start_index, end_index, *args):

        series_left: pd.Series = self.feature_left.load(
            instrument, start_index, end_index, *args
        )
        series_right: pd.Series = self.feature_right.load(
            instrument, start_index, end_index, *args
        )
        if (series_left.shape[0] < self.N) or (series_right.shape[0] < self.N):
            return pd.Series(dtype=np.float16)
        return calc_roll_cyq(series_left, series_right, self.N, "SRC")


class KRC(PairRolling):
    def __init__(self, feature_left, feature_right, N):

        super(KRC, self).__init__(feature_left, feature_right, N, "Krc")

    def _load_internal(self, instrument, start_index, end_index, *args):

        series_left: pd.Series = self.feature_left.load(
            instrument, start_index, end_index, *args
        )
        series_right: pd.Series = self.feature_right.load(
            instrument, start_index, end_index, *args
        )
        if (series_left.shape[0] < self.N) or (series_right.shape[0] < self.N):
            return pd.Series(dtype=np.float16)
        return calc_roll_cyq(series_left, series_right, self.N, "KRC")

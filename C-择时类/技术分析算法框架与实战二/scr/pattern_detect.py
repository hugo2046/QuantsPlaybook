"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-02-07 15:55:45
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-02-10 10:41:59
Description: 
"""
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
import talib
from loguru import logger
from scipy.signal import argrelmax, argrelmin

from .utils import calc_smooth


def _find_previous_max_idx(min_v: np.ndarray, max_v: np.ndarray) -> int:
    "获取低点前的一个高点"
    current_min_p: int = min_v[-1]
    current_max_p: int = max_v[-1]

    return current_max_p if current_max_p < current_min_p else max_v[-2]


class RoundingBottom:
    def __init__(
        self,
        code: str,
        ohlc: pd.DataFrame,
        /,
        *,
        bw: Union[str, float, int] = 1.5,
        smooth_func: Union[Callable, pd.Series] = calc_smooth,
        threshold_d: float = 10,
        threshold_s: float = 0.4,
        threshold_r: float = 0.03,
    ) -> None:

        self.code = code
        self.ohlc = ohlc
        self.bw = bw
        self.smooth_func = smooth_func
        self.threshold_d = threshold_d
        self.threshold_s = threshold_s
        self.threshold_r = threshold_r

    @property
    def get_key_feature(self) -> Dict:
        """获取关键特征"""
        close_ser: pd.Series = self.ohlc["close"].fillna(method="ffill")

        # 平滑收盘价用于确定Max-Min
        if isinstance(self.smooth_func, Callable):
            smooth: pd.Series = self.smooth_func(close_ser, self.bw)
        elif isinstance(self.smooth_func, pd.Series):
            smooth: pd.Series = self.smooth_func
        else:
            raise ValueError("smooth_func仅能为Callable或pd.Series!")

        current_idx: int = len(self.ohlc)  # 当前点
        p_current: float = close_ser.iloc[-1]  # 当前价格

        # 获取高点/低点
        min_v: pd.Series = argrelmin(smooth.values)[0]
        max_v: pd.Series = argrelmax(smooth.values)[0]

        # 获取离当前点最近的低点
        local_min_idx: int = min_v[-1]
        # 获取低点前的一个高点
        previous_max_idx: int = _find_previous_max_idx(min_v, max_v)

        if previous_max_idx > local_min_idx:
            raise ValueError("p_max在p_min点前!")

        # 获取最低价和最低价前面的最高价
        p_min: float = close_ser.iloc[local_min_idx]
        p_max: float = close_ser.iloc[previous_max_idx:local_min_idx].max()
        # 靠近p_current点的最高点
        p_right_max: float = close_ser.iloc[local_min_idx:current_idx].max()

        # local_max_idx有可能不等于previous_max_idx
        local_max_idx = self.ohlc.index.get_loc(
            close_ser.iloc[previous_max_idx:local_min_idx].idxmax()
        )

        # 左半段高点到低点的距离 该长度需要大于(threshold_d)
        d_left: int = local_min_idx - local_max_idx
        # 右半段低点到当前点的距离
        d_right: int = current_idx - local_min_idx

        # s在p_min左(右)侧下跌(上涨)的点所占比例大于某一参数(threshold_s)
        pct_chg: pd.Series = close_ser.pct_change()
        s_left: pd.Series = pct_chg.iloc[local_max_idx:local_min_idx]
        s_right: pd.Series = pct_chg.iloc[local_min_idx:current_idx]

        s_left_size: float = len(s_left)
        s_right_size: float = len(s_right)

        # s表示涨跌幅占比
        s_left_updown_ratio: float = (
            len(s_left[s_left < 0]) / s_left_size if s_left_size else 0
        )
        s_right_updown_ratio: float = (
            len(s_right[s_right > 0]) / s_right_size if s_right_size else 0
        )

        # r的均值小于某一参数平均收益(threshold_r)
        r_left_pct_avg: float = np.abs(s_left.mean())
        r_right_pct_vag: float = np.abs(s_right.mean())

        # (p_max/p_low-1)/d
        p_left: float = (p_max / p_min - 1) / d_left
        # (p_current/p_low-1)/d
        p_right: float = (p_current / p_min - 1) / d_right

        # 关键位置
        self.key_position: Dict = {
            "local_max_idx": local_max_idx,  # 平滑后的定位
            "local_min_idx": local_min_idx,  # 平滑后的定位
            "previous_max_idx": previous_max_idx,  # 价格中的定位
            "current_idx": current_idx,  # 当前值定位
        }
        return {
            "code": self.code,
            "p_current": p_current,  # 当前点
            "p_right_max": p_right_max,  # 当前点前一个高点
            "p_max": p_max,  # 最低点前一个高点
            "p_min": p_min,  # 最低点
            "d_left": d_left,  # 高点到低点的距离
            "d_right": d_right,  # 低点到当前点的距离
            "s_left_updown_ratio": s_left_updown_ratio,  # 左侧下跌占比
            "s_right_updown_ratio": s_right_updown_ratio,  # 右侧上涨占比
            "r_left_pct_avg": r_left_pct_avg,  # 左侧平均涨跌
            "r_right_pct_avg": r_right_pct_vag,  # 右侧平均涨跌
            "p_left": p_left,  # 左侧每日涨跌
            "p_right": p_right,  # 右侧每日涨跌
        }

    def is_pattern(self) -> bool:

        self.feature: Dict = self.get_key_feature

        # 当前价不能低于右侧曲线的最高价且小于等于p_high
        # 否则会有杯柄形态出现
        cond_a: bool = (self.feature["p_current"] >= self.feature["p_right_max"]) & (
            self.feature["p_current"] <= self.feature["p_max"]
        )

        # 左半弧与右半弧与p_min的距离需要大于threshold_d
        cond_b: bool = (self.feature["d_left"] > self.threshold_d) & (
            self.feature["d_right"] > self.threshold_d
        )
        # 左半弧与右半弧涨跌幅占比需要大于threshold_s
        cond_c: bool = (self.feature["s_left_updown_ratio"] > self.threshold_s) & (
            self.feature["s_right_updown_ratio"] > self.threshold_s
        )

        # r的均值小于某一参数
        cond_d: bool = (self.feature["r_left_pct_avg"] < self.threshold_r) & (
            self.feature["r_right_pct_avg"] < self.threshold_r
        )
        # (p_max/p_min−1)/d、(p_current/p_low-1)/d小于某一参数(本文取0.03)
        cond_e: bool = (self.feature["p_left"] < self.threshold_r) & (
            self.feature["p_right"] < self.threshold_r
        )

        return cond_a & cond_b & cond_c & cond_d & cond_e


def find_RoundingBottom(
    ohlc: pd.DataFrame,
    window: int = 200,
    /,
    *,
    threshold_size: int = 252,
    code: str = None,
    bw: Union[str, float, int] = 1.5,
) -> Tuple[bool, bool]:
    """寻找圆弧低

    Args:
        ohlc (pd.DataFrame): index-date open|high|low|close

    Returns:
        bool: 1-是否满足形态;2-是否符合买卖点
    """

    if len(ohlc) <= threshold_size:
        # rounding_bottom_loger.info(f"{code} ohlc数据长度低于252无法识别")
        return False

    # FIND_MAX_D: int = 20  # 局部最低点的qian
    THRESHOLD_D: int = 10  # 半弦d的阈值
    THRESHOLD_S: float = 0.4  # 下跌/上涨的占比比例
    THRESHOLD_R: float = 0.03  # 下跌/上涨的平均收益
    THRSHOLD_P: float = 0.3  # 当前点涨幅在下跌涨幅的70%~130%之间

    close_ser: pd.Series = ohlc["close"].fillna(method="ffill")
    # 平滑
    smooth: pd.Series = calc_smooth(close_ser, bw)
    # 均线
    ma200_ser: pd.Series = talib.EMA(close_ser, window)
    ma200_current: float = ma200_ser.iloc[-1]

    # 标记低点
    min_v: pd.Series = argrelmin(smooth.values)[0]
    # 标记高点
    max_v: pd.Series = argrelmax(smooth.values)[0]

    current_idx: int = len(ohlc)  # 当前点
    p_current: float = close_ser.iloc[-1]  # 当前价格

    local_min_idx: int = min_v[-1]  # 获取离当前点最近的低点
    # 获取低点前的一个高点
    previous_max_idx: int = _find_previous_max_idx(min_v, max_v)

    if previous_max_idx > local_min_idx:
        raise ValueError("p_max在p_min点前!")

    # 获取最低价和最低价前面的最高价
    p_min: float = close_ser.iloc[local_min_idx]
    p_max: float = close_ser.iloc[previous_max_idx:local_min_idx].max()

    # local_max_idx有可能不等于previous_max_idx
    local_max_idx = ohlc.index.get_loc(
        close_ser.iloc[previous_max_idx:local_min_idx].idxmax()
    )

    p_right_max: float = close_ser.iloc[local_min_idx:current_idx].max()

    d_left: int = local_min_idx - local_max_idx  # 半弦长
    d_right: int = current_idx - local_min_idx

    # s
    pct_chg: pd.Series = close_ser.pct_change()
    s_left: pd.Series = pct_chg.iloc[local_max_idx:local_min_idx]
    s_right: pd.Series = pct_chg.iloc[local_min_idx:current_idx]

    s_left_size: float = len(s_left)
    s_right_size: float = len(s_right)

    s_left_ratio: float = len(s_left[s_left < 0]) / s_left_size if s_left_size else 0
    s_right_ratio: float = (
        len(s_right[s_right > 0]) / s_right_size if s_right_size else 0
    )

    r_left: float = np.abs(s_left.mean())
    r_right: float = np.abs(s_right.mean())
    # (p_max/p_low-1)/d
    p_left: float = (p_max / p_min - 1) / d_left
    # (p_current/p_low-1)/d
    p_right: float = (p_current / p_min - 1) / d_right

    # 当前价不能低于右侧曲线的最高价且小于等于p_high
    # 否则会有杯柄形态出现
    cond_x1: bool = (p_current >= p_right_max) & (p_current <= p_max)
    cond_a: bool = (d_left > THRESHOLD_D) & (d_right > THRESHOLD_D)
    cond_b: bool = (s_left_ratio > THRESHOLD_S) & (s_right_ratio > THRESHOLD_S)
    # r的均值小于某一参数
    cond_c: bool = (r_left < THRESHOLD_R) & (r_right < THRESHOLD_R)
    # (p_max/p_min−1)/d、(p_current/p_low-1)/d小于某一参数(本文取0.03)
    cond_d: bool = (p_left < THRESHOLD_R) & (p_right < THRESHOLD_R)

    cond_pattern: bool = cond_x1 & cond_a & cond_b & cond_c & cond_d
    # -------上述确定是否为圆弧低形态-------

    # 判断是否是买点
    # 大于200日均线
    cond_e: bool = p_current >= ma200_current
    cond_f: bool = (
        (1 - THRSHOLD_P) * (p_max - p_min)
        < p_current - p_min
        < (1 + THRSHOLD_P) * (p_max - p_min)
    )

    current_dt: str = ohlc.index[-1].strftime("%Y-%m-%d")
    if cond_pattern:
        logger.info(
            f"当前时点:{current_dt} {code}=>符合圆弧底/v型结构:{cond_pattern},判断是否是买点:1.是否大于200日均线[{cond_e}];2.符合在p_high至p_low的30%~130%区间内[{cond_f}]"
        )

    all_count: bool = cond_pattern & cond_e & cond_f
    if all_count:
        logger.info(f"当前时点:{current_dt} {code}:为圆弧底/V型低且符合买点")
    return cond_pattern, cond_e & cond_f

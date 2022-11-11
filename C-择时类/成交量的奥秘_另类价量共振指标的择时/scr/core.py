'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-11-10 14:45:07
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-11-11 14:51:05
Description: 
'''
from typing import Tuple

import numpy as np
import pandas as pd
import talib


# 构造HMA
def HMA(price: pd.Series, window: int) -> pd.Series:
    """HMA均线

    Args:
        price (pd.Series): 价格数据. index-date values
        window (int): 计算窗口

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values
    """
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    return talib.WMA(
        2 * talib.WMA(price, int(window * 0.5)) - talib.WMA(price, window),
        int(np.sqrt(window)))


def calc_volume_momentum(close: pd.Series,
                         volume: pd.Series,
                         bma_window: int = 50,
                         ama_window: int = 100,
                         n_window: int = 3) -> pd.Series:
    """价量共振指标

    Args:
        close (pd.Series): index-date values-close
        volume (pd.Series): index-date values-volumne
        bma_window (int, optional): BMA的计算窗口期. Defaults to 50.
        ama_window (int, optional): 量能指标的分母项-AMA的计算窗口期. Defaults to 100.
        n_window (int, optional): BMA的分母滞后期. Defaults to 3.

    Returns:
        pd.Series: index-date values-价量指标
    """
    close, volume = close.align(volume, axis=0)

    # 价能指标
    BMA: pd.Series = HMA(close, bma_window)

    price_mom: pd.Series = BMA / BMA.shift(n_window)

    # 量能指标
    vol_mom: pd.Series = HMA(volume, 5) / HMA(volume, ama_window)

    # 价量共振指标
    return price_mom * vol_mom


def get_trendshake_filter(close: pd.Series, fast_window: int, slow_window: int,
                          threshold: Tuple[float]) -> pd.Series:
    """市场划分条件下的择时系统

    Args:
        close (pd.Series): index-date value-close
        fast_window (int): 短周期计算窗口
        slow_window (int): 长周期计算窗口
        threshold (Tuple[float]): 多头市场阈值;空调市场阈值

    Returns:
        pd.Series: _description_
    """
    threshold1, threshold2 = threshold
    # 市场划分
    filter_shake: pd.Series = (close.rolling(fast_window).mean() >
                               close.rolling(slow_window).mean())

    return filter_shake.apply(lambda x: np.where(x, threshold1, threshold2))

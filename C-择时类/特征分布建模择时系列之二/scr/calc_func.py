'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-22 13:22:49
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-09-01 14:32:55
Description: 
'''

import numpy as np
import pandas as pd
import talib


def create_signal(data: pd.DataFrame, fast_window: int = 5, slow_window: int = 100, start_dt: str = None, end_dt: str = None, threshold: float = 1.15, a: float = 1.5) -> pd.DataFrame:
    df = data.copy()
    # if window:
    #     AMA: pd.Series = HMA(df['volume'],window)
    # else:
    #     AMA: pd.Series = df['volume']
    df['volume_index']: pd.Series = HMA(
        df['volume'], fast_window) / HMA(df['volume'], slow_window)
    df['forward_returns'] = df['close'].pct_change(5).shift(-5)
    df['threshold_to_long_a'] = threshold
    df['threshold_to_long_b'] = np.power(threshold, -a)
    df['threshold_to_short'] = 1
    return df if (start_dt is None) and (end_dt is None) else df.loc[start_dt:end_dt]


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

    return talib.WMA(2 * talib.WMA(price, int(window * 0.5)) - talib.WMA(price, window), int(np.sqrt(window)))

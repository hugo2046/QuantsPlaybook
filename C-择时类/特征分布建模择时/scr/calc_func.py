'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-22 13:22:49
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-22 13:23:10
Description: 
'''
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

    hma = talib.WMA(
        2 * talib.WMA(price, int(window * 0.5)) - talib.WMA(price, window),
        int(np.sqrt(window)))

    return hma

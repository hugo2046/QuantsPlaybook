'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-22 13:22:49
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-23 13:21:19
Description: 
'''
from typing import List

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

def get_forward_returns(price:pd.Series,periods:List)->pd.DataFrame:
    """获取远期收益率

    Args:
        price (pd.Series): 价格序列
        periods (List): 周期

    Returns:
        pd.DataFrame: 收益
    """
    pct_chg:pd.DataFrame = pd.concat((price.pct_change(i).shift(-i) for i in periods),axis=1)
    pct_chg.columns = ['%s日收益'%i for i in periods]
    
    return pct_chg.dropna()

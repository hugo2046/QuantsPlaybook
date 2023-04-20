'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-04-20 14:33:27
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-04-20 14:52:49
Description: 

siegelslope_ma:重复中位数(RM)下的稳健回归
calc_icu_ma:ICU均线
'''
from typing import Union

import numpy as np
import pandas as pd


def siegelslopes_ma(price_ser: Union[pd.Series, np.ndarray],method:str="hierarchical") -> float:
    """Repeated Median (Siegel 1982)

    Args:
        price_ser (Union[pd.Series, np.ndarray]): index-date values-price or values-price

    Returns:
        float: float
    """
    from scipy import stats
    n: int = len(price_ser)
    res = stats.siegelslopes(price_ser, np.arange(n), method=method)
    return res.intercept + res.slope * (n-1)


def calc_icu_ma(price:pd.Series,N:int)->pd.Series:
    """计算ICU均线

    Args:
        price (pd.Series): index-date values-price
        N (int): 计算窗口
    Returns:
        pd.Series: index-date values-icu_ma
    """
    if len(price) <= N:
        raise ValueError("price length must be greater than N")
    
    return price.rolling(N).apply(siegelslopes_ma,raw=True)
'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-11-11 16:53:12
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-11-13 21:52:14
Description: 
'''
from typing import Dict

import pandas as pd


def trans2strftime(ser: pd.Series, fmt: str = '%Y-%m-%d') -> pd.Series:

    return pd.to_datetime(ser).dt.strftime(fmt)


# 回测参数
BACKTEST_CONFIG: Dict = dict(n=3, threshold= (1.125, 1.275),
                             bma_window=50, ama_window=100, fast_window=5, slow_window=90)

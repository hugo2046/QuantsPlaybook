'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-22 13:22:49
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-08-31 20:25:36
Description: 
'''

import numpy as np
import pandas as pd


def create_signal(data: pd.DataFrame, window: int, start_dt: str = None, end_dt: str = None, threshold: float = 1.15, a: float = 1.5) -> pd.DataFrame:
    df = data.copy()
    if window:
        AMA: pd.Series = df['volume'].rolling(window).mean()
    else:
        AMA: pd.Series = df['volume']
    df['volume_index']: pd.Series = AMA.rolling(5).mean() / AMA.rolling(100).mean()
    df['forward_returns'] = df['close'].pct_change(5).shift(-5)
    df['threshold_to_long_a'] = threshold
    df['threshold_to_long_b'] = np.power(threshold, -a)
    df['threshold_to_short'] = 1
    return df if (start_dt is None) and (end_dt is None) else df.loc[start_dt:end_dt]
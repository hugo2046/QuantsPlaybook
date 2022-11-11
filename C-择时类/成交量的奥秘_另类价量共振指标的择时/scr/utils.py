'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-11-11 16:53:12
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-11-11 17:07:15
Description: 
'''
import pandas as pd


def trans2strftime(ser: pd.Series, fmt: str = '%Y-%m-%d') -> pd.Series:

    return pd.to_datetime(ser).dt.strftime(fmt)
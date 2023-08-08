"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-08-02 11:11:45
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-08-02 11:13:59
Description: 
"""
import datetime as dt
from typing import Union
import pandas as pd


def str2date(date_str: str) -> pd.DatetimeIndex:
    return pd.to_datetime(date_str)


def format_date(
    date: Union[pd.Timestamp, dt.datetime, dt.date], fm: str = "%Y%m%d"
) -> str:
    date: pd.DatetimeIndex = pd.to_datetime(date)
    return date.strftime(fm)




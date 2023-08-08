"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-08-02 11:06:41
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-08-02 11:11:14
Description: 
"""
from functools import lru_cache
import pandas as pd
import numpy as np
from .tushare_api import TuShare
from .utils import format_date
from typing import Union
import datetime as dt


@lru_cache()
def get_trade_cal_frame() -> pd.DataFrame:
    my_ts = TuShare()
    cal_frame: pd.DataFrame = my_ts.trade_cal(exchange="SSE").sort_values("cal_date")
    cal_frame: pd.DataFrame = cal_frame.astype(
        {"cal_date": "datetime64[D]", "pretrade_date": "datetime64[D]"}
    )
    cal_frame.index = cal_frame["cal_date"]
    return cal_frame


@lru_cache()
def get_all_trade_days() -> pd.DatetimeIndex:
    trade_cal_frame: pd.DataFrame = get_trade_cal_frame()
    trade_cal_frame: np.ndarray = (
        trade_cal_frame.query("is_open==1")["cal_date"].sort_values().values
    )
    return pd.to_datetime(trade_cal_frame)


def get_trade_days(
    start_dt: Union[str, dt.datetime, dt.date], end_dt: Union[str, dt.datetime, dt.date]
) -> pd.DatetimeIndex:
    start_dt: str = format_date(start_dt, "%Y%m%d")
    end_dt: str = format_date(end_dt, "%Y%m%d")

    trade_cal: pd.DataFrame = get_trade_cal_frame().copy()
    trade_cal: pd.DataFrame = trade_cal.loc[start_dt:end_dt]
    trade_cal: pd.DataFrame = trade_cal.query("is_open==1")
    return pd.to_datetime(trade_cal["cal_date"].values)

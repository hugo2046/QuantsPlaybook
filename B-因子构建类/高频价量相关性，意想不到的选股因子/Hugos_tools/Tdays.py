from typing import (Tuple, List, Union, Callable,Dict,Any)
import functools
import pandas as pd
from jqdata import *


@functools.lru_cache()
def jq_all_trade_days() -> pd.DatetimeIndex:
    """使用jq接口获取交易日期
       从2005年起
    Returns:
        pd.DatetimeIndex: 交易日期
    """

    return pd.to_datetime(get_all_trade_days())


def create_cal(trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """构造交易日历

    Args:
        trade_dates (pd.DatatimeIndex, optional): 交易日. Defaults to None.

    Returns:
        pd.DataFrame: 交易日历表
    """
    
    min_date = trade_dates.min()
    max_date = trade_dates.max()

    dates = pd.date_range(min_date, max_date)
    df = pd.DataFrame(index=dates)

    df['is_tradeday'] = False
    df.loc[trade_dates, 'is_tradeday'] = True

    return df


def Tdaysoffset(watch_date: str, count: int, freq: str = None) -> pd.Timestamp:
    """日期偏移

    Args:
        watch_date (str): 观察日
        count (int): 偏离日
        freq (str):频率,D-日度,W-周度,M-月份,Y-年度
    Returns:
        dt.datetime: 目标日期
    """
    
    if isinstance(watch_date, str):
        watch_date = pd.to_datetime(watch_date)

    all_trade_days = get_all_trade_days()
    cal_frame = create_cal(trade_dates=all_trade_days)

    holiday = cal_frame.query('not is_tradeday').index
    trade_days = pd.offsets.CustomBusinessDay(weekmask='1'*7, holidays=holiday)

    if freq is None:

        target = watch_date + trade_days * 0 + trade_days * count

    else:

        target = watch_date + pd.DateOffset(**{freq: count}) + trade_days

    return target

# 获取年末季末时点
def get_trade_period(start_date: str, end_date: str, freq: str = 'ME') -> list:
    '''
    start_date/end_date:str YYYY-MM-DD
    freq:M月，Q季,Y年 默认ME E代表期末 S代表期初
    ================
    return  list[datetime.date]
    '''
    days = pd.Index(pd.to_datetime(get_trade_days(start_date, end_date)))
    idx_df = days.to_frame()

    if freq[-1] == 'E':
        day_range = idx_df.resample(freq[0]).last()
    else:
        day_range = idx_df.resample(freq[0]).first()

    day_range = day_range[0].dt.date

    return day_range.dropna().values.tolist()
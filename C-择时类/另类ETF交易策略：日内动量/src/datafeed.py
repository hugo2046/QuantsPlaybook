'''
Author: Hugo
Date: 2024-08-12 14:21:52
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-08-19 15:55:24
Description: 
'''
from typing import List, Tuple

import backtrader as bt
from backtrader.feeds import PandasDirectData

__all__ = ["ETFDataFeed"]

class ETFDataFeed(PandasDirectData):
    """
    OHLC 为后复权

    datetime必须为datetime64[ns]类型，其他字段不支int,float以外类型
    """

    params: Tuple[Tuple] = (
        ("datetime", 0),
        ("open", 1),
        ("high", 2),
        ("low", 3),
        ("close", 4),
        ("volume", 5),
        ("amount", 6),
        ("upperbound",7), # 上轨
        ("signal",8), # 信号
        ("lowerbound",9), # 下轨
        ("vwap",10), # vwap
        ("dtformat","%Y-%m-%d %H:%M:%S"),
        ("timeframe",bt.TimeFrame.Minutes),
    )

    lines: List[str] = (
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "upperbound",
        "signal",
        "lowerbound",
        "vwap",
    )
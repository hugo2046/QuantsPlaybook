'''
Author: Hugo
Date: 2024-08-12 14:21:52
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-10-29 13:19:40
Description: 加载数据
'''
from typing import List, Tuple

from backtrader.feeds import PandasDirectData

__all__ = ["ETFDataFeed"]

class DailyOHLCVUSLFeed(PandasDirectData):
    """
    OHLC 为后复权
    V-Volume
    U-Ubound
    S-Signal
    L-Lbound

    datetime必须为datetime64[ns]类型，其他字段不支int,float以外类型
    """

    params: Tuple[Tuple] = (
        ("datetime", 0),
        ("open", 1),
        ("high", 2),
        ("low", 3),
        ("close", 4),
        ("volume", 5),
        ("upperbound",6), # 上轨
        ("signal",7), # 信号
        ("lowerbound",8), # 下轨
       
    )

    lines: List[str] = (
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "upperbound",
        "signal",
        "lowerbound",
    )
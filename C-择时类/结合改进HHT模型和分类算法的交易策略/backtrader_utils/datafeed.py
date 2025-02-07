'''
Author: Hugo
Date: 2024-08-12 14:21:52
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-12-13 10:28:02
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
        ("binary_signal",6), # 0,1信号1-买入/持有，0-卖出/空仓
       
    )

    lines: List[str] = (
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "binary_signal",
    )
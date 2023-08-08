from typing import Callable, List

import pandas as pd

from .trade_cal import get_trade_days
from .tushare_api import TuShare


def distributed_query(
    query_func_name: Callable,
    symbol: str,
    start_date: str,
    end_date: str,
    limit=100000,
    **kwargs
):
    my_ts = TuShare()
    if isinstance(symbol, str):
        n_symbols: int = len(symbol.split(","))
    elif isinstance(symbol, list):
        n_symbols: int = len(symbol)
        symbol: str = ",".join(symbol)
    else:
        raise ValueError("symbol must be str or list")
    dates: pd.DatetimeIndex = get_trade_days(start_date, end_date)
    dates: pd.DatetimeIndex = dates.strftime("%Y%m%d")
    n_days: int = len(dates)

    if n_symbols * n_days > limit:
        n: int = limit // n_symbols

        df_list: List = []
        i: int = 0
        pos1, pos2 = n * i, n * (i + 1) - 1
        while pos2 < n_days:
            df: pd.DataFrame = getattr(my_ts, query_func_name)(
                ts_code=symbol, start_date=dates[pos1], end_date=dates[pos2], **kwargs
            )
            df_list.append(df)
            i += 1
            pos1, pos2 = n * i, n * (i + 1) - 1
        if pos1 < n_days:
            df = getattr(my_ts, query_func_name)(
                ts_code=symbol, start_date=dates[pos1], end_date=dates[-1], **kwargs
            )
            df_list.append(df)
        df: pd.DataFrame = pd.concat(df_list, axis=0)
    else:
        df: pd.DataFrame = getattr(my_ts, query_func_name)(
            ts_code=symbol, start_date=start_date, end_date=end_date, **kwargs
        )
    return df

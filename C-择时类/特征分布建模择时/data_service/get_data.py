'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-22 10:22:58
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-27 16:14:44
Description: 数据获取
'''

import functools
from typing import Callable, List

import numpy as np
import pandas as pd
from scr import Tdaysoffset, get_trade_days
from scr.tushare_api import TuShare
from scr.utils import format_dt

my_ts = TuShare()

from tqdm.notebook import tqdm


def check_query_date_params(func):
    """参数检查

    Args:
        func (_type_): _description_

    Returns:
        _type_: _description_
    """
    @functools.wraps(func)
    def wrapper(*args, **kws):

        kws = _check_query_date_params(*args, **kws)
        df = _prepare_format(func(**kws))
        return df

    return wrapper


def offset_limit(limit: int):
    """过滤最大数量的限制"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kws):

            kws = _check_query_date_params(*args, **kws)
            kws['limit'] = limit
            df = distributed_query(func, **kws)

            return _prepare_format(df)

        return wrapper

    return decorator


def distributed_query(query_func_name: Callable, code: str, start_date: str,
                      end_date: str, limit: int, **kwargs) -> pd.DataFrame:
    """绕过数量限制

    Args:
        query_func_name (Callable): _description_
        code (str): _description_
        start_date (str): _description_
        end_date (str): _description_
        limit (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    n_symbols = len(code.split(','))
    dates = get_trade_days(start_date, end_date)
    n_days = len(dates)

    if n_symbols * n_days > limit:
        n = limit // n_symbols

        df_list = []
        i = 0
        pos1, pos2 = n * i, n * (i + 1) - 1
        while pos2 < n_days:

            df = query_func_name(code=code,
                                 start_date=dates[pos1],
                                 end_date=dates[pos2],
                                 **kwargs)
            df_list.append(df)
            i += 1
            pos1, pos2 = n * i, n * (i + 1) - 1
        if pos1 < n_days:
            df = query_func_name(code=code,
                                 start_date=dates[pos1],
                                 end_date=dates[-1],
                                 **kwargs)
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
    else:
        df = query_func_name(code,
                             start_date=start_date,
                             end_date=end_date,
                             **kwargs)
    return df


def _prepare_format(df: pd.DataFrame) -> pd.DataFrame:
    """ts数据格式预处理

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dtype_mapping = {"trade_date": np.datetime64}
    col_mapping = {'ts_code': 'code'}

    return (df.pipe(pd.DataFrame.astype,
                    dtype_mapping).pipe(pd.DataFrame.rename,
                                        columns=col_mapping).pipe(
                                            pd.DataFrame.sort_values,
                                            by='trade_date'))


def _check_query_date_params(start_date: str = None,
                             end_date: str = None,
                             count: int = None,
                             **kws):
    """_summary_

    Args:
        start_date (str, optional): _description_. Defaults to None.
        end_date (str, optional): _description_. Defaults to None.
        count (int, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if (count is not None) and (start_date is not None):
        raise ValueError("不能同时指定 start_date 和 count 两个参数")

    if count is not None:

        end_date = pd.to_datetime(end_date)
        count = int(-count)
        start_date = Tdaysoffset(end_date, count)
    else:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

    out_params = {
        'start_date': format_dt(start_date, '%Y%m%d'),
        'end_date': format_dt(end_date, '%Y%m%d'),
    }

    out_params.update(**kws)

    return out_params


@check_query_date_params
def get_sales_depart_billboard(start_date: str = None,
                               end_date: str = None,
                               count: int = None) -> pd.DataFrame:
    """获取机构龙虎榜

    Args:
        start_date (str, optional): 起始日. Defaults to None.
        end_date (str, optional): 结束日. Defaults to None.
        count (int, optional): 周期. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    # 获取周期
    periods = get_trade_days(start_date, end_date)
    periods = periods.strftime('%Y%m%d')

    dfs: List = []
    for trade in tqdm(periods, desc='机构龙虎榜数据获取'):

        dfs.append(my_ts.top_inst(trade_date=trade))

    billboard_frame: pd.DataFrame = pd.concat(dfs)

    billboard_frame['trade_date'] = pd.to_datetime(
        billboard_frame['trade_date'])

    return billboard_frame


@offset_limit(8000)
def get_index_daily(code: str,
                    start_date: str = None,
                    end_date: str = None,
                    count: int = None) -> pd.DataFrame:

    periods = get_trade_days(start_date, end_date)
    data = my_ts.index_daily(ts_code=code,
                             start_date=start_date,
                             end_date=end_date)

    return data

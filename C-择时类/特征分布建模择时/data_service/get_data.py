'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-22 10:22:58
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-22 16:48:33
Description: 数据获取
'''

import functools
from typing import List

import numpy as np
import pandas as pd
from scr import Tdaysoffset, get_trade_days
from scr.tushare_api import TuShare
from scr.utils import format_dt

my_ts = TuShare()

from tqdm.notebook import tqdm


def check_query_date_params(func):
    @functools.wraps(func)
    def wrapper(*args, **kws):

        kws = _check_query_date_params(*args, **kws)
        return func(**kws)

    return wrapper


def _check_query_date_params(start_date: str = None,
                             end_date: str = None,
                             count: int = None,
                             **kws):

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


@check_query_date_params
def get_index_daily(code: str,
                    start_date: str = None,
                    end_date: str = None,
                    count: int = None) -> pd.DataFrame:

    data = my_ts.index_daily(ts_code=code,
                             start_date=start_date,
                             end_date=end_date)

    dtype_mapping = {"trade_date": np.datetime64}
    return (data.pipe(pd.DataFrame.astype,
                      dtype_mapping).pipe(pd.DataFrame.set_index,
                                          keys='trade_date').pipe(
                                              pd.DataFrame.rename,
                                              columns={
                                                  'ts_code': 'code'
                                              }).pipe(pd.DataFrame.sort_index,
                                                      ascending=True))

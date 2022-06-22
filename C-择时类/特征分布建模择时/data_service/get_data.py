'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-22 10:22:58
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-22 11:31:05
Description: 数据获取
'''

from typing import List

import pandas as pd
from scr import Tdaysoffset, get_trade_days
from scr.tushare_api import TuShare

my_ts = TuShare()

from tqdm.notebook import tqdm


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

    if (count is not None) and (start_date is not None):
        raise ValueError("不能同时指定 start_date 和 count 两个参数")

    if count is not None:

        end_date = pd.to_datetime(end_date)
        count = int(-count)
        start_date = Tdaysoffset(end_date, count)
    else:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

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

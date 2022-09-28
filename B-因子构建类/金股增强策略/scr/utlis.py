from typing import List, Union

import numpy as np
import pandas as pd
from jqdata import *


def load_gold_stock_csv() -> pd.DataFrame:
    """读取金股csv数据文件

    Returns
    -------
    pd.DataFrame
    """
    dtype_mapping = {'end_date': np.datetime64, 'write_date': np.datetime64}

    col_mapping = {
        'ticker_symbol_map_sec_type_name': 'sw_l3',
        'ticker_symbol_map_sec_id': 'code'
    }

    gold_stock_frame = pd.read_csv(r'data/gold_stock_frame.csv', index_col=[0])

    gold_stock_frame = (gold_stock_frame.pipe(
        pd.DataFrame.astype,
        dtype_mapping).pipe(pd.DataFrame.rename, columns=col_mapping).pipe(
            pd.DataFrame.assign,
            code=lambda x: x['code'].apply(normalize_code)))

    return gold_stock_frame


def view_author_stock(ser: pd.Series,
                      gold_stock_frame: pd.DataFrame) -> pd.DataFrame:
    """从gold_stock_frame按ser获取所推荐股票信息

    Parameters
    ----------
    ser : pd.Series
        MultiIndex level0-date level1-code values-proba
    gold_stock_frame : pd.DataFrame
        金股数据表

    Returns
    -------
    pd.DataFrame
        标的
    """
    months: List = ser.index.get_level_values(0).unique().tolist()
    author: List = ser.index.get_level_values(1).unique().tolist()

    return gold_stock_frame.query('end_date==@months and author==@author')


class TradeDays():
    """交易日时间处理相关"""
    def __init__(self):

        self.all_trade_days: pd.DatetimeIndex = pd.to_datetime(
            get_all_trade_days())
        self._tradedaysofmonth()

    def tradeday_of_month(self, watch_dt: str) -> int:
        """查询该交易日是当月的第N日"""
        watch_dt = pd.to_datetime(watch_dt)
        idx = self.TradedaysOfMonth.index.get_indexer([watch_dt],
                                                      method='nearest')[0]
        return self.TradedaysOfMonth.iloc[idx, 1]

    def get_tradedays_of_month(self,
                               year: Union[str, int] = None,
                               month: Union[str, int] = None,
                               num: int = None) -> pd.DataFrame:
        """获取月份的第N日"""
        if num is None:
            raise ValueError('num参数不能为空!')

        if (year is not None) and (month is None):

            cond = (self.TradedaysOfMonth.index.year
                    == year) & (self.TradedaysOfMonth['dayofmonth'] == num)

        elif (year is None) and (month is not None):

            cond = (self.TradedaysOfMonth.index.month
                    == month) & (self.TradedaysOfMonth['dayofmonth'] == num)

        else:

            cond = (self.TradedaysOfMonth.index.strftime('%Y%m')
                    == "{}{:02d}".format(year, month)) & (
                        self.TradedaysOfMonth['dayofmonth'] == num)

        return self.TradedaysOfMonth[cond]

    def get_tradedays_month_end(self,
                                year: Union[str, int] = None,
                                month: Union[str, int] = None) -> pd.DataFrame:
        """查询每月最后一个交易日"""

        trade_days = self._MonthEndOrMonthBegin('last')
        if (year is None) and (month is None):

            return trade_days

        elif (year is not None) and (month is None):

            cond = (trade_days.index.year == year)

        elif year is None:

            cond = (trade_days.index.month == month)

        else:

            cond = (trade_days.index.strftime('%Y%m') == "{}{:02d}".format(
                year, month))

        return trade_days[cond]

    def get_tradedays_month_begin(
            self,
            year: Union[str, int] = None,
            month: Union[str, int] = None) -> pd.DataFrame:
        """查询每月最后一个交易日"""
        trade_days = self._MonthEndOrMonthBegin('first')
        if year is None and month is None:
            return trade_days
        elif year is not None and month is None:
            cond = trade_days.index.year == year
        elif year is None:
            cond = trade_days.index.month == month
        else:
            cond = trade_days.index.strftime('%Y%m') == "{}{:02d}".format(
                year, month)
        return trade_days[cond]

    def _MonthEndOrMonthBegin(self, method: str) -> pd.DataFrame:

        cols_dic = {
            'last': ('MonthEnd(all)', 'MonthEnd'),
            'first': ('MonthBegin(all)', 'MonthBegin')
        }
        trade_days = self.TradedaysOfMonth.copy()

        func = {
            'last': trade_days.groupby(pd.Grouper(level=0, freq='M')).last,
            'first': trade_days.groupby(pd.Grouper(level=0, freq='MS')).first
        }

        trade_days = func[method]()

        trade_days[cols_dic[method][0]] = trade_days.index
        trade_days.index = trade_days['trade_days']
        trade_days.rename(columns={'trade_days': cols_dic[method][1]},
                          inplace=True)
        return trade_days.drop(columns=['dayofmonth'])

    def _tradedaysofmonth(self):

        tradedays_frame: pd.DataFrame = self._trans2frame()
        tradedays_frame['dayofmonth'] = tradedays_frame.groupby(
            pd.Grouper(level=0, freq='M'))['trade_days'].transform(
                lambda x: np.arange(1,
                                    len(x) + 1))
        self.TradedaysOfMonth = tradedays_frame

    def _trans2frame(self) -> pd.DataFrame:
        days = self.all_trade_days.to_frame()
        days.columns = ['trade_days']
        return days

'''
Author: shen.lan123@gmail.com
Date: 2022-04-18 17:03:51
LastEditTime: 2022-05-20 17:20:56
LastEditors: hugo2046 shen.lan123@gmail.com
Description: 用于获取数据
'''
import os
import sys

sys.path.append('/home/jquser/')
from typing import (List, Tuple, Dict, Callable, Union)
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")

from .my_scr import (get_dichotomy, get_quadrant, get_factors, get_pricing)

Begin = '2013-01-01'
End = '2022-02-28'
Last_date = '2022-03-31'


def get_data(method: Union[str, List] = 'ALL',
             start: str = Begin,
             end: str = End,
             last_date: str = Last_date):

    dic = {
        'dichotomy': _dump_dichotomy,
        'quadrant': _dump_quadrant,
        'factor': _dump_factor,
        'price': _dump_price
    }

    if isinstance(method, List):

        for m in method:

            dic[m](start=start, end=end, last_date=last_date)

        return

    if method.upper() == 'ALL':

        for func in dic.values():

            func(start=start, end=end, last_date=last_date)

    else:

        dic[method](start=start, end=end, last_date=last_date)


def _dump_dichotomy(start: str = Begin, end: str = End, **kw):

    print('数据获取(起始日:%s,结束日:%s' % (start, end))
    # 划分高低端象限
    print('开始划分高低端象限...')
    dichotomy_df = get_dichotomy(start, end)
    dichotomy_df.to_csv(r'Data/dichotomy.csv')
    print('高低端象限数据获取完毕!')


def _dump_quadrant(start: str = Begin, end: str = End, **kw):
    # 划分四象限
    print('开始划分四象限...')
    quandrant_df = get_quadrant(start, end)
    quandrant_df = quandrant_df.stack().to_frame('cat_type')
    quandrant_df.to_csv(r'Data/quandrant_df.csv')
    print('四象限数据获取完毕!')


def _dump_factor(**kw):

    if os.path.exists(r'Data/quandrant_df.csv'):

        quandrant_df = pd.read_csv(r'Data/quandrant_df.csv',
                                   index_col=[0],
                                   parse_dates=True)

        quandrant_df.columns = ['code', 'cat_type']
        quandrant_df.index.names = ['date']
        quandrant_df = pd.pivot_table(quandrant_df.reset_index(),
                                      index='date',
                                      columns='code',
                                      values='cat_type')

        print('开始获取因子数据...')
        factors_df = get_factors(quandrant_df)
        factors_df.to_csv(r'Data/factors_frame.csv')
        print('因子数据获取完毕!')
    else:
        print('缺少依赖数据:quandrant_df.csv,请先行下载quandrant_df.csv!')


def _dump_price(last_date=Last_date, **kw):

    if os.path.exists(r'Data/factors_frame.csv'):

        factors_df = pd.read_csv(r'Data/factors_frame.csv',
                                 index_col=[0, 1],
                                 parse_dates=True)
        print('开始获取收盘价数据...')
        pricing = get_pricing(factors_df, last_date)
        pricing.to_csv(r'Data/pricing.csv')
        print('收盘价数据获取完毕!')

    else:
        print('缺少依赖数据:quandrant_df.csv,请先行下载quandrant_df.csv!')


def load_data() -> List:

    files = [
        'dichotomy.csv', 'quandrant_df.csv', 'factors_frame.csv', 'pricing.csv'
    ]

    out_put = []

    for file in files:

        file_path = rf'Data/{file}'

        if os.path.exists(file_path):

            if file in [
                    'dichotomy.csv', 'factors_frame.csv', 'quandrant_df.csv'
            ]:
                df = pd.read_csv(file_path, index_col=[0, 1], parse_dates=True)
                df.index.names = ['date', 'asset']
                out_put.append(df)
            else:
                df = pd.read_csv(file_path, index_col=[0], parse_dates=True)
                df.index.names = ['date']
                out_put.append(df)
            print('%s文件读取完毕!' % file)

        else:

            print('%s文件不存在请下载!' % file)

    return out_put

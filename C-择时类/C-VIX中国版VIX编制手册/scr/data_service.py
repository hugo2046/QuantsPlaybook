'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-27 16:28:13
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-05-27 18:49:46
Description: 使用jqdatasdk/jqdata获取期权数据
'''
from typing import (List, Dict, Tuple, Union)

import datetime as dt
import pandas as pd
import numpy as np
from sqlalchemy.sql import func

from .utils import trans_ser2datetime
from jqdata import *


def get_opt_basic(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """查询期权基础信息

    Args:
        code (str): 标的代码
        start_date (str): 起始日
        end_date (str): 结束日

    Returns:
        pd.DataFrame: 
        
        | idnex | list_date | exercise_date | exercise_price | contract_type | code          |
        | :---- | :-------- | :------------ | :------------- | :------------ | :------------ |
        | 0     | 2021/7/29 | 2022/3/23     | 4.332          | CO            | 10003549.XSHG |
    """
    opt_basic: pd.DataFrame = opt.run_query(
        query(opt.OPT_CONTRACT_INFO.list_date,
              opt.OPT_CONTRACT_INFO.exercise_date,
              opt.OPT_CONTRACT_INFO.exercise_price,
              opt.OPT_CONTRACT_INFO.contract_type,
              opt.OPT_CONTRACT_INFO.code).filter(
                  opt.OPT_CONTRACT_INFO.underlying_symbol == code,
                  opt.OPT_CONTRACT_INFO.last_trade_date >= start_date,
                  opt.OPT_CONTRACT_INFO.list_date <= end_date))

    return opt_basic


def offset_limit_func(model, fields: Union[List, Tuple], limit: int,
                      *args) -> pd.DataFrame:
    """利用offset多次查询以跳过限制

    Args:
        model (_type_): model
        fields (Union[List, Tuple]): 查询字段
        limit (int): 限制
        args: 用于查询的条件
    Returns:
        pd.DataFrame

    """
    total_size: int = model.run_query(query(
        func.count('*')).filter(*args)).iloc[0, 0]
    print('总数%s' % total_size)
    dfs: List = []

    #以limit为步长循环offset的参数
    for i in range(0, total_size, limit):

        q = query(*fields).filter(*args).offset(i).limit(limit)  #自第i条数据之后进行获取
        df: pd.DataFrame = model.run_query(q)
        print(i, len(df))
        dfs.append(df)

    df: pd.DataFrame = pd.concat(dfs)

    return df


def get_opt_all_price(codes: Union[str, List]) -> pd.DataFrame:
    """查询codes标的的所有日线数据

    Args:
        codes (Union[str, List]): 期权标的

    Returns:
        pd.DataFrame:
        | idnex | list_date | exercise_date | exercise_price | contract_type | code          |
        | :---- | :-------- | :------------ | :------------- | :------------ | :------------ |
        | 0     | 2021/7/29 | 2022/3/23     | 4.332          | CO            | 10003549.XSHG |
    """
    if isinstance(codes, str):

        codes: List = [codes]

    fields: Tuple = tuple(
        getattr(opt.OPT_DAILY_PRICE, field)
        for field in ('date', 'close', 'code'))
    opt_price: pd.DataFrame = offset_limit_func(
        opt, fields, 4000, opt.OPT_DAILY_PRICE.code.in_(codes))

    return opt_price


def calc_maturity(exercise_date: pd.Series, trade_date: pd.Series,
                  days: int) -> pd.Series:
    """计算交易日到期权执行期直接的时间

        $maturity=\frac{(exercise_date - trade_date)}{days}$
    Args:
        exercise_date (pd.Series): 执行期
        date (pd.Series): 交易日
        days (int): 转化年

    Returns:
        pd.Series
    """
    exercise_date = trans_ser2datetime(exercise_date)
    trade_date = trans_ser2datetime(trade_date)

    return (exercise_date - trade_date).dt.days / days


def prepare_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取计算所需的基础数据

    Args:
        code (str): 期权标的
        start_date (str): 起始日
        end_date (str): 结束日

    Returns:
        pd.DataFrame
            | index | date      | exercise_date | close  | contract_type | exercise_price | maturity |
            | :---- | :-------- | :------------ | :----- | :------------ | :------------- | :------- |
            | 0     | 2021/7/29 | 2022/3/23     | 0.5275 | call          | 4.332          | 0.649315 |
    """
    # 获取期权基础信息
    opt_basic: pd.DataFrame = get_opt_basic(code, start_date, end_date)
    # 获取期权标的
    code_list: List = opt_basic['code'].unique().tolist()

    # 获取期权标的的所有日线数据
    opt_all_price: pd.DataFrame = get_opt_all_price(code_list)

    # 合并日线数据与基础信息数据
    opt_data: pd.DataFrame = pd.merge(opt_all_price, opt_basic, on='code')

    # 计算T日至到期日的距离
    opt_data['maturity'] = calc_maturity(opt_data['exercise_date'],
                                         opt_data['date'], 365)

    # 获取所需信息
    sel_col = 'date,exercise_date,close,contract_type,exercise_price,maturity'.split(
        ',')

    data = opt_data[sel_col].copy()

    data['contract_type'] = data['contract_type'].map({
        "CO": "call",
        "PO": "put"
    })

    data = data.sort_values('date')

    return data


"""

df_rate 数据结构
| date     | on    | 1w    | 2w    | 1m    | 3m    | 6m     | 9m     | 1y    |
| :------- | :---- | :---- | :---- | :---- | :---- | :----- | :----- | :---- |
| 2015/2/9 | 2.812 | 4.335 | 4.807 | 5.025 | 4.904 | 4.7796 | 4.7538 | 4.779 |

"""
# # 测试
# if __name__ == '__main__':

#     symbol:str = '510300.XSHG'
#     start_date:str = '2022-01-01'
#     end_date:str = '2022-05-26'

#     opt_date:pd.DataFrame = prepare_data(symbol,start_date,end_date)
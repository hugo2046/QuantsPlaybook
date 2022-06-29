'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-29 12:56:46
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-29 13:05:01
Description: 
'''
from typing import Dict, List

import pandas as pd
import vectorbt as vbt
from data_service import get_index_daily
from scr import HMA, calc_netbuy, get_exchange_set

if __name__ == "__main__":
    print('数据读取..')
    # 读取本地文件
    billboard_df: pd.DataFrame = pd.read_csv('../data/billboard.csv',
                                             encoding='utf-8',
                                             index_col=[0],
                                             parse_dates=['trade_date'])

    # 获取沪深300数据
    hs300: pd.DataFrame = get_index_daily(code='000300.SH',
                                          start_date='20130101',
                                          end_date='20220222')
    hs300.set_index('trade_date', inplace=True)
    print('划分席位...')
    # 席位划分
    exchange_set: Dict = get_exchange_set(billboard_df)

    is_netbuy_s: pd.Series = calc_netbuy(billboard_df[exchange_set['机构席位']],
                                         hs300['amount'])

    print('参数计算...')
    # 计算HMA信号
    is_netbuy_s_s: pd.Series = HMA(is_netbuy_s, 30)
    is_netbuy_s_l: pd.Series = HMA(is_netbuy_s, 100)

    to_buy1: pd.Series = (is_netbuy_s_s > is_netbuy_s_l) & (
        is_netbuy_s_s > 0) & (is_netbuy_s_l > 0)
    to_buy2: pd.Series = (is_netbuy_s_s < is_netbuy_s_l) & (
        is_netbuy_s_s < 0) & (is_netbuy_s_l < 0)

    # 注:信号滞后 不然会有未来
    entries: pd.Series = (to_buy1 | to_buy2).vbt.signals.fshift()
    exits: pd.Series = (~entries)

    direction = ['longonly']
    fees = 0.001

    pf = vbt.Portfolio.from_signals(
        close=hs300['close'].reindex(is_netbuy_s_s.index),
        price=hs300['open'].reindex(is_netbuy_s_s.index),
        entries=entries,
        exits=exits,
        direction=direction,
        fees=fees,
        log=True,
        freq='1D')
    print('画图...')
    pf.plot(subplots=['trade_pnl'])
    print('结束')
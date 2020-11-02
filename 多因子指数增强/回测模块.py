'''
Author: Hugo
Date: 2020-06-10 20:52:43
LastEditTime: 2020-11-02 10:08:02
LastEditors: Please set LastEditors
Description: 读取指数增强研究文档生成的result_df文件进行下单
'''


from jqdata import *
import pandas as pd
import numpy as np
from six import BytesIO  # 文件读取

enable_profile()  # 开启性能分析


def initialize(context):

    set_params()
    set_variables()
    set_backtest()

    run_monthly(Trade, -1, time='open', reference_security='000300.XSHG')


def set_params():

    g.result_df = pd.read_csv(
        BytesIO(read_file('result_df.csv')), index_col=[0],)


def set_variables():

    pass


def set_backtest():

    set_option("avoid_future_data", True)  # 避免数据
    set_option("use_real_price", True)  # 真实价格交易
    set_benchmark('000300.XSHG')  # 设置基准
    #log.set_level("order", "debuge")
    log.set_level('order', 'error')


# 每日盘前运行
def before_trading_start(context):

    # 手续费设置
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))

    # 根据不同的时间段设置手续费
    dt = context.current_dt

    if dt > datetime.datetime(2013, 1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    elif dt > datetime.datetime(2011, 1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))

    elif dt > datetime.datetime(2009, 1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))


def Trade(context):

    bar_time = context.current_dt.strftime('%Y-%m-%d')
    log.info('%s启动' % bar_time)

    if bar_time in g.result_df.index:
        print('存在')
        target_slice = g.result_df.loc[bar_time]
        BuyStock(context, target_slice)


def BuyStock(context, target_slice: pd.DataFrame):

    order_dict = target_slice.set_index('code')['w'].to_dict()

    for hold in context.portfolio.long_positions:
        if hold not in order_dict:
            order_target(hold, 0)

    totalasset = context.portfolio.total_value
    for buy_stock, pre in order_dict.items():

        order_target_value(buy_stock, pre * totalasset)

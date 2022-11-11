'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-10-28 18:08:47
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-11-11 16:59:41
Description: 
'''
from typing import List, Tuple

import pandas as pd

from .performance import Strategy_performance
from .plotly_chart import (
    plot_annual_returns,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_monthly_returns_dist,
    plot_monthly_returns_heatmap,
    plot_trade_pnl,
    plot_underwater,
    plotl_order_on_ohlc,
    plotly_table,
)


def get_flag(trade_list: pd.DataFrame) -> Tuple:
    """获取买卖点

    Args:
        trade_list (pd.DataFrame): _description_

    Returns:
        Tuple: buy_flag,sell_flag
    """
    buy_flag: pd.Series = trade_list[['datein', 'pricein']].set_index('datein')
    sell_flag: pd.Series = trade_list[['dateout',
                                       'priceout']].set_index('dateout')

    buy_flag.index = pd.to_datetime(buy_flag.index)
    sell_flag.index = pd.to_datetime(sell_flag.index)

    return buy_flag, sell_flag


def calc_win_ratio(ser: pd.Series) -> pd.Series:
    """计算盈利

    Args:
        ser (pd.Series): _description_

    Returns:
        pd.Series: _description_
    """
    return len(ser[ser > 0]) / len(ser)


def calc_profit_coss(ser: pd.Series) -> pd.Series:
    """盈亏比

    Args:
        ser (pd.Series): _description_

    Returns:
        pd.Series: _description_
    """
    return ser[ser > 0].sum() / ser[ser < 0].abs().sum()


def get_backtest_report(price: pd.Series, result: List) -> pd.DataFrame:

    ret: pd.Series = pd.Series(result[0].analyzers._TimeReturn.get_analysis())
    benchmark = price.pct_change()

    returns: pd.DataFrame = pd.concat((ret, benchmark), axis=1)
    returns.columns = ['策略', 'benchmark']

    return Strategy_performance(returns)


def get_trade_res(trade_list: pd.DataFrame) -> pd.Series:

    # 获取交易明细

    days = (trade_list['dateout'] - trade_list['datein']).dt.days

    return pd.DataFrame(
        {
            '总交易次数': len(trade_list),
            '持仓最长时间(自然天)': days.max(),
            '持仓最短时间(自然天)': days.min(),
            '平均持仓天数(自然天)': days.mean(),
            '胜率(%)': '{:.2%}'.format(calc_win_ratio(trade_list['pnl'])),
            '盈亏比': '{:.2}'.format(calc_profit_coss(trade_list['pnl']))
        },
        index=['交易指标'])


# tear
def analysis_rets(price: pd.Series, result: List) -> List:
    """净值表现情况

    Args:
        price (pd.Series): idnex-date values
        result (List): 回测结果
    """
    ret: pd.Series = pd.Series(result[0].analyzers._TimeReturn.get_analysis())
    benchmark = price.pct_change()
    benchmark, ret = benchmark.align(ret, join='left', axis=0)

    returns: pd.DataFrame = pd.concat((ret, benchmark), axis=1)
    returns.columns = ['策略', '基准']

    report_df: pd.DataFrame = Strategy_performance(returns)

    bt_risk_table = plotly_table(
        report_df.T.applymap(lambda x: '{:.2%}'.format(x)), '指标')

    cumulative_chart = plot_cumulative_returns(ret, benchmark)
    maxdrawdowns_chart = plot_drawdowns(ret)
    underwater_chart = plot_underwater(ret)
    annual_returns_chart = plot_annual_returns(ret)
    monthly_return_heatmap_chart = plot_monthly_returns_heatmap(ret)
    monthly_return_dist_chart = plot_monthly_returns_dist(ret)

    return bt_risk_table, cumulative_chart, maxdrawdowns_chart, underwater_chart, annual_returns_chart, monthly_return_heatmap_chart, monthly_return_dist_chart


def analysis_trade(price: pd.DataFrame, result: List) -> List:
    """交易情况

    Args:
        price (pd.DataFrame): index-date OHLCV数据
        result (_type_): _description_
    """
    trade_list: pd.DataFrame = pd.DataFrame(
        result[0].analyzers.tradelist.get_analysis())
    trade_list['pnl%'] /= 100
    trade_res: pd.DataFrame = get_trade_res(trade_list)

    trade_report = plotly_table(trade_res)
    orders_chart = plotl_order_on_ohlc(price, trade_list)
    pnl_chart = plot_trade_pnl(trade_list)

    return trade_report, orders_chart, pnl_chart

"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-10-31 11:08:01
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-12-05 13:28:52
Description: 
"""
from collections import namedtuple
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..VectorbtStylePlotting import (
    plot_annual_returns,
    plot_cumulative,
    plot_drawdowns,
    plot_monthly_dist,
    plot_monthly_heatmap,
    plot_orders,
    plot_pnl,
    plot_position,
    plot_table,
    plot_underwater,
)
from .performance import strategy_performance
from .utils import get_value_from_traderanalyzerdict


def get_transactions_frame(result: List) -> pd.DataFrame:
    """将transactions构建为df

    Args:
        result (List): 回测结果

    Returns:
        pd.DataFrame: index-date
    """
    transaction: Dict = result[0].analyzers._Transactions.get_analysis()
    df: pd.DataFrame = pd.DataFrame(
        index=list(transaction.keys()),
        data=np.squeeze(list(transaction.values())),
        columns=["amount", "price", "sid", "symbol", "value"],
    )
    df.index.names = ["date"]
    df.index = pd.to_datetime(df.index)
    return df


def get_trade_flag(result: List) -> pd.DataFrame:
    """买卖标记

    Args:
        result (List): 回测结果

    Returns:
        pd.DataFrame: indexz-number columns-datein|dateout|pricein|priceout

    如果priceout/dateout为np.nan则表示尚未平仓
    """
    transactions = get_transactions_frame(result)
    transactions = transactions.astype(
        {"amount": np.int32, "price": np.float32, "value": np.float32}
    )

    size = len(transactions)
    trade_date = transactions.index.tolist()
    price = transactions["price"].tolist()
    trade_date = trade_date + [np.nan] if size % 2 else trade_date
    price = price + [np.nan] if size % 2 else price

    size = size + 1 if size % 2 else size

    date_flag = np.array([(trade_date[i - 2 : i]) for i in np.arange(2, size + 1, 2)])
    price_flag = np.array([(price[i - 2 : i]) for i in np.arange(2, size + 1, 2)])

    return pd.DataFrame(
        data=np.hstack((date_flag, price_flag)),
        columns=["datein", "dateout", "pricein", "priceout"],
    )


# report
def get_backtest_report(price: pd.Series, result: List) -> pd.DataFrame:

    ret: pd.Series = pd.Series(result[0].analyzers._TimeReturn.get_analysis())
    benchmark = price.pct_change()

    returns: pd.DataFrame = pd.concat((ret, benchmark), axis=1)
    returns.columns = ["策略", "benchmark"]

    return strategy_performance(returns)


def create_trade_report_table(trader_analyzer: Dict) -> pd.DataFrame:

    won = get_value_from_traderanalyzerdict(trader_analyzer, "won", "total")
    won_money = get_value_from_traderanalyzerdict(
        trader_analyzer, "won", "pnl", "total"
    )
    lost_money = get_value_from_traderanalyzerdict(
        trader_analyzer, "lost", "pnl", "total"
    )
    total = get_value_from_traderanalyzerdict(trader_analyzer, "total", "total")
    streakWonLongest = get_value_from_traderanalyzerdict(
        trader_analyzer, "streak", "won", "longest"
    )
    streakLostLongest = get_value_from_traderanalyzerdict(
        trader_analyzer, "streak", "lost", "longest"
    )

    res: Dict = {
        "交易总笔数": total,
        "完结的交易笔数": get_value_from_traderanalyzerdict(
            trader_analyzer, "total", "closed"
        ),
        "未交易完结笔数": get_value_from_traderanalyzerdict(trader_analyzer, "total", "open"),
        "连续获利次数": streakWonLongest if streakWonLongest else np.nan,
        "连续亏损次数": streakLostLongest if streakLostLongest else np.nan,
        "胜率(%)": round(won / total, 4),
        "盈亏比": round(won_money / abs(lost_money), 4),
        "平均持仓天数": round(
            get_value_from_traderanalyzerdict(trader_analyzer, "len", "average"), 2
        ),
        "最大持仓天数": get_value_from_traderanalyzerdict(trader_analyzer, "len", "max"),
        "最短持仓天数": get_value_from_traderanalyzerdict(trader_analyzer, "len", "min"),
    }

    return pd.DataFrame(res, index=["交易统计"]).T


# tear
def analysis_rets(
    price: pd.Series,
    result: List,
    benchmark_rets: pd.Series = None,
    use_widgets: bool = False,
) -> namedtuple:
    """净值表现情况

    Args:
        price (pd.Series): idnex-date values
        result (List): 回测结果
    """
    report: namedtuple = namedtuple(
        "report",
        "risk_table,cumulative_chart,maxdrawdowns_chart,underwater_chart,annual_returns_chart,monthly_heatmap_chart,monthly_dist_chart",
    )
    rets: pd.Series = pd.Series(result[0].analyzers._TimeReturn.get_analysis())
    if benchmark_rets is None:
        benchmark_rets: pd.Series = price.pct_change()

    rets, benchmark_rets = rets.align(benchmark_rets, join="right", axis=0)

    returns: pd.DataFrame = pd.concat((rets, benchmark_rets), axis=1)
    returns.columns = ["Strategy", "Benchmark"]

    report_table: pd.DataFrame = strategy_performance(returns)

    risk_table: go.Figure = plot_table(
        report_table.T.applymap(lambda x: "{:.2%}".format(x)),
        index_name="指标",
        use_widgets=use_widgets,
    )

    cumulative_chart: go.Figure = plot_cumulative(
        rets,
        benchmark_rets,
        main_kwargs=dict(name="Close"),
        yaxis_tickformat=".2%",
        title="Cumulative",
        use_widgets=use_widgets,
    )
    maxdrawdowns_chart: go.Figure = plot_drawdowns(
        rets, use_widgets=use_widgets, title="Drawdowns"
    )
    underwater_chart: go.Figure = plot_underwater(
        rets, use_widgets=use_widgets, title="Underwater"
    )

    annual_returns_chart: go.Figure = plot_annual_returns(rets, use_widgets=use_widgets)
    monthly_heatmap_chart: go.Figure = plot_monthly_heatmap(
        rets, use_widgets=use_widgets
    )
    monthly_dist_chart: go.Figure = plot_monthly_dist(rets, use_widgets=use_widgets)

    return report(
        risk_table,
        cumulative_chart,
        maxdrawdowns_chart,
        underwater_chart,
        annual_returns_chart,
        monthly_heatmap_chart,
        monthly_dist_chart,
    )


def analysis_trade(
    price: Union[pd.Series, pd.DataFrame], result: List, use_widgets: bool = False
) -> namedtuple:

    report: namedtuple = namedtuple(
        "report", "trade_report,pnl_chart,orders_chart,position_chart"
    )
    trader_analyzer: Dict = result[0].analyzers._TradeAnalyzer.get_analysis()
    trade_res: pd.DataFrame = create_trade_report_table(trader_analyzer)
    trade_list: pd.DataFrame = pd.DataFrame(
        result[0].analyzers._TradeRecord.get_analysis()
    )
    trade_list: pd.DataFrame = trade_list.astype(
        {"datein": np.datetime64, "dateout": np.datetime64}
    )
    trade_report: go.Figure = plot_table(trade_res, use_widgets=use_widgets)

    pnl_chart: go.Figure = plot_pnl(trade_list, use_widgets=use_widgets, title="PnL")

    if isinstance(price, pd.Series):
        orders_chart: go.Figure = plot_orders(
            price, trade_list, use_widgets=use_widgets, title="Orders"
        )
        position_chart: go.Figure = plot_position(
            price, trade_list, use_widgets=use_widgets, title="Position"
        )

    elif isinstance(price, pd.DataFrame):

        print("TODO:尚未完工")

    return report(trade_report, pnl_chart, orders_chart, position_chart)


# def analysis_rets(price: pd.Series, result: List) -> List:
#     """净值表现情况

#     Args:
#         price (pd.Series): idnex-date values
#         result (List): 回测结果
#     """
#     ret: pd.Series = pd.Series(result[0].analyzers._TimeReturn.get_analysis())
#     benchmark = price.pct_change()
#     benchmark, ret = benchmark.align(ret, join="right", axis=0)

#     returns: pd.DataFrame = pd.concat((ret, benchmark), axis=1)
#     returns.columns = ["策略", "基准"]

#     report_df: pd.DataFrame = strategy_performance(returns)

#     bt_risk_table = plotly_table(
#         report_df.T.applymap(lambda x: "{:.2%}".format(x)), "指标"
#     )

#     cumulative_chart = plot_cumulative(ret, benchmark)
#     maxdrawdowns_chart = plot_drawdowns(ret)
#     underwater_chart = plot_underwater(ret)
#     annual_returns_chart = plot_annual_returns(result)
#     monthly_return_heatmap_chart = plot_monthly_returns_heatmap(ret)
#     monthly_return_dist_chart = plot_monthly_returns_dist(ret)

#     return (
#         bt_risk_table,
#         cumulative_chart,
#         maxdrawdowns_chart,
#         underwater_chart,
#         annual_returns_chart,
#         monthly_return_heatmap_chart,
#         monthly_return_dist_chart,
#     )


# def analysis_trade(price: pd.DataFrame, result: List) -> List:
#     """交易情况

#     Args:
#         price (pd.DataFrame): index-date OHLCV数据
#         result (_type_): _description_
#     """

#     traderecord: pd.DataFrame = get_trade_flag(result)
#     trader_analyzer: Dict = result[0].analyzers._TradeAnalyzer.get_analysis()
#     trade_res: pd.DataFrame = create_trade_report_table(trader_analyzer)

#     trade_report = plotly_table(trade_res)
#     orders_chart = plotl_order_on_ohlc(price, traderecord)

#     trade_list: pd.DataFrame = pd.DataFrame(
#         result[0].analyzers._TradeRecord.get_analysis()
#     )

#     pnl_chart = plot_trade_pnl(trade_list)

#     return trade_report, orders_chart, pnl_chart

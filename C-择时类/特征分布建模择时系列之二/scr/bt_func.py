'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-27 17:54:06
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-08-31 12:52:00
Description: 回测相关函数
'''
import datetime as dt
from collections import namedtuple
from typing import Dict, List, Tuple

import backtrader as bt
import backtrader.indicators as btind
import empyrical as ep
import ipywidgets as ipw
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backtrader.feeds import PandasData

from .plotting import (
    plot_annual_returns,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_monthly_returns_dist,
    plot_monthly_returns_heatmap,
    plot_orders_on_price,
    plot_trade_pnl,
    plot_underwater,
    plotly_table,
)


class bimodal_distribution_strategy(bt.Strategy):
    """交易逻辑:
    - 量能指标大于阈值Threshold=1.15,就做多全A指数;
    - 当量能指标小于1且大于$threshold^{-a}$ (其中$a\in$[1.5,3.5])阈值,那么做空全A指数;
    - 当量能指标小于$threshold^{-a}$，市场处于地量反弹的区域，后市做多
    """
    params = dict(threshold=1.15,  # 阈值
                  window=45,  # 量能指标AMA的计算窗口
                  a=1.5,  # 机制参数
                  trade_long=False,  # True为 仅多头；False为多空
                  )

    def log(self, txt: str, current_dt: dt.datetime = None) -> None:

        current_dt = current_dt or self.datas[0].datetime.date(0)
        print(f'{current_dt.isoformat()},{txt}')

    def __init__(self) -> None:

        self.order = None
        self.threshold_short = 1
        self.threshold_long_to_buy = np.power(self.params.threshold, -self.p.a)

        if self.params.window:
            AMA = btind.SMA(self.data.volume, period=self.params.window)
        else:
            AMA = self.data.volume

        volume_index = btind.SMA(AMA, period=5) / btind.SMA(AMA, period=100)
        self.trade_long = not self.params.trade_long

        self.to_long: pd.Series = bt.Or(
            volume_index > self.params.threshold,
            volume_index < self.threshold_long_to_buy)

        self.to_short: pd.Series = bt.And(
            volume_index < self.threshold_short,
            volume_index >= self.threshold_long_to_buy)

    def next(self):

        # 取消之前未执行的订单
        if self.order:
            self.cancel(self.order)

        if (self.position.size < 0) and self.to_long:

            self.close()
            self.order = self.order_target_percent(target=0.95)

        elif (self.position.size > 0) and self.to_short:

            self.close()
            if self.trade_long:
                self.order = self.order_target_percent(target=-0.95)

        else:

            if self.to_long:
                self.order = self.order_target_percent(target=0.95)

            if self.to_short and self.trade_long:

                self.order = self.order_target_percent(target=-0.95)

    def notify_order(self, order) -> None:

        # 未被处理得订单
        if order.status in [order.Submitted, order.Accepted]:

            return

        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                # buy
                self.log(
                    'BUY EXECUTED,ref:%.0f,Price:%.4f,Size:%.2f,Cost:%.4f,Comm %.4f,Stock:%s'
                    % (order.ref, order.executed.price, order.executed.size,
                       order.executed.value, order.executed.comm,
                       order.data._name))

            else:
                # sell
                self.log(
                    'SELL EXECUTED,ref:%.0f,Price:%.4f,Size:%.2f,Cost:%.4f,Comm %.4f,Stock:%s'
                    % (order.ref, order.executed.price, order.executed.size,
                       order.executed.value, order.executed.comm,
                       order.data._name))


class trade_list(bt.Analyzer):
    """获取交易明细"""

    def __init__(self):

        self.trades = []
        self.cumprofit = 0.0

    def notify_trade(self, trade):

        if trade.isclosed:

            brokervalue = self.strategy.broker.getvalue()
            dir = 'short'
            if trade.history[0].event.size > 0:
                dir = 'long'

            pricein = trade.history[len(trade.history) - 1].status.price
            priceout = trade.history[len(trade.history) - 1].event.price
            datein = bt.num2date(trade.history[0].status.dt)
            dateout = bt.num2date(trade.history[len(trade.history) -
                                                1].status.dt)
            if trade.data._timeframe >= bt.TimeFrame.Days:
                datein = datein.date()
                dateout = dateout.date()

            pcntchange = 100 * priceout / pricein - 100
            pnl = trade.history[len(trade.history) - 1].status.pnlcomm
            pnlpcnt = 100 * pnl / brokervalue
            barlen = trade.history[len(trade.history) - 1].status.barlen
            pbar = pnl / barlen
            self.cumprofit += pnl

            size = value = 0.0
            for record in trade.history:
                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value

            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen + 1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen + 1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein
            if dir == 'long':
                mfe = hp
                mae = lp
            if dir == 'short':
                mfe = -lp
                mae = -hp

            self.trades.append({
                'ref': trade.ref,
                'ticker': trade.data._name,
                'dir': dir,
                'datein': datein,
                'pricein': pricein,
                'dateout': dateout,
                'priceout': priceout,
                'chng%': round(pcntchange, 2),
                'pnl': pnl,
                'pnl%': round(pnlpcnt, 2),
                'size': size,
                'value': value,
                'cumpnl': self.cumprofit,
                'nbars': barlen,
                'pnl/bar': round(pbar, 2),
                'mfe%': round(mfe, 2),
                'mae%': round(mae, 2)
            })

    def get_analysis(self):
        return self.trades


# 考虑佣金和印花税的股票百分比费用
class StockCommission(bt.CommInfoBase):
    params = (('stamp_duty', 0.001),
              ('stocklike', True),  # 指定为股票模式
              ('commtype', bt.CommInfoBase.COMM_PERC),  # 使用百分比费用模式
              ('percabs', True),)  # commission 不以 % 为单位 # 印花税默认为 0.1%

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入时，只考虑佣金
            return abs(size) * price * self.p.commission
        elif size < 0:  # 卖出时，同时考虑佣金和印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        else:
            return 0


def get_backtesting(data: pd.DataFrame,
                    name: str,
                    strategy: bt.Strategy,
                    is_opt: bool = False,
                    **kw) -> namedtuple:
    """回测

    添加了百分比滑点(0.0001)
    当日信号次日开盘买入
    Args:
        data (pd.DataFrame): OHLC数据包含信号
        name (str): 数据名称
        strategy (bt.Strategy): 策略
        is_opt (bool): True-策略优化寻参 False-普通回测

    Returns:
        _type_: _description_
    """

    res = namedtuple('Res', 'result,cerebro')
    cerebro = bt.Cerebro()
    func_dic: Dict = {True: getattr(
        cerebro, 'optstrategy'), False: getattr(cerebro, 'addstrategy')}

    cerebro.broker.setcash(10e4)
    begin_dt = data.index.min()
    end_dt = data.index.max()

    datafeed = PandasData(dataname=data, fromdate=begin_dt, todate=end_dt)
    cerebro.adddata(datafeed, name=name)

    # 设置百分比滑点
    cerebro.broker.set_slippage_perc(perc=0.0001)

    # 设置交易费用
    comminfo = StockCommission(commission=0.0002, stamp_duty=0.001)  # 实例化
    cerebro.broker.addcommissioninfo(comminfo)

    # # 当日下单，当日收盘价成交
    # cerebro.broker.set_coc(True)
    # 添加策略
    #cerebro.addstrategy(strategy, **kw)
    func_dic[is_opt](strategy, **kw)

    # 添加分析指标
    # 返回年初至年末的年度收益率
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')
    # 计算最大回撤相关指标
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    # 计算年化收益
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    # 交易分析添加
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')
    # 计算夏普比率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='_SharpeRatio_A')
    # 返回收益率时序
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    cerebro.addanalyzer(trade_list, _name='tradelist')
    result = cerebro.run(tradehistory=True)

    return res(result, cerebro)


def analysis_rets(price: pd.Series, result: List):
    """净值表现情况

    Args:
        price (pd.Series): idnex-date values
        result (List): 回测结果
    """
    ret: pd.Series = get_time_returns(result)
    benchmark = price.pct_change()

    returns: pd.DataFrame = pd.concat((ret, benchmark), axis=1, join='inner')
    returns.columns = ['策略', '基准']

    df = pd.DataFrame()
    df['年化收益率'] = ep.annual_return(returns, period='daily')

    df['累计收益'] = returns.apply(lambda x: ep.cum_returns(x).iloc[-1])

    df['波动率'] = returns.apply(
        lambda x: ep.annual_volatility(x, period='daily'))

    df['夏普'] = returns.apply(ep.sharpe_ratio, period='daily')

    df['最大回撤'] = returns.apply(lambda x: ep.max_drawdown(x))

    df['索提诺比率'] = returns.apply(lambda x: ep.sortino_ratio(x, period='daily'))

    df['Calmar'] = returns.apply(lambda x: ep.calmar_ratio(x, period='daily'))

    #print_table(df, fmt='{:.2%}')
    #plot_algorithm_nav(result, price, '净值表现')

    f1 = plotly_table(df.applymap(lambda x: '{:.2%}'.format(x)), '指标')
    f1 = f1.update_layout(width=1200, height=300)

    f2 = plot_cumulative_returns(ret, benchmark)
    f3 = plot_drawdowns(ret)
    f4 = plot_underwater(ret)
    f5 = plot_annual_returns(ret)
    f6 = plot_monthly_returns_heatmap(ret)
    f7 = plot_monthly_returns_dist(ret)

    f1, f2, f3, f4, f5, f6, f7 = [
        go.FigureWidget(fig) for fig in [f1, f2, f3, f4, f5, f6, f7]
    ]
    suplots = [f1, f2, ipw.HBox([f3, f4]), f6, ipw.HBox([f5, f7])]
    box_layout = ipw.Layout(display='space-between',
                            border='3px solid black',
                            align_items='inherit')
    return ipw.VBox(suplots, layout=box_layout)


def analysis_trade(price: pd.DataFrame, result: List):
    """交易情况

    Args:
        price (pd.DataFrame): index-date OHLCV数据
        result (_type_): _description_
    """
    trade_list: pd.DataFrame = pd.DataFrame(
        result[0].analyzers.tradelist.get_analysis())
    trade_list['pnl%'] /= 100
    trade_res: pd.DataFrame = get_trade_res(trade_list)

    # print_table(trade_res)

    # buy_flag, sell_flag = get_flag(trade_list)
    # plot_trade_flag(price, buy_flag, sell_flag)
    f1 = plotly_table(trade_res)
    f1 = f1.update_layout(width=1100, height=300)
    f2 = plot_orders_on_price(price, trade_list)
    f3 = plot_trade_pnl(trade_list)
    subplots = [go.FigureWidget(fig) for fig in [f1, f2, f3]]

    box_layout = ipw.Layout(display='space-between',
                            border='3px solid black',
                            align_items='inherit')
    return ipw.VBox(subplots, layout=box_layout)


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


def get_trade_res(trade_list: pd.DataFrame) -> pd.Series:

    # 获取交易明细

    days = (trade_list['dateout'] - trade_list['datein']).dt.days

    return pd.DataFrame(
        {
            '总交易次数': len(trade_list),
            '持仓最长时间(自然天)': days.max(),
            '持仓最短时间(自然天)': days.min(),
            '平均持仓天数(自然天)': round(days.mean(), 2),
            '胜率(%)': '{:.2%}'.format(calc_win_ratio(trade_list['pnl'])),
            '盈亏比': '{:.2}'.format(calc_profit_coss(trade_list['pnl']))
        },
        index=['交易指标'])


def get_result_back_report(result) -> List:

    trader_analyzer = result.analyzers._TradeAnalyzer.get_analysis()
    return [result.params.a,
            result.params.window,
            round(result.analyzers._Returns.get_analysis()['rnorm100'], 4),
            -round(result.analyzers._DrawDown.get_analysis()
                   ['max']['drawdown'], 4),
            round(result.analyzers._SharpeRatio_A.get_analysis()
                  ['sharperatio'], 4),
            trader_analyzer['total']['total'],
            round((trader_analyzer['won']['total'] /
                   trader_analyzer['total']['total']) * 100, 2),
            round(abs(trader_analyzer['won']['pnl']['total'] /
                      trader_analyzer['lost']['pnl']['total']), 2),
            round((trader_analyzer['long']['won'] /
                   trader_analyzer['long']['total']) * 100, 2),
            round(abs(trader_analyzer['long']['pnl']['won']['total'] /
                      trader_analyzer['long']['pnl']['lost']['total']), 2),
            round((trader_analyzer['short']['won'] /
                   trader_analyzer['short']['total']) * 100, 2),
            round(abs(trader_analyzer['short']['pnl']['won']['total'] / trader_analyzer['short']['pnl']['lost']['total']), 2)]


def get_opt_stratgey_table(results: List) -> pd.DataFrame:

    columns: List = ['参数a', '窗口期', '累计收益(%)', '最大回撤(%)', '夏普',
                     '交易总数', '胜率(%)', '盈亏比', '多头胜率(%)', '多头盈亏比', '空头胜率(%)', '空头盈亏比']
    par_df = pd.DataFrame([get_result_back_report(res[0])
                          for res in results], columns=columns)

    par_df.set_index(['参数a', '窗口期'], inplace=True)

    return par_df


def get_time_returns(result) -> pd.Series:

    return pd.Series(result[0].analyzers._TimeReturn.get_analysis())

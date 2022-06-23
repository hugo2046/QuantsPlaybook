'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-27 17:54:06
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-23 16:52:20
Description: 回测相关函数
'''
import datetime as dt
from collections import namedtuple
from typing import Tuple

import backtrader as bt
import empyrical as ep
import pandas as pd
from backtrader.feeds import PandasData

from .plotting import get_strat_ret, plot_algorithm_nav, plot_trade_flag
from .utils import print_table


class add_pandas_data(PandasData):
    """用于加载回测用数据

    添加信号数据
    """
    lines = (
        'fast',
        'slow',
    )

    params = (
        ('fast', -1),
        ('slow', -1),
    )


class ma_cross(bt.Strategy):
    """策略逻辑:

    1.大幅相对净流入:IS_NetBuy_S_S>IS_NetBuy_S_L(短期均线大于长期均线)且短期均 线 IS_NetBuy_S_S>0 且长期均线 IS_NetBuy_S_L>0 做多
    2.大幅相对净流出:IS_NetBuy_S_S<IS_NetBuy_S_L(短期均线小于长期均线) 且短期 均线 IS_NetBuy_S_S<0 且长期均线 IS_NetBuy_S_L<0 做多
    """
    def log(self, txt: str, current_dt: dt.datetime = None) -> None:

        current_dt = current_dt or self.datas[0].datetime.date(0)
        print('%s,%s' % (current_dt.isoformat(), txt))

    def __init__(self) -> None:

        self.order = None

    def next(self):
        # 取消之前未执行的订单
        if self.order:
            self.cancel(self.order)

        # 大幅相对净流入
        to_buy1 = (self.datas[0].fast[0] > self.datas[0].slow[0]) and (
            self.datas[0].fast[0] > 0) and (self.datas[0].slow[0] > 0)

        # 大幅相对净流出
        to_buy2 = (self.datas[0].fast[0] < self.datas[0].slow[0]) and (
            self.datas[0].fast[0] < 0) and (self.datas[0].slow[0] < 0)

        # 检查是否有持仓
        if not self.position:

            if to_buy1 or to_buy2:
                # 全仓买入
                self.order = self.order_target_percent(target=0.9)

        # 有持仓但不满足规则
        elif (not to_buy1) or (not to_buy2):
            # 平仓
            self.order = self.close()

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


def get_backtesting(data: pd.DataFrame,
                    name: str,
                    strategy: bt.Strategy,
                    load_class: PandasData = add_pandas_data) -> namedtuple:
    """回测

    添加了百分比滑点(0.0001)
    当日信号次日开盘买入
    Args:
        data (pd.DataFrame): OHLC数据包含信号
        name (str): 数据名称
        strategy (bt.Strategy): 策略
        load_class (PandasData): 加载模块

    Returns:
        _type_: _description_
    """

    res = namedtuple('Res', 'result,cerebro')
    cerebro = bt.Cerebro()

    cerebro.broker.setcash(10e4)
    begin_dt = data.index.min()
    end_dt = data.index.max()

    datafeed = load_class(dataname=data, fromdate=begin_dt, todate=end_dt)
    cerebro.adddata(datafeed, name=name)

    # 设置百分比滑点
    cerebro.broker.set_slippage_perc(perc=0.0001)
    # # 当日下单，当日收盘价成交
    # cerebro.broker.set_coc(True)
    # 添加策略
    cerebro.addstrategy(strategy)
    # 添加分析指标
    # 返回年初至年末的年度收益率
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')
    # 计算最大回撤相关指标
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    # 计算年化收益
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    # 计算年化夏普比率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                        _name='_SharpeRatio',
                        timeframe=bt.TimeFrame.Days,
                        annualize=True,
                        riskfreerate=0)  # 计算夏普比率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='_SharpeRatio_A')
    # 返回收益率时序
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    cerebro.addanalyzer(trade_list, _name='tradelist')
    result = cerebro.run(tradehistory=True)

    return res(result, cerebro)


def analysis_rets(price: pd.Series, result):
    """净值表现情况

    Args:
        price (pd.Series): idnex-date values
        result (_type_): _description_
    """
    ret: pd.Series = get_strat_ret(result)
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

    print_table(df, fmt='{:.2%}')
    plot_algorithm_nav(result, price, '净值表现')


def analysis_trade(price: pd.DataFrame, result):
    """交易情况

    Args:
        price (pd.DataFrame): index-date OHLCV数据
        result (_type_): _description_
    """
    trade_list: pd.DataFrame = pd.DataFrame(
        result[0].analyzers.tradelist.get_analysis())

    trade_res: pd.DataFrame = get_trade_res(trade_list)

    print_table(trade_res)

    buy_flag, sell_flag = get_flag(trade_list)
    plot_trade_flag(price, buy_flag, sell_flag)


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
            '平均持仓天数(自然天)': days.mean(),
            '胜率(%)': '{:.2%}'.format(calc_win_ratio(trade_list['pnl'])),
            '盈亏比': '{:.2}'.format(calc_profit_coss(trade_list['pnl']))
        },
        index=['交易指标'])

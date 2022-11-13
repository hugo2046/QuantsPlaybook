'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-10-27 20:34:02
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-11-13 19:08:20
Description: 回测所需配件
'''
import datetime
from collections import namedtuple
from multiprocessing import cpu_count
from typing import Dict

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd

from .bt_strategy import VM_Strategy

MAX_CPU: int = int(cpu_count() * 0.8)

# class load_backtest_data(btfeeds.PandasData):

#     # 在基础数据之外额外增加信号 作为开平仓的依据
#     lines = ('indicator', )
#     params = (('indicator', -1), )


class trade_list(bt.Analyzer):
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
    params = (
        ('stamp_duty', 0.001),
        ('stocklike', True),  # 指定为股票模式
        ('commtype', bt.CommInfoBase.COMM_PERC),  # 使用百分比费用模式
        ('percabs', True),
    )  # commission 不以 % 为单位 # 印花税默认为 0.1%

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入时，只考虑佣金
            return abs(size) * price * self.p.commission
        elif size < 0:  # 卖出时，同时考虑佣金和印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        else:
            return 0


def get_backtesting(data: pd.DataFrame,
                    name: str,
                    strategy: bt.Strategy = VM_Strategy,
                    begin_dt: datetime.date = None,
                    end_dt: datetime.date = None,
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
        namedtuple: result,cerebro
    """

    res = namedtuple('Res', 'result,cerebro')

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1e8)
    if (begin_dt is None) or (end_dt is None):
        begin_dt = data.index.min()
        end_dt = data.index.max()
    else:
        begin_dt = pd.to_datetime(begin_dt)
        end_dt = pd.to_datetime(end_dt)
    # datafeed = load_backtest_data(dataname=data,
    #                               fromdate=begin_dt,
    #                               todate=end_dt)
    datafeed = bt.feeds.PandasData(dataname=data,
                                   fromdate=begin_dt,
                                   todate=end_dt)
    cerebro.adddata(datafeed, name=name)

    # 设置百分比滑点
    cerebro.broker.set_slippage_perc(perc=0.0001)

    # 设置交易费用
    comminfo = StockCommission(commission=0.0002, stamp_duty=0.001)
    cerebro.broker.addcommissioninfo(comminfo)

    # # 当日下单，当日收盘价成交
    # cerebro.broker.set_coc(True)
    # 添加策略
    cerebro.addstrategy(strategy)

    # 添加分析指标
    # 返回年初至年末的年度收益率
    # cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')
    # 计算最大回撤相关指标
    # cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    # 计算年化收益
    # cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    # 交易分析添加
    # cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')
    # 交易分析
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='_Transactions')
    # 计算夏普比率
    # cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='_SharpeRatio_A')
    # 返回收益率时序
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    cerebro.addanalyzer(trade_list, _name='tradelist')

    result = cerebro.run(tradehistory=True)

    return res(result, cerebro)

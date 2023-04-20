'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-04-20 14:43:52
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-04-20 14:52:52
Description: 双均线策略
'''

import backtrader as bt

from .bt_icu_ind import IcuMaInd


class CrossOverStrategy(bt.Strategy):
    """当close上穿signal时开仓,下穿时平仓
    T日收盘产生信号,T+1日开盘买入
    """

    params = (("verbose", True), ("periods", 5))
    import datetime as dt

    def log(
        self, txt: str, current_dt: dt.datetime = None, verbose: bool = True
    ) -> None:
        if verbose:
            current_dt = current_dt or self.datas[0].datetime.date(0)
            print(f"{current_dt.isoformat()},{txt}")

    def __init__(self) -> None:
        self.order = None
        # 当close上穿signal时crossover=1,下穿时crossover=-1
        self.signal = IcuMaInd(self.data, N=self.p.periods)  # 使用自定义指标
        # self.data.signal
        self.crossover = bt.indicators.CrossOver(self.data.close, self.signal)

    def next(self):
        # 检查是否有持仓
        if not self.position:
            # 10日均线上穿5日均线，买入
            if self.crossover > 0:
                self.order = self.order_target_percent(target=0.95)
        # # 10日均线下穿5日均线，卖出
        elif self.crossover < 0:
            self.order = self.close()  # 平仓，以下一日开盘价卖出

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已被处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, ref:%.0f,Price: %.4f, Size: %.2f, Cost: %.4f, Comm %.4f, Stock: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.size,
                        order.executed.value,
                        order.executed.comm,
                        order.data._name,
                    ),
                    verbose=self.p.verbose,
                )
            else:  # Sell
                self.log(
                    "SELL EXECUTED, ref:%.0f, Price: %.4f, Size: %.2f, Cost: %.4f, Comm %.4f, Stock: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.size,
                        order.executed.value,
                        order.executed.comm,
                        order.data._name,
                    ),
                    verbose=self.p.verbose,
                )



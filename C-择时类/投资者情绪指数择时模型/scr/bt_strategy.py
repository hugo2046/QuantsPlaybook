"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-11 10:03:20
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-11 11:29:09
Description: 策略
"""
import backtrader as bt

# 策略模板


class SignalStrategy(bt.Strategy):

    params = (
        ("open_threshold", 0.301),
        ("close_threshold", -0.301),
    )

    def log(self, txt, dt=None):
        # log记录函数
        dt = dt or self.datas[0].datetime.date(0)

        print(f"{dt.isoformat()}, {txt}")

    def __init__(self):

        self.dataclose = self.data.close
        self.signal = self.data.GSISI
        self.order = None

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s"
                    % (
                        order.ref,  # 订单编号
                        order.executed.price,  # 成交价
                        order.executed.value,  # 成交额
                        order.executed.comm,  # 佣金
                        order.executed.size,  # 成交量
                        order.data._name,
                    )
                )  # 股票名称
            else:  # Sell
                self.log(
                    "SELL EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm,
                        order.executed.size,
                        order.data._name,
                    )
                )

    def next(self):

        # 取消之前未执行的订单
        if self.order:
            self.cancel(self.order)

        if self.position:
            if (
                self.signal[0] <= self.params.close_threshold
                and self.signal[-1] <= self.params.close_threshold
            ):
                self.log("收盘价Close, %.2f" % self.dataclose[0])
                self.log(
                    "设置卖单SELL CREATE, %.2f信号为:%.2f,阈值为:%.2f"
                    % (self.dataclose[0], self.signal[0], self.params.close_threshold)
                )
                self.order = self.order_target_value(target=0.0)

        elif (
            self.signal[0] >= self.params.open_threshold
            and self.signal[-1] >= self.params.open_threshold
        ):
            self.log("收盘价Close, %.2f" % self.dataclose[0])
            self.log(
                "设置买单 BUY CREATE, %.2f,信号为:%.2f,阈值为:%.2f"
                % (self.dataclose[0], self.signal[0], self.params.open_threshold)
            )
            self.order = self.order_target_percent(target=0.95)

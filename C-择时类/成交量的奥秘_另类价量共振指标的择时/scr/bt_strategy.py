'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-10-28 17:13:42
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-11-11 16:39:24
Description: 策略模块
'''

import backtrader as bt
import backtrader.indicators as btind  # 导入策略分析模块


class VM_Indicator(bt.Indicator):
    """信号"""
    lines = ('Vol_Mom', )
    params = (('bma_window', 50), ('ama_window', 100), ('n', 3))

    def __init__(self):

        bma = btind.HullMovingAverage(self.data.close,
                                      period=self.p.bma_window)
        price_mom = bma / bma(-self.p.n)
        vol_mom = btind.HullMovingAverage(
            self.data.volume, period=5) / btind.HullMovingAverage(
                self.data.volume, period=self.p.ama_window)
        self.l.Vol_Mom = price_mom * vol_mom


class Shake_Filter(bt.Indicator):
    """市场行情过滤"""
    lines = ('Threshold', )
    params = (('fast_window', 5), ('slow_window', 90), ('threshold', (1.125,
                                                                      1.275)))

    def __init__(self):

        threshold1, threshold2 = self.p.threshold
        fast_line = btind.SMA(self.data.close, period=self.p.fast_window)
        slow_line = btind.SMA(self.data.close, period=self.p.slow_window)

        self.l.Threshold = bt.If(fast_line > slow_line, threshold1, threshold2)


# 策略模板


class VM_Strategy(bt.Strategy):

    params = (
        ('bma_window', 50),
        ('ama_window', 100),
        ('n', 3),
        ('fast_window', 5),
        ('slow_window', 90),
        ("threshold", (1.125, 1.275)),
    )

    def log(self, txt, dt=None):
        # log记录函数
        dt = dt or self.datas[0].datetime.date(0)

        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):

        self.dataclose = self.data.close
        self.vm = VM_Indicator(bma_window=self.p.bma_window,
                               ama_window=self.p.ama_window,
                               n=self.p.n)

        self.threshold = Shake_Filter(fast_window=self.p.fast_window,
                                      slow_window=self.p.slow_window,
                                      threshold=self.p.threshold)

        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 检查订单执行状态order.status：
            # Buy/Sell order submitted/accepted to/by broker
            # broker经纪人：submitted提交/accepted接受,Buy买单/Sell卖单
            # 正常流程，无需额外操作
            return

        # 检查订单order是否完成
        # 注意: 如果现金不足，经纪人broker会拒绝订单reject order
        # 可以修改相关参数，调整进行空头交易
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('买单执行BUY EXECUTED, 报价：%.2f' % order.executed.price)
            elif order.issell():
                self.log('卖单执行SELL EXECUTED,报价： %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单Order: 取消Canceled/保证金Margin/拒绝Rejected')

        # 检查完成，没有交易中订单（pending order）
        self.order = None

    def next(self):

        # 取消之前未执行的订单
        if self.order:
            self.cancel(self.order)

        if self.position:
            if self.vm <= self.threshold:
                self.log('收盘价Close, %.2f' % self.dataclose[0])
                self.log('设置卖单SELL CREATE, %.2f信号为:%.2f,阈值为:%.2f' %
                         (self.dataclose[0], self.vm[0], self.threshold[0]))
                self.order = self.order_target_value(target=0.)

        elif self.vm > self.threshold:
            self.log('收盘价Close, %.2f' % self.dataclose[0])
            self.log('设置买单 BUY CREATE, %.2f,信号为:%.2f,阈值为:%.2f' %
                     (self.dataclose[0], self.vm[0], self.threshold[0]))
            self.order = self.order_target_percent(target=0.95)

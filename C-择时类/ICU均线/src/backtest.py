"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-04-20 14:43:52
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-04-20 14:46:20
Description: 回测模块
"""
from typing import Dict, List
import empyrical as ep
import backtrader as bt
import pandas as pd

from .bt_strategy import CrossOverStrategy


class add_price(bt.feeds.PandasData):
    """用于加载回测用数据

    添加信号数据
    """

    lines = ("signal",)

    params = (("signal", -1),)


# 考虑佣金和印花税的股票百分比费用
class StockCommission(bt.CommInfoBase):
    params = (
        ("stamp_duty", 0.001),
        ("stocklike", True),  # 指定为股票模式
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 使用百分比费用模式
        ("percabs", True),
    )  # commission 不以 % 为单位 # 印花税默认为 0.1%

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入时，只考虑佣金
            return abs(size) * price * self.p.commission
        elif size < 0:  # 卖出时，同时考虑佣金和印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        else:
            return 0


def get_backtest(
    dataset: pd.DataFrame,
    N: int,
    start_dt: str = None,
    end_dt: str = None,
    name: str = "",
) -> List:
    """使用backtrader进行回测

    Args:
        dataset (pd.DataFrame): 数据集 不支持多个表的数据
        N (int): ICU计算的窗口期
        start_dt (str, optional): 起始日. Defaults to None.
        end_dt (str, optional): 结束日. Defaults to None.
        name (str, optional): 标的名称. Defaults to "".

    Returns:
        List: 回测结果
    """
    if (start_dt is None) or (end_dt is None):
        start_dt: pd.Timestamp = dataset.index.min()
        end_dt: pd.Timestamp = dataset.index.max()

    cerebro = bt.Cerebro()
    # 当日信号 当日收盘交易
    cerebro.broker.set_coc(True)
    datafeed: bt.feeds.PandasData = add_price(
        dataname=dataset, fromdate=start_dt, todate=end_dt
    )

    cerebro.adddata(datafeed, name=name)
    # 初始资金 100,000,000
    cerebro.broker.setcash(100000000.0)
    # 设置百分比滑点
    cerebro.broker.set_slippage_perc(perc=0.0001)

    # 设置交易费用
    comminfo = StockCommission(commission=0.0003, stamp_duty=0.001)
    cerebro.broker.addcommissioninfo(comminfo)

    # 添加分析指标
    # 返回年初至年末的年度收益率
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="_AnnualReturn")
    # 计算最大回撤相关指标
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="_DrawDown")
    # 计算年化收益
    cerebro.addanalyzer(bt.analyzers.Returns, _name="_Returns", tann=252)
    # 交易分析添加
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="_TradeAnalyzer")
    # 计算夏普比率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="_SharpeRatio_A")
    # 返回收益率时序
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")

    cerebro.addstrategy(CrossOverStrategy, verbose=False, periods=N)

    result: List = cerebro.run()

    return result


def runstrat(periods: Dict, dataset: pd.DataFrame, method: str) -> float:
    result: List = get_backtest(dataset, periods["N"])
    analyzers = result[0].analyzers

    def _get_rets() -> pd.Series:
        return pd.Series(analyzers._TimeReturn.get_analysis())

    if method == "ann":
        return analyzers._Returns.get_analysis()["rnorm100"]

    elif method == "sharpe":
        return analyzers._SharpeRatio_A.get_analysis()

    elif method == "dw":
        return analyzers._DrawDown.get_analysis()["max"]["drawdown"]

    elif method == "calmar":
        return ep.calmar_ratio(_get_rets())

    elif method == "cum":
        return ep.cum_returns(_get_rets()).iloc[-1]

    else:
        raise ValueError("method must be in ['ann','sharpe','dw','calmar','cum']")

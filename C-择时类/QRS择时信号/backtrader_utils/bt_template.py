"""
Author: Hugo
Date: 2024-10-26 21:31:21
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-10-28 16:24:06
Description: 用于运行backtrader策略的模板
"""

import backtrader as bt
from .datafeed import DailyOHLCVUSLFeed
from .engine import BackTesting
import pandas as pd
from typing import Dict

__all__ = ["run_template_strategy", "COMMISSION"]

# 设置初始金额及手续费
COMMISSION: Dict = dict(
    cash=1e8, commission=0.00015, stamp_duty=0.0001, slippage_perc=0.0001
)

# 设置策略参数
STRATEGY_PARAMS: Dict = {"verbose": False, "hold_num": 1}


def update_params(default_params: Dict, custom_params: Dict) -> Dict:
    if custom_params is None:
        return default_params
    default_params.update(custom_params)
    return default_params


def run_template_strategy(
    data: pd.DataFrame,
    code: str,
    strategy: bt.Strategy,
    strategy_kwargs: Dict = {},
    commission_kwargs: Dict = None,
):

    commission_kwargs: Dict = update_params(COMMISSION, commission_kwargs)
    strategy_kwargs: Dict = update_params(STRATEGY_PARAMS, strategy_kwargs)

    if isinstance(code, str):

        df: pd.DataFrame = data.query("code == @code").copy()

    elif isinstance(code, list):

        df: pd.DataFrame = data.query("code in @code").copy()
        strategy_kwargs["hold_num"] = len(code)

    df: pd.DataFrame = df.dropna(subset=["close", "upperbound"])
    bt_engine = BackTesting(**commission_kwargs)
    bt_engine.load_data(
        df,
        datafeed_cls=DailyOHLCVUSLFeed,
    )
    bt_engine.add_strategy(strategy, **strategy_kwargs)
    # 创建订单时不检查现金是否够用，执行时检查(使用下一个bar的价格作计算订单量)
    # bt_engine.cerebro.broker.set_checksubmit(False)
    bt_engine.cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days
    )
    bt_engine.cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
    bt_engine.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    # 与empyrical库中的sharpe_ratio函数计算结果一致
    bt_engine.cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=bt.TimeFrame.Days,
        riskfreerate=0,
        stddev_sample=True,
        annualize=True,
        _name="sharpe_ratio",
    )
    bt_engine.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    # 与empyrical库中的annual_return函数计算结果一致
    bt_engine.cerebro.addanalyzer(
        bt.analyzers.Returns, timeframe=bt.TimeFrame.Days, _name="annual_return"
    )
    result = bt_engine.cerebro.run()[0]

    return result

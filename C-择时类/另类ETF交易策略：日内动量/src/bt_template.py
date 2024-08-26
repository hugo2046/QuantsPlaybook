import backtrader as bt
from src.datafeed import ETFDataFeed
from src.engine import BackTesting
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

    df: pd.DataFrame = df.dropna(subset=["close","upperbound"])
    bt_engine = BackTesting(**commission_kwargs)
    bt_engine.load_data(
        df,
        datafeed_cls=ETFDataFeed,
    )
    bt_engine.add_strategy(strategy, **strategy_kwargs)
    # 创建订单时不检查现金是否够用，执行时检查(使用下一个bar的价格作计算订单量)
    # bt_engine.cerebro.broker.set_checksubmit(False)
    bt_engine.cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days
    )
    bt_engine.cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
    result = bt_engine.cerebro.run()[0]

    return result

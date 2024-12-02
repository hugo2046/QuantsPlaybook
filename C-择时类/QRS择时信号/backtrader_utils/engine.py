"""
Author: Hugo
Date: 2024-08-12 14:26:37
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-08-12 14:36:30
Description: 二次封装的backtrader引擎
"""

import backtrader as bt
import pandas as pd
from loguru import logger
# from rich.progress import track
from tqdm.notebook import tqdm as track
from typing import Tuple, List
from .datafeed import DailyOHLCVUSLFeed

__all__ = ["BackTesting", "StockCommission","cheak_dataframe_cols"]


# 手续费及滑点设置
class StockCommission(bt.CommInfoBase):
    params = (
        ("stamp_duty", 0.0001),
        ("stocklike", True),
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


def cheak_dataframe_cols(
    dataframe: pd.DataFrame, datafeed_cls: bt.feeds.PandasData
) -> bool:
    """
    检查数据框的列。

    参数:
        dataframe (pd.DataFrame): 要检查的数据框。
        datafeed_cls (bt.feeds.PandasData): 数据源类。

    返回值:
        bool: 如果检查通过，则返回True；否则返回False。
    """
    if not isinstance(datafeed_cls, bt.feed.MetaAbstractDataBase):
        raise ValueError(
            "datafeed_cls must be a subclass of bt.feeds.PandasData or bt.feeds.PandasDirectData"
        )

    cols: List[Tuple] = [
        (k, v)
        for k, v in datafeed_cls.params.__dict__.items()
        if isinstance(v, int) and k not in ("timeframe", "dtformat")
    ]
    sorted_cols: List[Tuple] = sorted(cols, key=lambda x: x[1])
    return dataframe[
        [
            v[0]
            for v in sorted_cols
            if (v[0] != "datetime") and (v[1] != 0 or v[1] is not None)
        ]
    ]


class BackTesting:

    def __init__(
        self,
        cash: int,
        commission: float = 0.00015,
        stamp_duty: float = 0.0001,
        slippage_perc: float = 0.0001,
    ) -> None:

        self.cerebro = bt.Cerebro()
        # 设置交易费用
        comminfo = StockCommission(
            commission=commission, stamp_duty=stamp_duty
        )  # 实例化
        self.cerebro.broker.addcommissioninfo(comminfo)

        # 滑点：双边各 0.0001
        self.cerebro.broker.set_slippage_perc(perc=slippage_perc)
        self.cerebro.broker.set_cash(cash)

        self.datas = pd.DataFrame()  # 用于储存ohlcv与因子数据

    def load_data(
        self,
        data: pd.DataFrame,
        start_dt: str = None,
        end_dt: str = None,
        datafeed_cls: bt.feed.MetaAbstractDataBase = None,
    ) -> None:

        if start_dt is not None:

            data: pd.DataFrame = data.loc[pd.to_datetime(start_dt):]

        if end_dt is not None:

            data: pd.DataFrame = data.loc[:pd.to_datetime(end_dt)]

        if (start_dt is None) and (end_dt is None):

            start_dt, end_dt = data.index.min(), data.index.max()
            data: pd.DataFrame = data.loc[start_dt:end_dt]

        self.datas = data
       

        # 加载数据
        logger.info("开始加载数据...")
        for code, df in track(
            data.groupby("code"), desc="数据加载到回测引擎..."
        ):

            df: pd.DataFrame = df.drop(columns=["code"])
            df: pd.DataFrame = cheak_dataframe_cols(df, datafeed_cls)
         
            if df["close"].dropna().empty:
                logger.warning(f"{code} close全为NaN,无有效数据，跳过...")
                continue

            datafeed:bt.feed.MetaAbstractDataBase = datafeed_cls(dataname=df.sort_index())
            self.cerebro.adddata(datafeed, name=code)

        logger.success("数据加载完毕！")

    def add_strategy(self, strategy: bt.Strategy, *args, **kwargs) -> None:

        self.cerebro.addstrategy(strategy, *args, **kwargs)

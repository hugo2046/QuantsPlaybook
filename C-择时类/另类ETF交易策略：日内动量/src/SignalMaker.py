"""
Author: Hugo
Date: 2024-08-13 15:30:53
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-08-13 15:31:04
Description: 

日内动量所需信号生成器
"""

from typing import List

import numpy as np
import pandas as pd

__all__ = ["NoiseArea"]


class NoiseArea:
    """
    另类ETF交易策略：日内动量
    ----------------------------

    日内动量策略通常源自买卖双方的力量在一段时间内, 存在持续且显著的不平衡。
    但由于股票市场的噪声水平较高, 故我们需要定义一个买卖双方力量平衡时, 价格
    的正常波动区域, 并称之为噪声区域。若价格在噪声区域中波动, 则认为不存在日
    内趋势。

    Attributes:
        ohlcv (pd.DataFrame): OHLCV数据
        pivot_frame (pd.DataFrame): 透视表，包含每个交易时间点的收盘价、开盘价和成交量
        close (pd.DataFrame): 收盘价数据
        open (pd.DataFrame): 开盘价数据

    Methods:
        calculate_intraday_vwap() -> pd.DataFrame:
            计算日内成交量加权平均价

        calculate_intraday_price_distance() -> pd.DataFrame:
            计算日内开盘价与收盘价的距离

        calculate_sigma(window: int = 14) -> pd.DataFrame:
            计算波动率

        calculate_bound(window: int = 14, method: str = "U") -> pd.DataFrame:
            计算上下边界

        fit(window: int = 14) -> pd.DataFrame:
            拟合模型，返回包含上下边界和日内成交量加权平均价的数据表
    """

    def __init__(self, ohlcv: pd.DataFrame) -> None:
        """
        初始化SignalMaker对象。

        参数:
            ohlcv (pd.DataFrame): 包含code,trade_time,open,close,volume字段的DataFrame。

        属性:
            ohlcv (pd.DataFrame): 输入的ohlcv数据。
            pivot_frame (pd.DataFrame): 根据trade_time和code进行透视的ohlcv数据。
            close (pd.DataFrame): 透视数据中的close列。
            open (pd.DataFrame): 在09:30:00时刻的open列。
        """
        self.ohlcv: pd.DataFrame = ohlcv
        self.pivot_frame: pd.DataFrame = pd.pivot_table(
            ohlcv,
            index="trade_time",
            columns="code",
            values=["close", "open", "volume"],
        )

        self.close: pd.DataFrame = self.pivot_frame["close"]
        self.open: pd.DataFrame = self.pivot_frame.at_time("09:30:00")["open"]

    def calculate_intraday_vwap(self) -> pd.DataFrame:
        """计算日内成交量加权平均价"""
        return self.pivot_frame.groupby(
            self.pivot_frame.index.date, group_keys=False
        ).apply(
            lambda df: (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        )

    def calculate_intraday_price_distance(self) -> pd.DataFrame:
        """计算日内开盘价与收盘价的距离"""
        pct_chg: pd.DataFrame = (
            self.close.div(self.open.reindex(self.close.index).ffill()) - 1
        )

        return pct_chg.abs()

    def calculate_sigma(self, window: int = 14) -> pd.DataFrame:
        """计算波动率"""

        distance: pd.DataFrame = self.calculate_intraday_price_distance()

        return distance.groupby(distance.index.time, group_keys=False).apply(
            lambda ser: ser.rolling(window=window).mean()
        )

    def calculate_bound(self, window: int = 14, method: str = "U") -> pd.DataFrame:
        """计算上下边界"""
        sigma: pd.DataFrame = self.calculate_sigma(window)

        idx: pd.DatetimeIndex = sigma.index
        sigma.index = sigma.index.normalize()

        daily_idx: pd.DatetimeIndex = self.open.index.normalize()
        cols: List[str] = self.open.columns

        if method.upper() == "U":

            threshold: pd.DataFrame = pd.DataFrame(
                np.maximum(
                    self.open.values, self.close.at_time("15:00:00").shift(1).values
                ),
                index=daily_idx,
                columns=cols,
            )
            out: pd.DataFrame = threshold.mul(1 + sigma, axis=0)

        elif method.upper() == "L":
            threshold: pd.DataFrame = pd.DataFrame(
                np.minimum(
                    self.open.values, self.close.at_time("15:00:00").shift(1).values
                ),
                index=daily_idx,
                columns=cols,
            )
            out: pd.DataFrame = threshold.mul(1 - sigma, axis=0)

        out.index = idx
        return out

    def concat_signal(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        拼接信号

        参数:
            window (int): 计算信号的窗口大小

        返回:
            pd.DataFrame: 包含上下边界和日内成交量加权平均价的数据表
        """

        upperbound: pd.DataFrame = self.calculate_bound(window, method="U")
        lowerbound: pd.DataFrame = self.calculate_bound(window, method="L")
        vwaps: pd.DataFrame = self.calculate_intraday_vwap()
        signal: pd.Series = self.ohlcv.set_index(["trade_time", "code"])[
            "close"
        ].to_frame(name="signal")
        return (
            pd.concat(
                [
                    data.set_index(["trade_time", "code"]),
                    upperbound.stack().to_frame(name="upperbound"),
                    signal,
                    lowerbound.stack().to_frame(name="lowerbound"),
                    vwaps.stack().to_frame(name="vwap"),
                ],
                axis=1,
            )
            .reset_index()
            .sort_values(["trade_time", "code"])
        )

    def fit(self, window: int = 14) -> pd.DataFrame:

        return self.concat_signal(self.ohlcv, window)

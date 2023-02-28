"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-12 17:04:44
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-12 17:08:24
Description: 
"""
import empyrical as ep
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from talib import EMA


class Factor_Calculator(BaseEstimator, TransformerMixin):
    def __init__(self, price: pd.DataFrame) -> None:

        self.price = price
        self.price.index = pd.to_datetime(self.price.index)
        self.close_df: pd.DataFrame = price["close"]
        self.low_df: pd.DataFrame = price["low"]
        self.high_df: pd.DataFrame = price["high"]
        self.open_df: pd.DataFrame = price["open"]
        self.vol_df: pd.DataFrame = price["vol"]
        self.amount_df: pd.DataFrame = price["amount"]

        if "turnover_rate_f" in price.columns.levels[0]:
            self.is_null = False
            self.turnover_df: pd.DataFrame = price["turnover_rate_f"]
        else:
            self.is_null = True
            print("turnover_rate_f缺失!")

    def fit(self, X, y=None) -> pd.DataFrame:

        return self

    def transform(
        self,
        factor_name: str,
        window: int = None,
        window1: int = None,
        window2: int = None,
    ) -> pd.DataFrame:

        return getattr(self, factor_name)(window, window1, window2)

    def second_order_mom(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 动量-二阶动量
        step_a: pd.DataFrame = (
            self.close_df
            - self.close_df.apply(
                lambda x: x.dropna().rolling(window1).mean(engine="numba")
            ).shift(1)
        ).div(self.close_df)

        return (step_a - step_a.shift(window2)).apply(
            lambda x: np.nan if x.dropna().empty else EMA(x.dropna(), window)
        )

    def diff_period_mom(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 动量-动量期限差

        if window1 <= window2:
            raise ValueError(
                f"diff_period_mom:window1({window1})不能小于window2({window2})"
            )

        return (self.close_df - self.close_df.shift(window1)) / self.close_df.shift(
            window1
        ) - (self.close_df - self.close_df.shift(window2)) / self.close_df.shift(
            window2
        )

    def amount_std(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 波动率-成交金额波动
        return -self.amount_df.apply(
            lambda x: x.dropna().rolling(window).std(engine="numba")
        )

    def volume_std(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 波动率-成交量波动率
        return -self.vol_df.apply(
            lambda x: x.dropna().rolling(window).std(engine="numba")
        )

    def turnover_pct(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:

        if self.is_null:
            raise ValueError("数据中turnover_rate_f不存在!")
        # 换手率变化

        if window1 <= window2:
            raise ValueError(f"turnover_pct:window1({window1})不能小于window2({window2})")

        avg1: pd.DataFrame = self.turnover_df.apply(
            lambda x: x.dropna().rolling(window1).mean()
        )
        avg2: pd.DataFrame = self.turnover_df.apply(
            lambda x: x.dropna().rolling(window2).mean()
        )
        return avg1.div(avg2)

    def long_short(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 多空对比总量
        return -((self.close_df - self.low_df).div(self.high_df - self.close_df)).apply(
            lambda x: x.dropna().rolling(window).sum(engine="numba")
        )

    def long_short_pct(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 多空对比变化
        if window1 <= window2:
            raise ValueError(f"long_short_pct:window1({window1})不能小于window2({window2})")

        step_a: pd.DataFrame = (
            self.close_df - self.low_df - self.high_df + self.close_df
        ).div(self.high_df - self.low_df)

        step_b: pd.DataFrame = self.vol_df * step_a

        step_c: pd.DataFrame = step_b.apply(lambda x: EMA(x.dropna(), window1))
        step_d: pd.DataFrame = step_b.apply(lambda x: EMA(x.dropna(), window2))
        return step_c.sub(step_d)

    def price_vol_rank_cov(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 量价背离协方差
        rank_close: pd.DataFrame = self.close_df.apply(
            lambda x: x.dropna().rolling(window).rank()
        )
        rank_vol: pd.DataFrame = self.vol_df.apply(
            lambda x: x.dropna().rolling(window).rank()
        )

        cov: pd.DataFrame = rank_close.rolling(window).cov(other=rank_vol)

        return -cov

    def price_vol_cor(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 量价相关系数
        return -self.close_df.rolling(window).corr(self.vol_df)

    def price_divergence(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 一阶量价背离
        rank_vol: pd.DataFrame = self.vol_df.pct_change().apply(
            lambda x: x.dropna().rolling(window).rank()
        )
        rank_price: pd.DataFrame = (self.close_df.div(self.open_df) - 1).apply(
            lambda x: x.dropna().rolling(window).rank()
        )
        return -rank_vol.rolling(window).corr(rank_price)

    def price_amf(
        self, window: int = None, window1: int = None, window2: int = None
    ) -> pd.DataFrame:
        # 量幅同向因子
        rank_vol: pd.DataFrame = self.vol_df.pct_change().apply(
            lambda x: x.dropna().rolling(window).rank()
        )

        rank_price: pd.DataFrame = (self.high_df.div(self.low_df) - 1).apply(
            lambda x: x.dropna().rolling(window).rank()
        )
        return rank_vol.rolling(window).corr(rank_price)


def calc_group_returns(
    factor_data: pd.DataFrame, values_col: str = "1D"
) -> pd.DataFrame:
    """计算各组因子收益率

    Args:
        factor_data (pd.DataFrame): alphalens经get_clean_factor_and_forward_returns处理后的数据
        values_col (str, optional): _description_. Defaults to "1D".

    Returns:
        pd.DataFrame: index-date columns-group_num values-returns
    """
    return pd.pivot_table(
        factor_data.reset_index(),
        index="date",
        columns="factor_quantile",
        values=values_col,
    )


def calc_annual_return(factor_data, sel_name: str = None) -> np.float32:

    rets: pd.DataFrame = calc_group_returns(factor_data)

    if sel_name:
        rets["Hedging"] = rets[5] - rets[1]
        return ep.annual_return(rets)[sel_name]
    else:
        return ep.annual_return(rets).mean()


def calc_ic_avg(factor_data) -> np.float32:

    icir: pd.Series = factor_data.groupby(level="date").apply(
        lambda x: x["1D"].corr(x["factor"], method="spearman")
    )

    return icir.mean()


def calc_mono_score(
    factor_data: pd.DataFrame, values_col: str = "1D", is_abs: bool = False
) -> np.float32:
    """中单调性得分

    Args:
        factor_data (pd.DataFrame): alphalens经get_clean_factor_and_forward_returns处理后的数据
        values_col (str, optional): 因子收益的列名. Defaults to "1D".

    Returns:
        np.float32: 中单调性得分
    """
    max_group_num: int = factor_data["factor_quantile"].max()
    if max_group_num != 5:

        raise ValueError(f"计算Mono Score单调性得分需要分五组进行分析!(当前最大分组为:{max_group_num})")

    group_annual_ret: pd.Series = ep.annual_return(
        calc_group_returns(factor_data, values_col)
    )
    score: np.float32 = (group_annual_ret[5] - group_annual_ret[1]) / (
        group_annual_ret[4] - group_annual_ret[2]
    )
    return np.abs(score) if is_abs else score


def clac_factor_cumulative(
    factor_data: pd.DataFrame,
    values_col: str = "1D",
    usedf: bool = False,
    calc_excess: bool = False,
) -> pd.DataFrame:

    factor_rets: pd.DataFrame = calc_group_returns(factor_data, values_col)
    if calc_excess:
        max_col: int = factor_rets.columns.max()
        factor_rets["Hedging"] = factor_rets[max_col] - factor_rets[1]

    factor_cums: pd.DataFrame = ep.cum_returns(factor_rets)

    return factor_cums if usedf else transform2snsdata(factor_cums, "Cum")


def transform2snsdata(df: pd.DataFrame, col_name: str) -> pd.DataFrame:

    return df.stack().to_frame(col_name).reset_index()

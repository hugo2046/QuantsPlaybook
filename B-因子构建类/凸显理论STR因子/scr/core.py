'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-04-11 15:34:21
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-04-11 15:41:59
Description: 计算组件
'''

import pandas as pd
import numpy as np


def calc_sigma(df: pd.DataFrame, bench: pd.Series = None) -> pd.DataFrame:
    """计算sigma

    Args:
        df (pd.DataFrame): 当日截面pct_chg
        bench (pd.Series, optional): 指数收益序列 index-datetime values-pct_chg. Defaults to None.
                                     当为None时,使用截面上的所有股票的平均收益率作为benchmark

    Returns:
        pd.Series: index-datetime columns-code values-sigma
    """

    if bench is None:
        bench: pd.DataFrame = df.mean(axis=1)

    a: pd.DataFrame = df.sub(bench, axis=0).abs()
    b: pd.DataFrame = df.abs().add(bench.abs(), axis=0) + 0.1

    return a.div(b)


def calc_weight(sigma: pd.DataFrame, delta: float = 0.7) -> pd.DataFrame:
    """计算权重

    Args:
        sigma (pd.DataFrame): index-datetime columns-code values-sigma

    Returns:
        pd.DataFrame: index-datetime columns-code values-weight
    """

    rank: pd.DataFrame = sigma.rank(axis=1,ascending=False)

    a: pd.DataFrame = rank.apply(lambda x: np.power(delta, x), axis=1)
    # b: pd.DataFrame = a.apply(lambda x: np.multiply(x, 1 / len(x)), axis=1).sum(axis=1)
    b: pd.DataFrame = a.mean(axis=1)
    return a.div(b, axis=0)
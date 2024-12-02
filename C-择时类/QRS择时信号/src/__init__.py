"""
Author: Hugo
Date: 2024-10-25 16:56:52
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-10-28 14:31:24
Description: 存放此次案例所需函数
"""

import pandas as pd
from typing import List
from .plotting_utils import calculate_bin_means


def concat_signal_vs_forward_returns(
    signal: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    将信号数据和前瞻收益数据合并为一个 DataFrame。

    :param signal: 信号数据，类型为 pandas.DataFrame。
    :param forward_returns: 前瞻收益数据，类型为 pandas.DataFrame。
    :return: 合并后的 DataFrame，包含信号和前瞻收益数据。
    """

    return (
        pd.concat({"signal": signal, "forward_returns": forward_returns})
        .stack()
        .unstack(level=[2, 0])
        .sort_index(axis=1)
    )


def concat_ohlc_vs_signal(
    data: pd.DataFrame, signal: pd.DataFrame, target_codes: List[str] = None
) -> pd.DataFrame:
    """
    将信号数据添加到目标代码的OHLCV数据中。

    :param data: 包含OHLCV数据的DataFrame
    :param signal: 包含信号数据的DataFrame
    :param target_codes: 目标代码列表，如果为None，则使用信号数据的所有列
    :return: 合并后的DataFrame
    """
    if target_codes is None:
        target_codes: pd.Index = signal.columns

    # 初始化结果列表
    dfs: List[pd.DataFrame] = []
    for code in target_codes:
        # 筛选出目标代码的OHLCV数据
        df: pd.DataFrame = data.query("code==@code")[
            ["code", "open", "high", "low", "close", "volume"]
        ]
        # 对齐信号数据和OHLCV数据
        signal_df, df = signal.align(df, join="left", axis=0)
        # 合并信号数据和OHLCV数据
        ohlcvs: pd.DataFrame = pd.concat(
            (df, signal_df[code].to_frame(name="signal")), axis=1
        )
        # 添加到结果列表
        dfs.append(ohlcvs)
    return pd.concat(dfs)


def calc_signal_bins_corr(
    signal_and_forward_return: pd.DataFrame, step: float = 0.01, threshold: int = None
) -> float:
    """
    计算信号与未来收益的相关性。

    :param signal_and_forward_return: 包含信号和未来收益的 DataFrame，列索引的第一级表示不同的信号，第二级表示信号和收益。
    :type signal_and_forward_return: pd.DataFrame
    :param step: 计算 bin 均值时的步长，默认为 0.01。
    :type step: float, optional
    :param threshold: 过滤 bin 中元素数量的阈值，默认为 None。如果为 None，则过滤掉元素数量小于等于 5 的 bin。
    :type threshold: int, optional
    :return: 信号与未来收益的相关性。
    :rtype: float
    """

    def _calc_corr(df) -> float:
        df: pd.DataFrame = df.droplevel(0, axis=1)
        test_ser: pd.Series = calculate_bin_means(df, step=step)
        if not threshold:

            test_ser: pd.Series = test_ser.query("count>5")["mean"]
        else:
            test_ser: pd.Series = test_ser["mean"]

        test_ser.index = test_ser.index.map(lambda x: x.mid)
        test_ser: pd.DataFrame = test_ser.to_frame(name="returns").reset_index()
        return test_ser["signal"].corr(test_ser["returns"])

    return signal_and_forward_return.groupby(axis=1, level=0).apply(_calc_corr)

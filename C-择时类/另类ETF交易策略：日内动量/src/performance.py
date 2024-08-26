"""
Author: Hugo
Date: 2024-08-21 09:03:53
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-08-21 09:05:43
Description: 
"""

from typing import Dict, List
import pandas as pd
import pyfolio as pf
from .utils import get_strategy_return, trans_minute_to_daily, check_index_tz


__all__ = [
    "show_perf_stats",
    "multi_asset_show_perf_stats",
    "multi_strategy_show_perf_stats",
]


def show_perf_stats(strat, minutes_close: pd.Series = None, return_df: bool = False):
    """
    展示策略的绩效指标。

    参数：
        - returns: pd.Series 对象，策略的收益率数据。
        - benchmark: pd.Series 对象，基准的收益率数据。
    """

    returns: pd.Series = get_strategy_return(strat)

    if minutes_close is not None:
        benchmark_rets: pd.Series = (
            trans_minute_to_daily(minutes_close).pct_change()
            if minutes_close is not None
            else None
        )

        benchmark_rets: pd.DataFrame = check_index_tz(benchmark_rets)
    else:
        benchmark_rets = None
        
    returns: pd.DataFrame = check_index_tz(returns)

    return pf.plotting.show_perf_stats(
        returns, factor_returns=benchmark_rets, return_df=return_df
    )


def multi_asset_show_perf_stats(
    strats: Dict, minutes_close: pd.DataFrame = None
) -> pd.DataFrame:
    """
    多资产展示绩效统计信息。
    参数：
    - strats (Dict): 策略字典，包含多个资产的策略。
    - minutes_close (pd.DataFrame, 可选): 分钟级别的收盘价数据，默认为None。
    返回：
    - pd.DataFrame: 合并后的绩效统计信息数据框。
    示例：
    ```python
    strats = {
        "asset1": strat1,
        "asset2": strat2,
        ...
    }
    result = multi_asset_show_perf_stats(strats, minutes_close)
    ```
    """
    dfs: List[pd.DataFrame] = []

    for code, strat in strats.items():
        strat_minutes_close: pd.Series = (
            minutes_close.query("code == @code")["close"]
            if minutes_close is not None
            else None
        )
        dfs.append(show_perf_stats(strat, strat_minutes_close, return_df=True))

    return pd.concat(dfs, keys=list(strats.keys()), axis=1)


def multi_strategy_show_perf_stats(
    strats: Dict, minutes_close: pd.DataFrame = None
) -> pd.DataFrame:
    """
    多策略展示绩效统计信息。
    参数：
    - strats (Dict): 策略字典，包含多个策略名称和策略对象的键值对。
    - minutes_close (pd.DataFrame, 可选): 分钟级别的收盘价数据框，用于计算绩效统计信息。默认为None。
    返回：
    - pd.DataFrame: 合并后的绩效统计信息数据框，包含每个策略的绩效统计信息。
    示例：
    ```python
    strats = {
        "策略1": strategy1,
        "策略2": strategy2,
        ...
    }
    result = multi_strategy_show_perf_stats(strats, minutes_close)
    ```
    """

    dfs: List[pd.DataFrame] = []

    for strategy_name, strat in strats.items():
        code: str = strat.data._name
        strat_minutes_close: pd.Series = (
            minutes_close.query("code == @code")["close"]
            if minutes_close is not None
            else None
        )
        dfs.append(show_perf_stats(strat, strat_minutes_close, return_df=True))

    return pd.concat(dfs, keys=list(strats.keys()), axis=1)

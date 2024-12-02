"""
Author: Hugo
Date: 2024-08-21 09:03:53
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-08-21 09:05:43
Description: 
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import pyfolio as pf

from .utils import check_index_tz, get_strategy_return, trans_minute_to_daily

__all__ = [
    "show_perf_stats",
    "show_trade_stats",
    "multi_asset_show_perf_stats",
    "multi_strategy_show_perf_stats",
    "multi_strategy_show_trade_stats",
]


def show_trade_stats(strat) -> pd.DataFrame:
    """
    显示交易策略的统计信息，包括年化收益率、夏普比率、最大回撤比例、交易次数、胜率和盈亏比。

    :param strat: 交易策略实例
    :return: 包含交易统计信息的 DataFrame
    """

    def get_analyzer_value(
        analyzer_name: str, key: str, subkey: str = None, default=np.nan
    ):
        """
        获取分析器的值。

        :param analyzer_name: 分析器名称
        :param key: 主键
        :param subkey: 子键（可选）
        :param default: 默认值
        :return: 分析器的值
        """
        analyzer = strat.analyzers.getbyname(analyzer_name).get_analysis()
        if subkey:
            return analyzer.get(key, {}).get(subkey, default)
        return analyzer.get(key, default)

    # 初始化统计信息
    stats = {
        "Annual return": get_analyzer_value("annual_return", "rnorm100"),
        "Sharpe ratio": get_analyzer_value("sharpe_ratio", "sharperatio"),
        "Max drawdown": get_analyzer_value("drawdown", "max", "drawdown"),
        "Trade Num": np.nan,
        "Win Rate": np.nan,
        "Win Loss Ratio": np.nan,
    }

    # 获取交易分析器信息
    trade_analyzer: Dict = (
        strat.analyzers.getbyname("trade_analyzer").get_analysis().get("long", {})
    )
    if trade_analyzer:
        total = trade_analyzer.get("total", np.nan)
        won = trade_analyzer.get("won", np.nan)
        stats["Trade Num"] = total
        stats["Win Rate"] = (won / total)*100 if total else np.nan
        pnl: Dict = trade_analyzer.get("pnl", {})
        if pnl:
            avg_win = pnl.get("won", {}).get("average", np.nan)
            avg_loss = pnl.get("lost", {}).get("average", np.nan)
            stats["Win Loss Ratio"] = avg_win / abs(avg_loss) if avg_loss else np.nan

    # 格式化统计信息
    def format_stat(value, is_percentage=False):
        if np.isnan(value):
            return np.nan
        return f"{value:.2f}%" if is_percentage else f"{value:.2f}"

    stats["Annual return"] = format_stat(stats["Annual return"], is_percentage=True)
    stats["Sharpe ratio"] = format_stat(stats["Sharpe ratio"])
    stats["Max drawdown"] = format_stat(stats["Max drawdown"], is_percentage=True)
    stats["Win Rate"] = format_stat(stats["Win Rate"], is_percentage=True)
    stats["Win Loss Ratio"] = format_stat(stats["Win Loss Ratio"])

    return pd.Series(stats).to_frame(name="Trade Stats")


def show_perf_stats(strat, minutes_close: pd.Series = None, usdf: bool = False):
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
        returns, factor_returns=benchmark_rets, usdf=usdf
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
        dfs.append(show_perf_stats(strat, strat_minutes_close, usdf=True))

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


def multi_strategy_show_trade_stats(strats: Dict) -> pd.DataFrame:
    """
    多策略展示交易统计信息。
    参数：
    - strats (Dict): 策略字典，包含多个策略名称和策略对象的键值对。
    返回：
    - pd.DataFrame: 合并后的交易统计信息数据框，包含每个策略的交易统计信息。
    示例：
    ```python
    strats = {
        "策略1": strategy1,
        "策略2": strategy2,
        ...
    }
    result = multi_strategy_show_trade_stats(strats)
    ```
    """

    return pd.concat(
        (show_trade_stats(strat) for strat in strats.values()),
        keys=list(strats.keys()),
        axis=1,
    )

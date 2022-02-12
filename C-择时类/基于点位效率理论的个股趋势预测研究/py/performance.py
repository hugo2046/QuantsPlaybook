'''
Author: Hugo
Date: 2021-06-16 14:45:53
LastEditTime: 2021-10-29 14:11:08
LastEditors: Please set LastEditors
Description: 组合评价相关函数
'''

import numpy as np
import pandas as pd
import empyrical as ep
# from .windapi_support import get_wsd_data
# from . import timeseries as ts

from typing import (List, Tuple, Union)


def _adjust_returns(returns, adjustment_factor):
    """
    Returns a new :py:class:`pandas.Series` adjusted by adjustment_factor.
    Optimizes for the case of adjustment_factor being 0.

    Parameters
    ----------
    returns : :py:class:`pandas.Series`
    adjustment_factor : :py:class:`pandas.Series` / :class:`float`

    Returns
    -------
    :py:class:`pandas.Series`
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns.copy()
    return returns - adjustment_factor


def information_ratio(returns, factor_returns):
    """
    Determines the Information ratio of a strategy.

    Parameters
    ----------
    returns : :py:class:`pandas.Series` or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns: :class:`float` / :py:class:`pandas.Series`
        Benchmark return to compare returns against.

    Returns
    -------
    :class:`float`
        The information ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/information_ratio for more details.

    """
    if len(returns) < 2:
        return np.nan

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.mean(active_return) / tracking_error


# 风险指标
def Strategy_performance(returns: pd.DataFrame,
                         mark_benchmark: str = 'benchmark',
                         periods: str = 'daily') -> pd.DataFrame:
    '''
    风险指标计算

    returns:index-date col-数据字段
    mark_benchmark:用于指明基准
    periods：频率
    '''

    df: pd.DataFrame = pd.DataFrame()

    df['年化收益率'] = ep.annual_return(returns, period=periods)

    df['累计收益'] = returns.apply(lambda x: ep.cum_returns(x).iloc[-1])

    df['波动率'] = returns.apply(
        lambda x: ep.annual_volatility(x, period=periods))

    df['夏普'] = returns.apply(ep.sharpe_ratio, period=periods)

    df['最大回撤'] = returns.apply(lambda x: ep.max_drawdown(x))

    df['索提诺比率'] = returns.apply(lambda x: ep.sortino_ratio(x, period=periods))

    df['Calmar'] = returns.apply(lambda x: ep.calmar_ratio(x, period=periods))

    # 相对指标计算
    if mark_benchmark in returns.columns:

        select_col = [col for col in returns.columns if col != mark_benchmark]
        df['IR'] = returns[select_col].apply(
            lambda x: information_ratio(x, returns[mark_benchmark]))

        df['Alpha'] = returns[select_col].apply(
            lambda x: ep.alpha(x, returns[mark_benchmark], period=periods))

        df['Beta'] = returns[select_col].apply(
            lambda x: ep.beta(x, returns[mark_benchmark]))

        # 计算相对年化波动率
        df['超额收益率'] = df['年化收益率'] - \
            df.loc[mark_benchmark, '年化收益率']

    return df.T


# def show_worst_drawdown_periods(returns: pd.Series,
#                                 benchmark_code: str = "000300.SH",
#                                 top: int = 5):
#     """
#     Prints information about the worst drawdown periods.

#     Prints peak dates, valley dates, recovery dates, and net
#     drawdowns.

#     Parameters
#     ----------
#     returns : pd.Series
#         Daily returns of the strategy, noncumulative.
#          - See full explanation in tears.create_full_tear_sheet.
#     top : int, optional
#         Amount of top drawdowns periods to plot (default 5).
#     """

#     drawdown_df = ts.gen_drawdown_table(returns, top=top)
#     drawdown_df.index = list(range(1, len(drawdown_df) + 1))

#     phase_change = compare_phase_change(returns, benchmark_code, top)

#     df = pd.concat((drawdown_df, phase_change), axis=1)

#     # print_table(
#     #     df.sort_values('区间最大回撤 %', ascending=False),
#     #     name='序号',
#     #     float_format='{0:.2f}'.format,
#     # )

#     return df

# def compare_phase_change(returns: pd.Series,
#                          benchmark_code: str,
#                          top: int = 5) -> pd.DataFrame:
#     '''
#     对比策略与基准在回撤区间内的收益
#     ------
#         returns:策略净值收益率
#         benchmark_code:基准的代码
#     '''

#     beginDt = returns.index.min()
#     endDt = returns.index.max()

#     benchmark = get_wsd_data(benchmark_code,
#                              'pct_chg',
#                              beginDt,
#                              endDt,
#                              'priceAdj=B',
#                              usedf=True)

#     benchmark = benchmark['PCT_CHG'] / 100

#     df = pd.DataFrame(columns=['策略收益%', '基准收益%'],
#                       index=list(range(1, top + 1)))

#     drawdowns_list = ts.get_top_drawdowns(returns, top=top)

#     for i, v in enumerate(drawdowns_list):

#         peak_date, _, recovery_date = v

#         if pd.isnull(recovery_date):

#             df.loc[i + 1, '策略收益%'] = np.nan
#             df.loc[i + 1, '基准收益'] = np.nan

#         else:
#             df.loc[i + 1, '策略收益%'] = ep.cum_returns(
#                 returns.loc[peak_date:recovery_date]).iloc[-1]
#             df.loc[i + 1, '基准收益%'] = ep.cum_returns(
#                 benchmark.loc[peak_date:recovery_date])[-1]

#     return df
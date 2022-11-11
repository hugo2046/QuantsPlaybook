'''
Author: Hugo
Date: 2021-06-16 14:45:53
LastEditTime: 2022-11-11 16:58:02
LastEditors: hugo2046 shen.lan123@gmail.com
Description: 组合评价相关函数
'''
import empyrical as ep
import numpy as np
import pandas as pd


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
    periods:频率
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

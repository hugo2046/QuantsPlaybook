import pandas as pd
import empyrical as ep
import numpy as np
from scipy import stats


def get_performance_table(return_df: pd.DataFrame,
                          benchmark_name: str = None,
                          periods: str = 'daily') -> pd.DataFrame:
    """收益指标

    Args:
        return_df (pd.DataFrame): 收益率表格
        benchmark_name (str): 基准的列名
        periods (str, optional): 频率. Defaults to 'daily'.

    Returns:
        pd.DataFrame
    """
    ser: pd.DataFrame = pd.DataFrame()
    ser['年化收益率'] = ep.annual_return(return_df, period=periods)
    ser['累计收益'] = ep.cum_returns(return_df).iloc[-1]
    ser['波动率'] = return_df.apply(
        lambda x: ep.annual_volatility(x, period=periods))
    ser['夏普'] = return_df.apply(ep.sharpe_ratio, period=periods)
    ser['最大回撤'] = return_df.apply(lambda x: ep.max_drawdown(x))

    if benchmark_name is not None:

        select_col = [
            col for col in return_df.columns if col != benchmark_name
        ]

        ser['IR'] = return_df[select_col].apply(
            lambda x: information_ratio(x, return_df[benchmark_name]))
        ser['Alpha'] = return_df[select_col].apply(
            lambda x: ep.alpha(x, return_df[benchmark_name], period=periods))

        ser['超额收益'] = ser['年化收益率'] - ser.loc[benchmark_name,
                                             '年化收益率']  #计算相对年化波动率

    return ser.T


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


def get_information_table(ic_data: pd.DataFrame) -> pd.DataFrame:
    """计算IC相关指标

    Args:
        ic_data (pd.DataFrame): index-date columns-IC

    Returns:
        pd.DataFrame
    """
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    return ic_summary_table.apply(lambda x: x.round(3)).T


def calc_mono_score(returns: pd.DataFrame) -> float:
    """计算单调性得分

    Args:
        returns (pd.DataFrame): MultiIndex level0-date level1-code columns-五分组
        列名必须为1，2，3，4，5

    Returns:
        float: Mono得分
    """
    group_mean = returns.mean()

    return (group_mean[5] - group_mean[1]) / (group_mean[4] - group_mean[2])
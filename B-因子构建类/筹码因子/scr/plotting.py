"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-03-27 15:02:44
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-03-29 10:50:17
Description: 画图
"""
from typing import List, Tuple

import empyrical as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
# from alphalens.utils import quantize_factor
from matplotlib import ticker
from scipy import stats

from .cyq import calc_dist_chips

sns.set_theme(style="whitegrid")
# plt中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt显示负号
plt.rcParams["axes.unicode_minus"] = False


############################# 画图计算用组件 #############################
def _get_score_ic(pred_label: pd.DataFrame):
    """

    :param pred_label:
    :return:
    """
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how="any", inplace=True)
    _ic = concat_data.groupby(level="datetime").apply(
        lambda x: x["label"].corr(x["score"])
    )
    _rank_ic = concat_data.groupby(level="datetime").apply(
        lambda x: x["label"].corr(x["score"], method="spearman")
    )
    return pd.DataFrame({"ic": _ic, "rank_ic": _rank_ic})


def _get_score_return(pred_label: pd.DataFrame, N: int = 5, **kwargs) -> pd.DataFrame:
    """预测值分组收益

    Args:
        pred_label (pd.DataFrame): _description_
        N (int, optional): 分组数. Defaults to 5.

    Returns:
        pd.DataFrame: _description_
    """
    pred_label_drop: pd.DataFrame = pred_label.dropna(subset=["score"])
    pred_label_drop["group"] = pred_label_drop.groupby(level="datetime")[
        "score"
    ].transform(lambda df: pd.qcut(df, N, labels=False, **kwargs) + 1)
    pred_label_drop["group"] = pred_label_drop["group"].apply(lambda x: "Group%d" % x)
    ts_df: pd.DataFrame = pd.pivot_table(
        pred_label_drop.reset_index(), index="datetime", columns="group", values="label"
    )

    ts_df["long-short"] = ts_df["Group5"] - ts_df["Group%d" % N]
    ts_df["long-average"] = (
        ts_df["Group5"] - pred_label.groupby(level="datetime")["label"].mean()
    )

    return ts_df


def _get_auto_correlation(pred_label: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """IC自回归系数

    Args:
        pred_label (pd.DataFrame): _description_
        lag (int, optional): _description_. Defaults to 1.

    Returns:
        pd.DataFrame: _description_
    """
    pred: pd.DataFrame = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument")["score"].shift(lag)
    ac = pred.groupby(level="datetime").apply(
        lambda x: x["score"].rank(pct=True).corr(x["score_last"].rank(pct=True))
    )
    return ac.to_frame("value")


def _get_group_turnover(
    pred_label: pd.DataFrame, N: int = 5, lag: int = 1
) -> pd.DataFrame:
    """计算组合换手率

    Args:
        pred_label (pd.DataFrame): _description_
        N (int, optional): 分组. Defaults to 5.
        lag (int, optional): 滞后期. Defaults to 1.

    Returns:
        pd.DataFrame: _description_
    """
    pred: pd.DataFrame = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument")["score"].shift(lag)
    top = pred.groupby(level="datetime").apply(
        lambda x: 1
        - x.nlargest(len(x) // N, columns="score")
        .index.isin(x.nlargest(len(x) // N, columns="score_last").index)
        .sum()
        / (len(x) // N)
    )
    bottom = pred.groupby(level="datetime").apply(
        lambda x: 1
        - x.nsmallest(len(x) // N, columns="score")
        .index.isin(x.nsmallest(len(x) // N, columns="score_last").index)
        .sum()
        / (len(x) // N)
    )
    return pd.DataFrame(
        {
            "Top": top,
            "Bottom": bottom,
        }
    )


############################# 筹码分布 #############################


def plot_dist_chips(
    df: pd.DataFrame,
    method: str,
    title: str = "",
    figsize: tuple = (14, 6),
    ax: plt.axes = None,
) -> plt.axes:
    """画筹码分布

    Args:
        df (pd.DataFrame): index-date columns-close|avg,high,low,vol,turnover_rate
        method (str): 计算分布的方法
            triang: 三角分布
            uniform: 平均分布
            turn_coeff: 换手率系数
        title (str, optional): 标题. Defaults to "".
        figsize (tuple, optional): figsize. Defaults to (14, 6).
        ax (plt.axes, optional): _description_. Defaults to None.

    Returns:
        plt.axes:
    """
    cum_vol: pd.Series = calc_dist_chips(df, method)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title)

    if method in {"triang", "uniform"}:
        ax.bar(x=cum_vol.index, height=cum_vol.values, width=0.01, edgecolor="#4c89bc")
        min_p, max_p = cum_vol.index.min(), cum_vol.index.max()
        ax.set_xlim(min_p - 0.5, max_p + 0.5)
    else:
        cum_vol.index = np.round(cum_vol.index, 4)
        cum_vol.plot.bar(ax=ax)
        ax.set_xticks(range(0, len(cum_vol), 5))

    ax.set_xlabel("price")
    ax.set_ylabel("volume")

    return ax


############################# 复合因子 #############################


def _calculate_mdd(cum_returns: pd.Series) -> pd.Series:
    """计算最大回撤"""
    return cum_returns - cum_returns.cummax()


def report_graph(report_df: pd.DataFrame, figsize: Tuple = None) -> plt.figure:
    df: pd.DataFrame = report_df[["return", "cost", "bench"]].copy()

    cum_frame: pd.DataFrame = (
        df.pipe(
            pd.DataFrame.assign, cum_return_with_cost=lambda x: x["return"] - x["cost"]
        )
        .pipe(pd.DataFrame.drop, columns="cost")
        .pipe(pd.DataFrame.apply, ep.cum_returns)
        .pipe(pd.DataFrame.rename, columns={"return": "cum_return_without_cost"})
        .pipe(pd.DataFrame.sort_index,axis=1,key=lambda x: x.str.lower())
    )

    if figsize is None:
        figsize = (18, 8)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    sns.lineplot(cum_frame, ax=axes[0])
    axes[0].axhline(0, ls="--", color="black")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "%.2f%%" % (x * 100))
    )
    _calculate_mdd(cum_frame["cum_return_with_cost"]).plot.area(
        ax=axes[1], color="#ea9393", label="cum_return_with_cost"
    )
    _calculate_mdd(cum_frame["cum_return_without_cost"]).plot.area(
        ax=axes[1], color="#d62728", label="cum_return_without_cost"
    )
    axes[1].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "%.2f%%" % (x * 100))
    )
    axes[1].set_ylabel("Drawdown")
    axes[1].axhline(0, color="black")
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.legend()
    return fig


def model_performance_graph(
    pred_label: pd.DataFrame,
    figsize: Tuple = None,
    N=5,
    lag=1,
    dist=stats.norm,
    **kwargs,
) -> plt.figure:
    figsize: Tuple = (18, 25) if figsize is None else figsize

    plt.close("all")
    fig = plt.figure(figsize=figsize)

    ts_cum_ax = plt.subplot2grid((6, 2), (0, 0), colspan=2)
    ls_hist_ax = plt.subplot2grid((6, 2), (1, 0))
    la_hist_ax = plt.subplot2grid((6, 2), (1, 1))
    ts_ic_ax = plt.subplot2grid((6, 2), (2, 0), colspan=2)
    ic_hist_ax = plt.subplot2grid((6, 2), (3, 0))
    ic_qq_ax = plt.subplot2grid((6, 2), (3, 1))
    auto_corr_ax = plt.subplot2grid((6, 2), (4, 0), colspan=2)
    turnover_ax = plt.subplot2grid((6, 2), (5, 0), colspan=2)

    # CumulativeReturn
    ts_df: pd.DataFrame = _get_score_return(pred_label, N=N, **kwargs)

    ts_cum_ax.set_title("Cumulative Return")
    sns.lineplot(data=ep.cum_returns(ts_df), ax=ts_cum_ax)
    ts_cum_ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "%.2f%%" % (x * 100))
    )
    ts_cum_ax.axhline(0, color="black", lw=1, ls="--")

    # ts_df:pd.DataFrame = ts_df.loc[:, ["long-short", "long-average"]]
    # _bin_size:float = float(((t_df.max() - t_df.min()) / 20).min())

    ls_hist_ax.set_title("Long-Short")
    sns.histplot(data=ts_df["long-short"], kde=True, ax=ls_hist_ax)

    la_hist_ax.set_title("Long-Average")
    sns.histplot(data=ts_df["long-average"], kde=True, ax=la_hist_ax)

    # IC
    ic_frame: pd.DataFrame = _get_score_ic(pred_label)
    ts_ic_ax.set_title("Score IC")
    sns.lineplot(data=ic_frame, markers=True, ax=ts_ic_ax)
    # QQ plot
    _plt_fig: plt.figure = sm.qqplot(
        ic_frame["ic"].dropna(), dist=dist, fit=True, line="45"
    )
    plt.close(_plt_fig)
    qqplot_data = _plt_fig.gca().lines

    dist_name = "Normal" if isinstance(dist, stats.norm.__class__) else "Unknown"

    sns.regplot(
        x=qqplot_data[0].get_xdata(),
        y=qqplot_data[0].get_ydata(),
        line_kws={"color": "red"},
        ax=ic_qq_ax,
    )
    ic_qq_ax.set_title(f"IC {dist_name} Dist. Q-Q")
    ic_qq_ax.set_ylabel("Observed Quantile")
    ic_qq_ax.set_xlabel(f"{dist_name} Distribution Quantile")

    ic_hist_ax.set_title("IC")
    sns.histplot(data=ic_frame["ic"].dropna(), kde=True, ax=ic_hist_ax)

    # AutoCorr
    _df: pd.DataFrame = _get_auto_correlation(pred_label, lag=lag)
    auto_corr_ax.set_title("Auto Correlation")
    sns.lineplot(data=_df, ax=auto_corr_ax)
    # Turnover
    r_df: pd.DataFrame = _get_group_turnover(pred_label, N, lag)
    turnover_ax.set_title("Top-Bottom Turnover")
    sns.lineplot(data=r_df, ax=turnover_ax)

    plt.tight_layout()
    return fig


def plot_score_ic(
    pred_label: pd.DataFrame,
    dist=stats.norm,
    ax: plt.axes = None,
    figsize: Tuple = None,
) -> plt.axes:
    """画IC,Rank_IC

    Args:
        pred_label (pd.DataFrame): MultiIndex-datetime,code columns-score,label
        ax (plt.axes, optional): Defaults to None.
        figsize (Tuple, optional): 画图大小. Defaults to None.

    Returns:
        plt.axes:
    """

    if figsize is None:
        figsize: Tuple = (18, 8)
    fig = plt.figure(figsize=figsize)

    ic_ts = plt.subplot(211)
    ic_hist = plt.subplot(223)
    ic_qq = plt.subplot(224)

    ic_frame: pd.DataFrame = _get_score_ic(pred_label)
    ic_ts.set_title("Score IC")
    sns.lineplot(data=ic_frame, markers=True, ax=ic_ts)
    # QQ plot
    _plt_fig = sm.qqplot(ic_frame["ic"].dropna(), dist=dist, fit=True, line="45")
    plt.close(_plt_fig)
    qqplot_data = _plt_fig.gca().lines

    dist_name = "Normal" if isinstance(dist, stats.norm.__class__) else "Unknown"

    sns.regplot(
        x=qqplot_data[0].get_xdata(),
        y=qqplot_data[0].get_ydata(),
        line_kws={"color": "red"},
        ax=ic_qq,
    )
    ic_qq.set_title(f"IC {dist_name} Dist. Q-Q")
    ic_qq.set_ylabel("Observed Quantile")
    ic_qq.set_xlabel(f"{dist_name} Distribution Quantile")

    ic_hist.set_title("IC")
    sns.histplot(data=ic_frame["ic"].dropna(), kde=True, ax=ic_hist)

    return fig


def plot_group_score_return(
    pred_label: pd.DataFrame, N: int = 5, figsize: Tuple = None, **kwargs
):
    if "show_long_short" in kwargs:
        show_long_short: bool = kwargs.get("show_long_short", False)
        del kwargs["show_long_short"]

    ts_df: pd.DataFrame = _get_score_return(pred_label, N, **kwargs)

    if figsize is None:
        figsize: Tuple = (18, 8)

    fig = plt.figure(figsize=figsize)
    ts_line = plt.subplot(211)
    long_short_hist = plt.subplot(223)
    long_avg_hist = plt.subplot(224)

    ts_line.set_title("Cumulative Return")

    select: List = [
        col
        for col in ts_df.columns.tolist()
        if col not in ("long-short", "long-average")
    ]
    if show_long_short:
        select: List = ts_df.columns.tolist()

    sns.lineplot(data=ep.cum_returns(ts_df[select]), ax=ts_line)
    ts_line.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ts_line.axhline(0, color="black", lw=1, ls="--")

    # t_df:pd.DataFrame = t_df.loc[:, ["long-short", "long-average"]]
    # _bin_size:float = float(((t_df.max() - t_df.min()) / 20).min())

    long_short_hist.set_title("Long-Short")
    sns.histplot(data=ts_df["long-short"], kde=True, ax=long_short_hist)

    long_avg_hist.set_title("Long-Average")
    sns.histplot(data=ts_df["long-average"], kde=True, ax=long_avg_hist)

    return fig


def plot_factor_autocorr(
    pred_label: pd.DataFrame, lag=1, ax: plt.axes = None, figsize: Tuple = None
) -> tuple:
    _df: pd.DataFrame = _get_auto_correlation(pred_label, lag=lag)

    figsize: Tuple = (18, 6) if figsize is None else figsize
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title("Auto Correlation")
    sns.lineplot(data=_df, ax=ax)

    return fig


def plot_group_turnover(
    pred_label: pd.DataFrame, N=5, lag=1, ax: plt.axes = None, figsize: Tuple = None
) -> tuple:
    r_df: pd.DataFrame = _get_group_turnover(pred_label, N, lag)

    figsize: Tuple = (18, 6) if figsize is None else figsize
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title("Top-Bottom Turnover")
    sns.lineplot(data=r_df, ax=ax)

    return fig

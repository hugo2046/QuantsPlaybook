"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-03 09:43:37
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-12 17:14:46
Description: 
"""

from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from alphalens.utils import (MaxLossExceededError,
                             get_clean_factor_and_forward_returns,
                             quantize_factor)

from .core import clac_factor_cumulative
from .opt_func import _get_err_msg_value

# from matplotlib.gridspec import GridSpec


plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def plot_ts_icir(
    factor_data: pd.DataFrame, value_col: str = "1D", title: str = None, ax=None
) -> plt.figure:
    """计算因子ICIR

    Args:
        factor_data (pd.DataFrame): alphalens经get_clean_factor_and_forward_returns处理后的数据
        value_col (str, optional): 因子收益列名. Defaults to "1D".
        title (str, optional): 图表名称. Defaults to None.

    Returns:
        plt.figure
    """
    if title is None:
        title: str = "Information Coefficient"

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 4))

    icir: pd.Series = factor_data.groupby(level="date").apply(
        lambda x: x[value_col].corr(x["factor"], method="spearman")
    )

    icir.plot(ax=ax, lw=0.5, alpha=0.8)
    icir.rolling(20).mean().plot(color="ForestGreen", lw=2, ax=ax)
    ax.set_title(title)
    ax.axhline(0.0, linestyle="-", color="black", lw=1, alpha=0.8)
    ax.legend(["ICIR", "1 month moving avg"], loc="upper right")

    ax.text(
        0.05,
        0.95,
        "Mean %.3f \n Std. %.3f" % (icir.mean(), icir.std()),
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 1, "pad": 5},
        transform=ax.transAxes,
        verticalalignment="top",
    )

    return ax


def plot_group_cumulative(factor_cums: pd.DataFrame) -> mpl.figure.Figure:
    fig = mpl.figure.Figure(figsize=(18, 7))
    subfigs = fig.subfigures(1, 2, width_ratios=[3, 1])
    # 分组累计收益情况
    (
        so.Plot(factor_cums, x="date", y="Cum")
        .facet(
            "factor_quantile",
            wrap=3,
        )
        .add(so.Line(alpha=0.3), group="factor_quantile", col=None)
        .add(so.Line(linewidth=3))
        # .layout(size=(18, 5), engine="constrained")
        .share(x=False)
        .label(title="{} Group Cumulative Rate".format)
        .scale(y=so.Continuous().label(like="{x:.2%}"))
        .on(subfigs[0])
        .plot()
    )
    # 分组累计收益率
    (
        so.Plot(factor_cums, x="factor_quantile", y="Cum")
        .add(so.Bar(alpha=1), so.Agg(lambda x: x.iloc[-1]))
        .label(title="Group Cumulative")
        .scale(y=so.Continuous().label(like="{x:.2%}"))
        .on(subfigs[1])
        .plot()
    )

    return fig


def plot_group_distribution(factor_data: pd.DataFrame):

    fig, axs = plt.subplots(1, 2, figsize=(18, 4))

    axs[0].set_title("Daily Return By Factor Quantile")
    sns.violinplot(factor_data, y="1D", x="factor_quantile", ax=axs[0])

    axs[1].set_title("Daily Factor Value By Quantile")
    sns.violinplot(factor_data, y="factor", x="factor_quantile", ax=axs[1])

    return axs


def plot_qlib_factor_dist(
    pred_label_df: pd.DataFrame,
    calc_excess: bool = True,
    title: str = "",
    no_raise:bool=False
) -> None:

    factor_data = pred_label_df.copy()
    factor_data.index.names = ["date", "asset"]
    factor_data.columns = ["factor", "1D"]

    factor_data["factor_quantile"] = quantize_factor(factor_data, no_raise=no_raise)

    factor_cums: pd.DataFrame = clac_factor_cumulative(
        factor_data, calc_excess=calc_excess
    )
    # constrained_layout=True
    fig = plt.figure(figsize=(18, 7 * 2))

    subfigs = fig.subfigures(2, 1, hspace=0.07)

    subfigsnest1 = subfigs[0].subfigures(1, 2, width_ratios=[3, 1])
    subfigsnest2 = subfigs[1].subplots(2, 2)  # subfigs[1].subplots(1, 2)
    gs = subfigsnest2[1, 1].get_gridspec()
    for ax in subfigsnest2[1, :]:
        ax.remove()

    # 分组累计收益情况
    (
        so.Plot(factor_cums, x="date", y="Cum")
        .facet(
            "factor_quantile",
            wrap=3,
        )
        .add(so.Line(alpha=0.3), group="factor_quantile", col=None)
        .add(so.Line(linewidth=3))
        # .layout(size=(18, 5), engine="constrained")
        .share(x=False)
        .label(title="{} Group Cumulative Rate".format)
        .scale(y=so.Continuous().label(like="{x:.2%}"))
        .on(subfigsnest1[0])
        .plot()
    )
    # 分组累计收益率
    (
        so.Plot(factor_cums, x="factor_quantile", y="Cum")
        .add(so.Bar(alpha=1), so.Agg(lambda x: x.iloc[-1]))
        .label(title="Group Cumulative")
        .scale(y=so.Continuous().label(like="{x:.2%}"))
        .on(subfigsnest1[1])
        .plot()
    )
    sns.set_theme(style="whitegrid")
    subfigsnest2[0, 0].set_title("Daily Return By Factor Quantile")
    sns.violinplot(
        factor_data,
        y="1D",
        x="factor_quantile",
        inner="quart",
        linewidth=1,
        ax=subfigsnest2[0, 0],
    )

    subfigsnest2[0, 1].set_title("Daily Factor Value By Quantile")
    sns.violinplot(
        factor_data,
        y="factor",
        x="factor_quantile",
        inner="quart",
        linewidth=1,
        ax=subfigsnest2[0, 1],
    )
    sns.despine(left=True, bottom=True)

    axbig = subfigs[1].add_subplot(gs[1, :])
    plot_ts_icir(factor_data, ax=axbig)
    sns.reset_defaults()

    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(title, fontsize="xx-large", y=1.02)

    return fig


def plot_factor_dist(
    factor: Union[pd.DataFrame, pd.Series],
    price: pd.DataFrame,
    calc_excess: bool = True,
    max_loss: float = 0.35,
    title: str = "",
) -> None:

    if isinstance(factor, pd.DataFrame):
        factor = factor.stack()
    try:
        factor_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            factor=factor, prices=price, quantiles=5, periods=(1,), max_loss=max_loss
        )
    except MaxLossExceededError as e:
        err_str: str = str(e)
        print(err_str)
        err_value: np.float32 = _get_err_msg_value(err_str)
        factor_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            factor=factor,
            prices=price,
            quantiles=5,
            periods=(1,),
            max_loss=err_value,
        )
    factor_cums: pd.DataFrame = clac_factor_cumulative(
        factor_data, calc_excess=calc_excess
    )
    # constrained_layout=True
    fig = plt.figure(figsize=(18, 7 * 2))

    subfigs = fig.subfigures(2, 1, hspace=0.07)

    subfigsnest1 = subfigs[0].subfigures(1, 2, width_ratios=[3, 1])
    subfigsnest2 = subfigs[1].subplots(2, 2)  # subfigs[1].subplots(1, 2)
    gs = subfigsnest2[1, 1].get_gridspec()
    for ax in subfigsnest2[1, :]:
        ax.remove()

    # 分组累计收益情况
    (
        so.Plot(factor_cums, x="date", y="Cum")
        .facet(
            "factor_quantile",
            wrap=3,
        )
        .add(so.Line(alpha=0.3), group="factor_quantile", col=None)
        .add(so.Line(linewidth=3))
        # .layout(size=(18, 5), engine="constrained")
        .share(x=False)
        .label(title="{} Group Cumulative Rate".format)
        .scale(y=so.Continuous().label(like="{x:.2%}"))
        .on(subfigsnest1[0])
        .plot()
    )
    # 分组累计收益率
    (
        so.Plot(factor_cums, x="factor_quantile", y="Cum")
        .add(so.Bar(alpha=1), so.Agg(lambda x: x.iloc[-1]))
        .label(title="Group Cumulative")
        .scale(y=so.Continuous().label(like="{x:.2%}"))
        .on(subfigsnest1[1])
        .plot()
    )
    sns.set_theme(style="whitegrid")
    subfigsnest2[0, 0].set_title("Daily Return By Factor Quantile")
    sns.violinplot(
        factor_data,
        y="1D",
        x="factor_quantile",
        inner="quart",
        linewidth=1,
        ax=subfigsnest2[0, 0],
    )

    subfigsnest2[0, 1].set_title("Daily Factor Value By Quantile")
    sns.violinplot(
        factor_data,
        y="factor",
        x="factor_quantile",
        inner="quart",
        linewidth=1,
        ax=subfigsnest2[0, 1],
    )
    sns.despine(left=True, bottom=True)

    axbig = subfigs[1].add_subplot(gs[1, :])
    plot_ts_icir(factor_data, ax=axbig)
    sns.reset_defaults()

    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(title, fontsize="xx-large", y=1.02)

    return

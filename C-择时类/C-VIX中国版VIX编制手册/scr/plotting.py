'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-07 10:09:17
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-07 14:03:02
Description: 
'''
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def plot_indicator(price: pd.Series,
                   indicator: pd.Series,
                   label: Tuple,
                   ylabel: Tuple,
                   title: str = '',
                   ax: mpl.axes = None) -> mpl.axes:
    """画指标与价格走势图

    Args:
        price (pd.Series): 价格 index-date value-close
        indicator (pd.Series): 指标 index-date value-指标
        label (Tuple): 标签1,标签2
        ylabel (Tuple): 标签1,标签2
        title (str, optional): 标题. Defaults to ''.
        ax (mpl.axes, optional): 轴. Defaults to None.

    Returns:
        mpl.axes: _description_
    """
    label1, label2 = label
    ylabel1, ylabel2 = ylabel

    if ax is None:
        ax = plt.gca()

    price.plot(ax=ax, color='#FFD700', title=title, label=label1)
    ax.set_ylabel(ylabel1)
    ax_twin = indicator.plot(ax=ax, secondary_y=True, color='r', label=label2)
    ax_twin.set_ylabel(ylabel2)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()

    ax.legend(handles1 + handles2, labels1 + labels2)
    return ax


def plot_qunatile_signal(price: pd.Series, signal: pd.Series, window: int,
                         bound: Tuple,title:str='') -> mpl.axes:
    """画价格与信号的关系图

    Args:
        price (pd.Series): 价格
        signal (pd.Series): 信号
        window (int): 滚动时间窗口
        bound (Tuple): bound[0]-上轨百分位数,bound[1]-下轨百分位数

    Returns:
        mpl.axes: _description_
    """
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(3, 1)

    ax1 = plt.subplot(gs[:2, :])
    ax2 = plt.subplot(gs[2:, :])

    price.plot(ax=ax1, title=title)

    signal.plot(ax=ax2, color='darkgray', label='signal')

    # 构建上下轨
    up, lw = bound
    ub: pd.Series = signal.rolling(window).apply(
        lambda x: np.percentile(x, up), raw=True)

    lb: pd.Series = signal.rolling(window).apply(
        lambda x: np.percentile(x, lw), raw=True)
    # 画上下轨
    ub.plot(ls='--', color='r', ax=ax2, label='ub')
    lb.plot(ls='--', color='green', ax=ax2, label='lb')
    ax2.legend()

    plt.subplots_adjust(hspace=0)
    return fig


def plot_quantreg_res(model: pd.DataFrame,
                      title: str = '',
                      ax: mpl.axes = None) -> mpl.axes:
    """画百分数回归图

    Args:
        model (pd.DataFrame): 百分位数回归模型结果
        title (str, optional): 标题. Defaults to ''.
        ax (_type_, optional): Defaults to None.

    Returns:
        _type_: _description_
    """
    if ax is None:

        ax = plt.gca()

    ax.set_title(title)
    ax.plot(model['q'], model['vix'], color='black')
    # ax.plot(model['q'], model['lb'], color='green', ls='--')
    # ax.plot(model['q'], model['ub'], color='red', ls='--')
    ax.fill_between(model['q'], model['ub'], model['lb'], alpha=0.2)
    return ax


def plot_hist2d(endog: pd.Series,
                exog: pd.Series,
                title: str = '',
                ax: mpl.axes = None) -> mpl.axes:
    """_summary_

    Args:
        endog (pd.Series): _description_
        exog (pd.Series): _description_
        title (str, optional): _description_. Defaults to ''.
        ax (mpl.axes, optional): _description_. Defaults to None.

    Returns:
        mpl.axes: _description_
    """
    if ax is None:
        ax = plt.gca()

    ax.set_title(title)
    ax.hist2d(endog, exog, bins=10, cmap='Blues')
    ax.set_ylabel('收益率')
    ax.set_xlabel('vix')
    ax.axhline(0, ls='--', color='black')
    #plt.colorbar(ax=ax)
    return ax


def plot_group_ret(endog: pd.Series,
                   exog: pd.Series,
                   title: str = '',
                   ax: mpl.axes = None) -> mpl.axes:
    """_summary_

    Args:
        endog (pd.Series): _description_
        exog (pd.Series): _description_
        title (str, optional): _description_. Defaults to ''.
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        mpl.axes: _description_
    """
    if ax is None:

        ax = plt.gca()

    # 采用百分位分组 分为10组
    group_ser: pd.Series = pd.qcut(endog, 10, False) + 1
    df: pd.DataFrame = group_ser.to_frame('group')
    df['next'] = exog
    df.index.names = ['date']

    group_avg_ret: pd.Series = pd.pivot_table(df.reset_index(),
                                              index='date',
                                              columns='group',
                                              values='next').mean()

    ax.set_title(title)
    xmajor_formatter = mpl.ticker.FuncFormatter(lambda x, pos: '%.2f%%' %
                                                (x * 100))
    ax.yaxis.set_major_formatter(xmajor_formatter)
    group_avg_ret.plot.bar(ax=ax, color='#1f77b4')
    ax.axhline(0, color='black')

    return ax

'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-07 10:09:17
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-28 11:28:03
Description: 画图相关函数
'''
from typing import Tuple, Union

import empyrical as ep
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

# 设置字体 用来正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False


def plot_indicator(price: pd.Series,
                   indincator: Union[pd.Series, pd.DataFrame],
                   title: str = '') -> mpl.axes:
    """标的与信号的关系图

    Args:
        price (pd.Series): 标的价格走势
        indincator (Union[pd.Series,pd.DataFrame]): 信号
        title (str, optional): 标题. Defaults to ''.

    Returns:
        mpl.axes: _description_
    """
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(3, 1)

    ax1 = plt.subplot(gs[:2, :])
    ax2 = plt.subplot(gs[2:, :])

    price.plot(ax=ax1, color='darkgray', title=title)
    ax1.legend()
    indincator.plot(ax=ax2, color=['red', 'green'], label='signal')

    ax2.legend()

    plt.subplots_adjust(hspace=0)
    return gs


def plot_qunatile_signal(price: pd.Series,
                         signal: pd.Series,
                         window: int,
                         bound: Tuple,
                         title: str = '') -> mpl.axes:
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
    return gs


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
    # plt.colorbar(ax=ax)
    return ax


def plot_quantile_group_ret(endog: pd.Series,
                            exog: pd.Series,
                            title: str = '',
                            group: int = 10,
                            ax: mpl.axes = None) -> mpl.axes:
    """将信号按10档分组画与未来收益的关系图

    Args:
        endog (pd.Series): 未来收益
        exog (pd.Series): 信号
        title (str, optional): 标题. Defaults to ''.
        ax (_type_, optional): ax. Defaults to None.

    Returns:
        mpl.axes: 图
    """
    if ax is None:

        ax = plt.gca()

    # 根据特征分值分组
    # 采用百分位分组 分为10组
    group = pd.qcut(exog, group, labels=False) + 1
    group_ret = endog.groupby(group).mean()

    # 画图
    ax.set_title(title)
    ax = group_ret.plot(kind='bar', figsize=(18, 6), color='1f77b4')
    # 画5组累计收益
    group_ret.rolling(5).sum().plot(kind='line', color='red', secondary_y=True)

    ax.yaxis.set_major_formatter('{x:.2%}')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.axhline(0, color='black')

    return ax


def plot_distribution(signal: pd.Series,
                      index_close: pd.Series,
                      forward: int,
                      title: str = '') -> mpl.axes:
    """画信号对应的N日涨幅与信号分布图

    Args:
        signal (pd.Series): _description_
        index_close (pd.Series): _description_
        forward (int): _description_
        axes (mpl.axes, optional): _description_. Defaults to None.

    Returns:
        mpl.axes: _description_
    """

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    gs = GridSpec(1, 5)

    # 计算未来N日收益
    forward_df: pd.Series = index_close.pct_change(forward).shift(
        -forward).iloc[:-forward]
    # 合并
    distribution_df: pd.DataFrame = pd.concat((signal, forward_df), axis=1)
    distribution_df.columns = ['signal', 'next_ret']
    # 信号分组
    distribution_df['group'] = pd.qcut(
        distribution_df['signal'], 50, labels=False) + 1
    group_returns: pd.Series = distribution_df.groupby(
        'group')['next_ret'].mean()
    # 信号分组收益
    roll_cum: pd.Series = group_returns.rolling(5).sum()

    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3:])

    ## 分组收益滚动累加
    ax1.set_title('信号期望收益分布')
    group_returns.plot(kind='bar', ax=ax1)
    roll_cum.plot(kind='line', color='red', secondary_y=True, ax=ax1)

    ax1.yaxis.set_major_formatter('{x:.2%}')
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_major_formatter('{x:.0f}')

    ax2.set_title('信号分布')
    sns.histplot(distribution_df['signal'], ax=ax2)
    plt.subplots_adjust(wspace=0.6)
    plt.suptitle(title)

    return gs


def plot_trade_flag(price: pd.DataFrame, buy_flag: pd.Series,
                    sell_flag: pd.Series):
    """买卖点标记

    Args:
        price (pd.DataFrame): _description_
        buy_flag (pd.Series): _description_
        sell_flag (pd.Series): _description_

    Returns:
        _type_: _description_
    """
    buy_flag = buy_flag.reindex(price.index)
    sell_flag = sell_flag.reindex(price.index)

    # 设置蜡烛图风格
    mc = mpf.make_marketcolors(up='r', down='g', wick='i', edge='i', ohlc='i')

    s = mpf.make_mpf_style(marketcolors=mc)

    buy_apd = mpf.make_addplot(buy_flag,
                               type='scatter',
                               markersize=100,
                               marker='^',
                               color='r')
    sell_apd = mpf.make_addplot(sell_flag,
                                type='scatter',
                                markersize=100,
                                marker='v',
                                color='g')
    ax = mpf.plot(price,
                  type='candle',
                  style=s,
                  datetime_format='%Y-%m-%d',
                  volume=True,
                  figsize=(18, 6),
                  addplot=[buy_apd, sell_apd],
                  warn_too_much_data=2000)

    return ax


def plot_algorithm_nav(result, price: pd.Series, title: str = '') -> mpl.axes:
    """画净值表现

    Args:
        result (_type_): 回测结果
        price (pd.Series): 基准价格
        title (str, optional): 标题. Defaults to ''.

    Returns:
        mpl.axes: 图
    """
    # 净值表现
    rets = get_strat_ret(result)

    align_rest, align_price = rets.align(price, join='left')

    cum: pd.Series = ep.cum_returns(align_rest, 1)
    ax = cum.plot(figsize=(18, 6), color='r', label='策略净值')

    (align_price / align_price[0]).loc[rets.index].plot(ls='--',
                                                        color='darkgray',
                                                        label='基准',
                                                        ax=ax)
    plt.legend()

    return ax


def get_strat_ret(result) -> pd.Series:

    return pd.Series(result[0].analyzers._TimeReturn.get_analysis())

'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-07 10:09:17
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-29 18:03:38
Description: 画图相关函数
'''
from typing import Tuple, Union

import empyrical as ep
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

from .timeseries import get_drawdown_table

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


"""plotly画图"""


def plot_drawdowns(cum_ser: pd.Series):
    """标记最大回撤

    Parameters
    ----------
    cum_ser : pd.Series
        index-date value-累计收益率

    Returns
    -------
    _type_
        _description_
    """

    fig = go.Figure()
    idx = cum_ser.index
    fig.add_trace(
        go.Scatter(x=idx,
                   y=cum_ser.tolist(),
                   mode='lines',
                   name='Algorithm_cum',
                   line=dict(color='#9467bd')))

    dtype_mapping = {
        '回撤开始日': np.datetime64,
        '回撤最低点日': np.datetime64,
        '回撤恢复日': np.datetime64
    }
    drawdown_table = drawdown_table.pipe(pd.DataFrame.astype, dtype_mapping)

    # 获取点位
    drawdown_table = get_drawdown_table(cum_ser, 5)
    # 判断近期是否处于回撤状态
    cond = drawdown_table['回撤恢复日'].isna()

    # 获取恢复点
    recovery_dates = [
        d for d, c in zip(drawdown_table['回撤恢复日'], cond) if not c
    ]
    recovery_values = cum_ser.loc[recovery_dates].tolist()

    # 获取开始点
    peak_dates = [d for d in drawdown_table['回撤开始日']]
    peak_values = cum_ser.loc[peak_dates].tolist()

    # 获取低点
    valley_dates = [d for d in drawdown_table['回撤最低点日']]
    valley_values = cum_ser.loc[valley_dates].tolist()

    # 是否进行中
    if len(recovery_dates) < len(drawdown_table):
        active_dates = idx[-1]
        active_values = cum_ser[active_dates]

    is_active = False  # 是否正在处于恢复期
    # 画区间
    for num, rows in drawdown_table.iterrows():

        peak_date = rows['回撤开始日']
        valley_date = rows['回撤最低点日']
        recovery_date = rows['回撤恢复日']

        if not isinstance(recovery_date, pd.Timestamp):
            is_active = True
            # 低点至还在回撤
            fig.add_vrect(x0=peak_date,
                          x1=idx[-1],
                          fillcolor='#eadec5',
                          opacity=0.5,
                          layer='below',
                          line_width=0)

        else:
            # 低点至回复
            fig.add_vrect(x0=valley_date,
                          x1=recovery_date,
                          fillcolor='#b7d7c5',
                          opacity=0.5,
                          layer='below',
                          line_width=0)

            # 开始回撤至低点
            fig.add_vrect(
                x0=peak_date,
                x1=valley_date,
                fillcolor="#eabdc5",
                opacity=0.5,
                layer="below",
                line_width=0,
            )

    # 开始点
    fig.add_trace(
        go.Scatter(mode="markers",
                   x=peak_dates,
                   y=peak_values,
                   marker_symbol='diamond',
                   marker_line_color="#c92d1f",
                   marker_color="#c92d1f",
                   marker_line_width=2,
                   marker_size=5,
                   name='Peak'))
    # 低点
    fig.add_trace(
        go.Scatter(mode="markers",
                   x=valley_dates,
                   y=valley_values,
                   marker_symbol='diamond',
                   marker_line_color="#387ced",
                   marker_color="#387ced",
                   marker_line_width=2,
                   marker_size=5,
                   name='Valley'))
    # 恢复点
    fig.add_trace(
        go.Scatter(mode="markers",
                   x=recovery_dates,
                   y=recovery_values,
                   marker_symbol='diamond',
                   marker_line_color="#37b13f",
                   marker_color="#37b13f",
                   marker_line_width=2,
                   marker_size=5,
                   name='Recovery'))

    if is_active:
        fig.add_trace(
            go.Scatter(mode="markers",
                       x=[active_dates],
                       y=[active_values],
                       marker_symbol='diamond',
                       marker_line_color="#ffaa00",
                       marker_color="#ffaa00",
                       marker_line_width=2,
                       marker_size=5,
                       name='Active'))

    fig.update_layout(yaxis_tickformat='.2%',
                      yaxis_title="Underwater",
                      title={
                          'text': 'Drawdown',
                          'x': 0.5,
                          'y': 0.9
                      },
                      hovermode="x unified",
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    return fig


def plot_trade_pnl(trade_stats: pd.DataFrame):
    """画盈亏图

    Args:
        trade_stats (pd.DataFrame): pnl%为小数

    Returns:
        _type_: _description_
    """
    cond = trade_stats['pnl%'] > 0
    fig = go.Figure()

    a_df = trade_stats.loc[cond, ['dateout', 'ref', 'pnl', 'pnl%']]
    b_df = trade_stats.loc[~cond, ['dateout', 'ref', 'pnl', 'pnl%']]
    fig.add_trace(
        go.Scatter(
            x=a_df['dateout'],
            y=a_df['pnl%'],
            mode='markers',
            name='Close - Profit',
            customdata=a_df[['ref', 'pnl', 'pnl%']],
            hovertemplate=
            'Position Id: %{customdata[0]}<br>Exit Timestamp: %{x}<br>PnL: %{customdata[1]:.6f}<br>Return: %{customdata[2]:.2%}',
            marker=dict(size=a_df['pnl%'].abs(),
                        sizemode='area',
                        color='rgb(181,31,18)',
                        sizeref=2. * a_df['pnl%'].abs().max() / (23.**2),
                        line={
                            'color': 'rgb(181,31,18)',
                            'width': 1
                        },
                        symbol='circle',
                        sizemin=4)))

    fig.add_trace(
        go.Scatter(
            x=b_df['dateout'],
            y=b_df['pnl%'],
            mode='markers',
            name='Close - Loss',
            customdata=b_df[['ref', 'pnl', 'pnl%']],
            hovertemplate=
            'Position Id: %{customdata[0]}<br>Exit Timestamp: %{x}<br>PnL: %{customdata[1]:.6f}<br>Return: %{customdata[2]:.2%}',
            marker=dict(
                size=b_df['pnl%'].abs(),
                sizemode='area',
                color='rgb(38,123,44)',
                sizeref=2. * b_df['pnl%'].abs().max() / (23.**2),
                line={
                    'color': 'rgb(38,123,44)',
                    'width': 1
                },
                symbol='circle',
                sizemin=4,
            )))

    fig.add_hline(y=0, line_dash='dash')
    fig.update_layout(legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5,
                                  traceorder='normal'),
                      yaxis_title="Trade PnL",
                      yaxis_tickformat='.2%',
                      title={
                          'text': 'Trade PnL',
                          'x': 0.5,
                          'y': 0.9
                      })

    return fig


def plot_underwater(cum_ser: pd.Series):
    """画动态回撤

    Parameters
    ----------
    cum_ser : pd.Series
        _description_

    Returns
    -------
    _type_
        _description_
    """
    maxdrawdown = cum_ser / cum_ser.cummax() - 1

    idx = maxdrawdown.index

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=idx,
                   y=maxdrawdown.tolist(),
                   fill='tozeroy',
                   line=dict(color='#e2b6b1'),
                   showlegend=False))  # fill down to xaxis
    fig.add_trace(
        go.Scatter(x=idx,
                   y=maxdrawdown.tolist(),
                   mode='lines',
                   name='Drawdown',
                   line=dict(color='#dc3912')))

    fig.add_hline(y=0, line_dash='dash')
    fig.update_layout(yaxis_tickformat='.2%',
                      title={
                          'text': 'Underwater',
                          'x': 0.5,
                          'y': 0.9
                      },
                      hovermode="x unified",
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    return fig


def plot_cumulative_returns(returns: pd.Series, benchmark: pd.Series):
    """画累计收益率

    Parameters
    ----------
    returns : pd.Series
        index-date values-策略累计收益率
    benchmark : pd.Series
        index-date values-基准累计收益率
    title : str, optional
        _description_, by default ''

    Returns
    -------
    _type_
        _description_
    """

    idx = returns.index

    fig = go.Figure()
    # area
    fig.add_trace(
        go.Scatter(x=idx,
                   y=returns.tolist(),
                   fill='tozeroy',
                   line=dict(color='#a0ccac'),
                   showlegend=False))  # fill down to xaxis

    fig.add_trace(
        go.Scatter(x=idx,
                   y=returns.tolist(),
                   mode='lines',
                   name='Algorithm_cum',
                   line=dict(color='#9467bd')))
    # 基准这些
    fig.add_trace(
        go.Scatter(x=idx,
                   y=benchmark.tolist(),
                   mode='lines',
                   name='Benchmark',
                   line=dict(color='#7f7f7f')))

    fig.add_hline(y=0, line_dash='dash')
    fig.update_layout(hovermode="x unified",
                      title={
                          'text': 'Cumuilative Returns',
                          "x": 0.5,
                          "y": 0.9
                      },
                      yaxis_tickformat='.2%',
                      yaxis_title="Cumuilative Returns",
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))
    return fig


def plot_orders(trade_df: pd.DataFrame, price: pd.Series):
    """交易点标记

    Args:
        trade_df (pd.DataFrame): _description_
        price (pd.Series): index-date values-price
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=price.index,
                   y=price.tolist(),
                   mode='lines',
                   name='Close',
                   line=dict(color='#1f77b4')))

    fig.add_trace(
        go.Scatter(mode="markers",
                   x=trade_df['datein'],
                   y=price.loc[pd.to_datetime(trade_df['datein'])],
                   marker_symbol='triangle-up',
                   marker_line_color="#c92d1f",
                   marker_color="#c92d1f",
                   marker_line_width=2,
                   marker_size=10,
                   name='Buy'))

    fig.add_trace(
        go.Scatter(mode="markers",
                   x=trade_df['dateout'],
                   y=price.loc[pd.to_datetime(trade_df['dateout'])],
                   marker_symbol='triangle-down',
                   marker_line_color="#3db345",
                   marker_color="#3db345",
                   marker_line_width=2,
                   marker_size=10,
                   name='Sell'))

    fig.update_layout(title={
        'text': 'Orders',
        'x': 0.5,
        'y': 0.9
    },
                      yaxis_title="Price",
                      hovermode="x unified",
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    return fig

def plot_annual_returns(returns: pd.Series):
    """画每年累计收益

    Args:
        returns (pd.Series): index-date values-returns

    Returns:
        _type_: _description_
    """
    ann_ret_df = ep.aggregate_returns(returns, 'yearly')

    colors = ['#4e57c6' if v > 0 else 'crimson' for v in ann_ret_df]

    fig = go.Figure(
        go.Bar(x=ann_ret_df.values,
               y=ann_ret_df.index,
               orientation='h',
               marker_color=colors))

    fig.update_layout(title={
        'text': 'Annual returns',
        'x': 0.5,
        'y': 0.9
    },
                      yaxis_title="Year",
                      xaxis_tickformat='.2%',
                      hovermode="x unified",
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    fig.add_vline(x=ann_ret_df.mean(), line_dash='dash')

    return fig
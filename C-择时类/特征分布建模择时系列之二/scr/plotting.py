'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-07 10:09:17
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-10-20 15:04:42
Description: 画图相关函数
'''
from typing import Dict, List, Tuple

import empyrical as ep
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from alphalens.tears import GridFigure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from plotly.graph_objs import Figure

from .timeseries import get_drawdown_table
from .utils import trans2strftime

# 设置字体 用来正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False


def plot_indicator(data: pd.DataFrame, title: str = '') -> mpl.axes:
    """画信号与k线的关系图标

    Returns
    -------
    mpl.axes
    """
    mc = mpf.make_marketcolors(up='r', down='g')
    s = mpf.make_mpf_style(marketcolors=mc)
    size = len(data)

    config: Dict = {
        'style': s,
        'type': 'candle',
        'volume': True,
        'figsize': (18, 6),
        'datetime_format': '%Y-%m-%d',
        'warn_too_much_data': size
    }

    addplot = [
        mpf.make_addplot(data['threshold_to_long_a'],
                         color='r',
                         panel=2,
                         linestyle='dashdot'),
        mpf.make_addplot(data['volume_index'], color='#3785bc', panel=2),
        mpf.make_addplot(data['threshold_to_short'], color='g', panel=2),
        mpf.make_addplot(data['threshold_to_long_b'],
                         color='r',
                         panel=2,
                         linestyle='dashdot'),
    ]

    col: List = 'open,high,low,close,volume'.split(',')
    return mpf.plot(data[col], **config, addplot=addplot, title=title)


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


# def plot_distribution(signal: pd.Series,
#                       index_close: pd.Series,
#                       forward: int,
#                       title: str = '') -> mpl.axes:
#     """画信号对应的N日涨幅与信号分布图

#     Args:
#         signal (pd.Series): _description_
#         index_close (pd.Series): _description_
#         forward (int): _description_
#         axes (mpl.axes, optional): _description_. Defaults to None.

#     Returns:
#         mpl.axes: _description_
#     """

#     fig, axes = plt.subplots(1, 2, figsize=(20, 6))
#     gs = GridSpec(1, 5)

#     # 计算未来N日收益
#     forward_df: pd.Series = index_close.pct_change(forward).shift(
#         -forward).iloc[:-forward]
#     # 合并
#     distribution_df: pd.DataFrame = pd.concat((signal, forward_df), axis=1)
#     distribution_df.columns = ['signal', 'next_ret']
#     # 信号分组
#     distribution_df['group'] = pd.qcut(
#         distribution_df['signal'], 50, labels=False) + 1
#     group_returns: pd.Series = distribution_df.groupby(
#         'group')['next_ret'].mean()
#     # 信号分组收益
#     roll_cum: pd.Series = group_returns.rolling(5).sum()

#     ax1 = fig.add_subplot(gs[0, :3])
#     ax2 = fig.add_subplot(gs[0, 3:])

#     # 计算统计指标
#     avg, std, kur, skew = distribution_df['signal'].mean(
#     ), distribution_df['signal'].std(), distribution_df['signal'].kurt(
#     ), distribution_df['signal'].skew()
#     # 显示统计指标
#     ax2.text(.05,
#              .95,
#              " Mean %.3f \n Std. %.3f \n kurtosis %.3f \n Skew %.3f" %
#              (avg, std, kur, skew),
#              fontsize=16,
#              bbox={
#                  'facecolor': 'white',
#                  'alpha': 1,
#                  'pad': 5
#              },
#              transform=ax2.transAxes,
#              verticalalignment='top')

#     # 分组收益滚动累加
#     ax1.set_title('信号期望收益分布')
#     group_returns.plot(kind='bar', ax=ax1)
#     roll_cum.plot(kind='line', color='red', secondary_y=True, ax=ax1)

#     ax1.yaxis.set_major_formatter('{x:.2%}')
#     ax1.xaxis.set_major_locator(MultipleLocator(5))
#     ax1.xaxis.set_major_formatter('{x:.0f}')

#     ax2.set_title('信号分布')
#     sns.histplot(distribution_df['signal'], ax=ax2)
#     plt.subplots_adjust(wspace=0.6)
#     plt.suptitle(title)

#     return gs


def get_distribution_data(signal_ser: pd.Series,
                          forward_returns: pd.Series,
                          q: int = 50,
                          window: int = 5,
                          group: bool = False) -> pd.DataFrame:
    """构建用于画信号于累计收益分布的数据

    Parameters
    ----------
    signal_ser : pd.Series
        信号数据
    forward_returns : pd.Series
        未来收益率数据
    window : int, optional
        累计收益的窗口, by default 5
    group : bool, optional
        False最终显示信号分组 True最终显示1-q的聚合分组, by default False

    Returns
    -------
    pd.DataFrame
        index columns-forward_ret_avg|roll_ret
    """
    if not (isinstance(signal_ser, pd.Series)
            and isinstance(forward_returns, pd.Series)):

        raise ValueError('signal_ser和forward_returns数据类型必须为pd.Series')

    # 数据处理部分
    combine_df = pd.concat((signal_ser, forward_returns), axis=1)
    combine_df.columns = ['signal', 'forward_ret']

    labels = False if group else None
    combine_df['group'] = pd.qcut(combine_df['signal'], q=q, labels=labels)

    if group:
        combine_df['group'] += 1

    aggregation_frame: pd.Series = combine_df.groupby(
        'group')['forward_ret'].mean()
    aggregation_frame: pd.DataFrame = aggregation_frame.to_frame(
        'forward_ret_avg')
    aggregation_frame['roll_ret'] = aggregation_frame[
        'forward_ret_avg'].rolling(window).sum()

    return aggregation_frame


def get_ticks_from_index(ser_index: pd.Series) -> np.ndarray:

    left: np.ndarray = ser_index.categories.left.values
    right: np.ndarray = ser_index.categories.right.values

    return np.sort(np.unique(np.append(left, right)))


def plot_hist_signal_with_cum(aggregation_frame: pd.DataFrame,
                              is_categories_index: bool = True,
                              title: str = '',
                              ax: mpl.axes = None) -> mpl.axes:
    """画信号分布于累计收益的分布图

    Parameters
    ----------
    aggregation_frame : pd.DataFrame
        index-Any column-forward_ret_avg|roll_ret
    is_categories_index : bool, optional
        当为True时aggregation_frame的索引为categories, by default True
    ax : mpl.axes, optional, by default None

    Returns
    -------
    mpl.axes
    """
    if ax is None:

        fig, ax = plt.subplots(figsize=(18, 6))

    if is_categories_index:

        idx: np.ndarray = np.arange(len(aggregation_frame) + 1)
        slice_idx: np.ndarray = idx[:-1]
        ticks_arr = get_ticks_from_index(aggregation_frame.index)

    else:

        idx: np.ndarray = np.arange(len(aggregation_frame))
        slice_idx: np.ndarray = idx
        ticks_arr = aggregation_frame.index

    ax.bar(slice_idx, aggregation_frame['forward_ret_avg'], align='edge')
    ax.plot(slice_idx, aggregation_frame['roll_ret'], color='r')
    ax.axhline(0, color='darkgray')
    ax.set_xticks(idx, labels=ticks_arr, rotation=90)
    ax.yaxis.set_major_formatter('{x:.2%}')
    ax.set_title(title)
    ax.set_ylabel('收益率(%)')
    ax.set_xlabel('信号分布')

    return ax


def plot_distribution(signal_ser: pd.Series,
                      forward_ret_ser: pd.Series,
                      forward_window: int = 5,
                      q: int = 50,
                      group: bool = True,
                      title: str = '',
                      ax: mpl.axes = None) -> mpl.axes:
    """画信号对应的N日涨幅与信号分布图

    Args:
        signal_ser (pd.Series): 信号数据
        forward_ret_ser (pd.Series): 未来N日收益率
        forward_window (int): 未来收益率累计窗口
        q (int): 分组q值. Defaults to None.
        group (bool): 显示为聚合分组还是因子原始值.Defaults to True.
        title (str): 标题

    Returns:
        mpl.axes
    """
    fig = plt.gcf() if ax is not None else plt.figure(figsize=(22, 6))
    gs = GridSpec(1, 5)
    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3:])
    aggregation_frame = get_distribution_data(signal_ser,
                                              forward_ret_ser,
                                              q=q,
                                              window=forward_window,
                                              group=group)

    is_categories_index: bool = not group
    ax1 = plot_hist_signal_with_cum(aggregation_frame, is_categories_index,
                                    '信号期望与累计收益分布', ax1)

    avg, std, kur, skew = signal_ser.mean(), signal_ser.std(), signal_ser.kurt(
    ), signal_ser.skew()

    ax2.text(0.65,
             0.95, (" Mean %.3f \n Std. %.3f \n kurtosis %.3f \n Skew %.3f" %
                    (avg, std, kur, skew)),
             fontsize=16,
             bbox={
                 'facecolor': 'white',
                 'alpha': 1,
                 'pad': 5
             },
             transform=ax2.transAxes,
             verticalalignment='top')

    ax2.set_title('信号分布')
    sns.histplot(signal_ser.dropna(), ax=ax2)
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

    return mpf.plot(price,
                    type='candle',
                    style=s,
                    datetime_format='%Y-%m-%d',
                    volume=True,
                    figsize=(18, 6),
                    addplot=[buy_apd, sell_apd],
                    warn_too_much_data=2000)


def plot_algorithm_nav(result: List,
                       price: pd.Series,
                       title: str = '') -> mpl.axes:
    """画净值表现

    Args:
        result (_type_): 回测结果
        price (pd.Series): 基准价格
        title (str, optional): 标题. Defaults to ''.

    Returns:
        mpl.axes: 图
    """
    # 净值表现
    rets: pd.Series = pd.Series(result[0].analyzers._TimeReturn.get_analysis())

    align_rest, align_price = rets.align(price, join='left')

    cum: pd.Series = ep.cum_returns(align_rest, 1)
    ax = cum.plot(figsize=(18, 6), color='r', label='策略净值')

    (align_price / align_price[0]).loc[rets.index].plot(ls='--',
                                                        color='darkgray',
                                                        label='基准',
                                                        ax=ax)
    plt.legend()

    return ax


"""plotly画图"""


def plot_drawdowns(returns: pd.Series) -> Figure:
    """标记最大回撤

    Parameters
    ----------
    cum_ser : pd.Series
        index-date value-收益率

    Returns
    -------
    _type_
        _description_
    """

    fig = go.Figure()
    cum_ser: pd.Series = ep.cum_returns(returns)
    idx: pd.DatetimeIndex = cum_ser.index
    fig.add_trace(
        go.Scatter(x=idx,
                   y=cum_ser.tolist(),
                   mode='lines',
                   name='Algorithm_cum',
                   line=dict(color='#9467bd')))

    dtype_mapping: Dict = {
        '回撤开始日': np.datetime64,
        '回撤最低点日': np.datetime64,
        '回撤恢复日': np.datetime64
    }

    # 获取点位
    drawdown_table: pd.DataFrame = get_drawdown_table(
        returns, 5).dropna(subset=['区间最大回撤 %'])

    drawdown_table: pd.DataFrame = drawdown_table.pipe(pd.DataFrame.astype,
                                                       dtype_mapping)
    # 判断近期是否处于回撤状态
    cond: pd.Series = drawdown_table['回撤恢复日'].isna()

    # 获取恢复点
    recovery_dates: List = [
        d for d, c in zip(drawdown_table['回撤恢复日'], cond) if not c
    ]
    recovery_values: List = cum_ser.loc[recovery_dates].tolist()

    # 获取开始点
    peak_dates: List = [d for d in drawdown_table['回撤开始日'] if not pd.isnull(d)]
    peak_values: List = cum_ser.loc[peak_dates].tolist()

    # 获取低点
    valley_dates: List = [
        d for d in drawdown_table['回撤最低点日'] if not pd.isnull(d)
    ]
    valley_values: List = cum_ser.loc[valley_dates].tolist()

    # 是否进行中
    if len(recovery_dates) < len(drawdown_table):
        active_dates: pd.Timestamp = idx[-1]
        active_values: float = cum_ser[active_dates]

    is_active: bool = False  # 是否正在处于恢复期
    # 画区间
    for num, rows in drawdown_table.iterrows():

        peak_date: pd.Timestamp = rows['回撤开始日']
        valley_date: pd.Timestamp = rows['回撤最低点日']
        recovery_date: pd.Timestamp = rows['回撤恢复日']

        if not isinstance(recovery_date, pd.Timestamp):
            is_active: bool = True
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
                          'y': 0.93
                      },
                      hovermode="x unified",
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    return fig


def plot_trade_pnl(trade_stats: pd.DataFrame) -> Figure:
    """画盈亏图

    Args:
        trade_stats (pd.DataFrame): pnl%为小数

    Returns:
        _type_: _description_
    """
    cond: pd.Series = trade_stats['pnl%'] > 0
    fig = go.Figure()

    a_df: pd.DataFrame = trade_stats.loc[cond,
                                         ['dateout', 'ref', 'pnl', 'pnl%']]
    b_df: pd.DataFrame = trade_stats.loc[~cond,
                                         ['dateout', 'ref', 'pnl', 'pnl%']]

    fig.add_trace(
        go.Scatter(
            x=trans2strftime(a_df['dateout']),
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
            x=trans2strftime(b_df['dateout']),
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


def plot_underwater(returns: pd.Series) -> Figure:
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
    cum_ser: pd.Series = ep.cum_returns(returns, 1)
    maxdrawdown: pd.Series = cum_ser / cum_ser.cummax() - 1

    idx: pd.DatetimeIndex = maxdrawdown.index

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


def plot_cumulative_returns(returns: pd.Series,
                            benchmark: pd.Series) -> Figure:
    """画累计收益率

    Parameters
    ----------
    returns : pd.Series
        index-date values-策略收益率
    benchmark : pd.Series
        index-date values-基准收益率
    title : str, optional
        _description_, by default ''

    Returns
    -------
    Figure
        _description_
    """

    returns = ep.cum_returns(returns)
    benchmark = ep.cum_returns(benchmark)

    idx: pd.DatetimeIndex = returns.index

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


def plot_orders_on_price(price: pd.Series, trade_df: pd.DataFrame) -> Figure:
    """交易点标记

    Args:
        trade_df (pd.DataFrame): _description_
        price (pd.Series): index-date values-price
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=price.index.strftime('%Y-%m-%d'),
                   y=price.tolist(),
                   mode='lines',
                   name='Close',
                   line=dict(color='#1f77b4')))

    fig.add_trace(
        go.Scatter(mode="markers",
                   x=trans2strftime(trade_df['datein']),
                   y=price.loc[pd.to_datetime(trade_df['datein'])],
                   marker_symbol='triangle-up',
                   marker_line_color="#c92d1f",
                   marker_color="#c92d1f",
                   marker_line_width=2,
                   marker_size=10,
                   name='Buy'))

    fig.add_trace(
        go.Scatter(mode="markers",
                   x=trans2strftime(trade_df['dateout']),
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


def plotl_order_on_ohlc(ohlc: pd.DataFrame,
                        trade_list: pd.DataFrame,
                        *,
                        showlegend: bool = False,
                        title: str = '',
                        rows: int = None,
                        cols: int = None) -> Figure:
    """画k线并标记

    Args:
        ohlc (pd.DataFrame): _description_
        res (namedtuple): _description_
        title (str, optional): _description_. Defaults to ''.
    """
    def get_holidays(ohlc: pd.DataFrame) -> List:
        """用于过滤非交易日"""
        idx = pd.to_datetime(ohlc.index)
        begin = idx.min()
        end = idx.max()

        days = pd.date_range(begin, end)

        return [i.strftime('%Y-%m-%d') for i in set(days).difference(idx)]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=ohlc.index,
                                 open=ohlc['open'],
                                 high=ohlc['high'],
                                 low=ohlc['low'],
                                 close=ohlc['close'],
                                 increasing_line_color='red',
                                 decreasing_line_color='green',
                                 showlegend=False),
                  row=rows,
                  col=cols)

    # 添加买卖点
    # Buy
    fig.add_trace(
        go.Scatter(
            x=trans2strftime(trade_list['datetin']),
            y=list(trade_list['pricein'].values * (1 - 0.05)),
            name='Buy',
            mode='markers',
            marker_size=15,
            marker_symbol='triangle-up',
            showlegend=False,
            marker_color='red',
            #line=dict(color='royalblue', width=7,dash='solid')
        ),
        row=rows,
        col=cols)

    # Sell
    fig.add_trace(
        go.Scatter(
            x=trans2strftime(trade_list['datetout']),
            y=list(trade_list['priceout'].values * (1 + 0.03)),
            name='Sell',
            mode='markers',
            marker_size=15,
            marker_symbol='triangle-down',
            marker_color='green',
            showlegend=False,
            #line=dict(color='royalblue', width=7,dash='solid')
        ),
        row=rows,
        col=cols)

    holidays = get_holidays(ohlc)
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            # dict(bounds=["sat", "mon"]), # 隐藏周六、周日
            dict(values=holidays)  # 隐藏特定假期
        ])
    fig.update_layout(hovermode='x unified')
    # showlegend=False)
    fig.update_layout(title=title,
                      xaxis_tickformat='%Y-%m-%d',
                      showlegend=True)

    return fig


def plot_annual_returns(returns: pd.Series) -> Figure:
    """画每年累计收益

    Args:
        returns (pd.Series): index-date values-returns

    Returns:
        _type_: _description_
    """
    ann_ret_df: pd.Series = ep.aggregate_returns(returns, 'yearly')

    colors: List = ['#4e57c6' if v > 0 else 'crimson' for v in ann_ret_df]

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
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    fig.add_vline(x=ann_ret_df.mean(), line_dash='dash')

    return fig


def plot_monthly_returns_heatmap(returns: pd.Series) -> Figure:
    """画每月收益热力图

    Parameters
    ----------
    returns : pd.Series
        index-date values-returns

    Returns
    -------
    _type_
        _description_
    """
    monthly_ret_table: pd.Series = ep.aggregate_returns(returns, 'monthly')
    monthly_ret_table: pd.Series = monthly_ret_table.unstack().round(3)

    fig = go.Figure(data=go.Heatmap(
        z=monthly_ret_table.values,
        x=monthly_ret_table.columns.map(str),
        y=monthly_ret_table.index.map(str),
        text=monthly_ret_table.values,
        texttemplate="%{text:.2%}",
    ))
    fig.update_layout(
        title={
            'text': 'Monthly returns (%)',
            "x": 0.5,
            "y": 0.9
        },
        yaxis_title="Year",
        xaxis_title='Month',
    )
    return fig


def plot_monthly_returns_dist(returns: pd.Series) -> Figure:
    """画分月收益直方图

    Parameters
    ----------
    returns : pd.Series
        index-date values-returns

    Returns
    -------
    _type_
        _description_
    """
    monthly_ret_table = pd.DataFrame(ep.aggregate_returns(returns, 'monthly'),
                                     columns=['Returns'])
    fig = px.histogram(monthly_ret_table, x='Returns')
    mean_returns = monthly_ret_table['Returns'].mean()
    fig.add_vline(x=mean_returns,
                  line_dash='dash',
                  annotation_text='Mean:{:.2f}'.format(mean_returns))
    fig.update_layout(
        hovermode="x unified",
        title={
            'text': 'Distribution of monthly returns',
            "x": 0.5,
            "y": 0.9
        },
        yaxis_title="Number of months",
        xaxis_tickformat='.2%',
        xaxis_title='Returns',
    )
    return fig


def plotly_table(df: pd.DataFrame, index_name: str = '') -> Figure:

    df.index.names = [index_name]
    df: pd.DataFrame = df.reset_index()

    headerColor = 'grey'
    # rowEvenColor = 'lightgrey'
    # rowOddColor = 'white'

    fig = go.Figure(data=[
        go.Table(
            header=dict(values=df.columns,
                        line_color='darkslategray',
                        fill_color=headerColor,
                        align=['left', 'center'],
                        font=dict(color='white', size=12)),
            cells=dict(
                values=df.T.values,
                line_color='darkslategray',
                # 2-D list of colors for alternating rows
                #fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
                align=['left', 'center'],
                font=dict(color='darkslategray', size=11)))
    ])

    return fig


def plot_params_table_visualization(par_frame: pd.DataFrame,
                                    rows: int,
                                    size: int = None):

    if (size is not None) and (rows is None):
        rows = size // 2 + 1 if size % 2 else size

    elif (size is None) and (rows is None):
        raise ValueError('size和rows不能同时为空!')

    elif size is not None:

        raise ValueError('size和rows不能同时存在!')

    # 可视化遍历参数的结果
    cols = 2

    g = GridFigure(rows, cols)
    g.fig = plt.figure(figsize=(14, rows * 3.4))

    for name, df in par_frame.groupby(level='窗口期'):

        ax1 = df.reset_index(level='窗口期', drop=True)["年化收益率(%)"].plot(
            marker="o", title=f"极端参数与年化收益率(window={name})", ax=g.next_cell())
        ax1.axhline(0, ls=':', color='darkgray')
        ax1.set_ylabel("年化收益率(%)")
        ax2 = df.reset_index(level='窗口期', drop=True)["夏普"].plot(
            marker="o", title=f"极端参数与夏普比率(window={name})", ax=g.next_cell())
        ax2.set_ylabel("夏普比率")

    plt.subplots_adjust(hspace=0.5)

    plt.show()
    g.close()
    return

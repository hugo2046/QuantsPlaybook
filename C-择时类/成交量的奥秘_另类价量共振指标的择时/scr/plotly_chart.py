from typing import Dict, List

import empyrical as ep
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from .timeseries import gen_drawdown_table
from .utils import trans2strftime

"""plotly画图"""


def _get_holidays(ohlc: pd.DataFrame) -> List:
    """用于过滤非交易日"""
    idx = pd.to_datetime(ohlc.index)
    begin = idx.min()
    end = idx.max()

    days = pd.date_range(begin, end)

    return [i.strftime('%Y-%m-%d') for i in set(days).difference(idx)]


def add_shape_to_ohlc(fig: Figure, target: pd.Series) -> Figure:
    """在OHLC上添加标记

    Args:
        fig (Figure): 需要添加标记的图形
        target (pd.Series): index-date values-0,1标记 1为需要标记的日期

    Returns:
        Figure: 标记后的图形
    """
    target.index = pd.to_datetime(target.index)
    target_idx: pd.Index = target[target == 1].index

    for x in target_idx:
        x = x.strftime('%Y-%m-%d')
        fig.add_shape(x0=x,
                      x1=x,
                      y0=0.,
                      y1=1,
                      xref='x',
                      yref='paper',
                      opacity=0.5,
                      line_width=5,
                      line_color='LightSalmon')

    return fig


def plot_candlestick(price: pd.DataFrame, vol: bool = False, title: str = ""):

    # Create subplots and mention plot grid size

    candlestick = go.Candlestick(x=price.index,
                                 open=price["open"],
                                 high=price["high"],
                                 low=price["low"],
                                 close=price["close"],
                                 increasing_line_color='red',
                                 decreasing_line_color='green',
                                 name='price')

    rangeselector_dict: Dict = dict(buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(step="all"),
    ])

    if vol:

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, ""),
            row_width=[0.2, 0.7],
        )

        # 绘制k数据
        fig.add_trace(candlestick, row=1, col=1)

        # 绘制成交量数据
        bar_colors: np.ndarray = np.where(price["close"] > price["open"],
                                          "#ff3232", "#399b3d")
        fig.add_trace(
            go.Bar(x=price.index,
                   y=price["volume"],
                   showlegend=False,
                   marker_color=bar_colors,
                   name='volume'),
            row=2,
            col=1,
        )

        fig.update_layout(
            yaxis_title="Price",
            yaxis2_title="Volume",
            xaxis1_tickformat="%Y-%m-%d",
            xaxis2_tickformat="%Y-%m-%d",
            showlegend=False,
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=True,
        )
        fig.update_xaxes(rangeslider_visible=True,
                         rangeselector=rangeselector_dict,
                         row=1,
                         col=1)
        # Do not show OHLC's rangeslider plot
        fig.update(layout_xaxis_rangeslider_visible=False)

    else:

        fig = go.Figure(candlestick)
        fig.update_layout(xaxis_tickformat="%Y-%m-%d",
                          showlegend=False,
                          hovermode="x unified",
                          title={
                              'text': title,
                              'x': 0.5,
                              'y': 0.95,
                              'xanchor': 'center',
                              'yanchor': 'top'
                          },
                          font={'size': 18})
        fig.update_xaxes(rangeslider_visible=True,
                         rangeselector=rangeselector_dict)

    # 去除休市的日期，保持连续
    dt_breaks = _get_holidays(price)
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)], showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# def plot_ohlc(ohlc: pd.DataFrame, title: str = '') -> Figure:
#     """画k线图

#     Args:
#         ohlc (pd.DataFrame): index-date OHLC
#         title (str, optional): 标题. Defaults to ''.

#     Returns:
#         Figure:
#     """
#     fig = go.Figure()

#     fig.add_trace(
#         go.Candlestick(x=ohlc.index,
#                        open=ohlc['open'],
#                        high=ohlc['high'],
#                        low=ohlc['low'],
#                        close=ohlc['close'],
#                        increasing_line_color='red',
#                        decreasing_line_color='green',
#                        showlegend=False))

#     holidays = _get_holidays(ohlc)
#     fig.update_xaxes(
#         rangeslider_visible=False,
#         rangebreaks=[
#             # dict(bounds=["sat", "mon"]), # 隐藏周六、周日
#             dict(values=holidays)  # 隐藏特定假期
#         ])
#     fig.update_layout(hovermode='x unified')
#     # showlegend=False)
#     fig.update_layout(title=title,
#                       font={'size': 18},
#                       xaxis_rangeslider_visible=True,
#                       xaxis_tickformat='%Y-%m-%d',
#                       showlegend=True)

#     return fig


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
    drawdown_table: pd.DataFrame = gen_drawdown_table(returns, 5)
    drawdown_table.dropna(subset=['区间最大回撤(%)'], inplace=True)
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
    peak_dates: List = list(drawdown_table['回撤开始日'])
    peak_values: List = cum_ser.loc[peak_dates].tolist()

    # 获取低点
    valley_dates: List = list(drawdown_table['回撤最低点日'])
    valley_values: List = cum_ser.loc[valley_dates].tolist()

    # 是否进行中
    if len(recovery_dates) < len(drawdown_table):
        active_dates: pd.Timestamp = idx[-1]
        active_values: float = cum_ser[active_dates]

    is_active: bool = False  # 是否正在处于恢复期
    # 画区间
    for num, rows in drawdown_table.iterrows():

        peak_date: pd.Timestamp = rows['回撤开始日']
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
            valley_date: pd.Timestamp = rows['回撤最低点日']
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
                      xaxis_tickformat='%Y-%m-%d',
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
                      xaxis_tickformat='%Y-%m-%d',
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
                      xaxis_tickformat='%Y-%m-%d',
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
            x=trans2strftime(trade_list['datein']),
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
            x=trans2strftime(trade_list['dateout']),
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

    holidays = _get_holidays(ohlc)
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            dict(values=holidays)  # 隐藏特定假期
        ])
    fig.update_layout(hovermode='x unified')

    fig.update_layout(title={
        'text': 'Orders',
        'x': 0.5,
        'y': 0.9
    },
                      yaxis_title="Price",
                      hovermode="x unified",
                      xaxis_tickformat='%Y-%m-%d',
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    return fig


def plot_annual_returns(returns: pd.Series) -> Figure:
    """画每年累计收益

    Args:
        returns (pd.Series): index-date values-returns

    Returns:
        _type_: _description_
    """
    ann_ret_df: pd.Series = ep.aggregate_returns(returns, 'yearly')

    colors: List = ['crimson' if v > 0 else '#7a9e9f' for v in ann_ret_df]

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
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

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
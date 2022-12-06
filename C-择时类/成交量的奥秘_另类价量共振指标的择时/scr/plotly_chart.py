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


def GridPlotly(df1: pd.Series, df2: pd.DataFrame, cols: int = 4):

    size = df1.shape[1]
    rows = size // cols + 1 if size % 2 else size // cols

    fig = make_subplots(rows=rows,
                        cols=cols,
                        subplot_titles=df1.columns.tolist())

    row = 1
    col = 1

    def _plotly_add_nav(ser, benchmark, fig, row, col):

        fig.append_trace(go.Scatter(x=ser.index,
                                    y=ser.values,
                                    line=dict(color='red'),
                                    name=ser.name),
                         row=row,
                         col=col)
        fig.append_trace(go.Scatter(x=benchmark.index,
                                    y=benchmark.values,
                                    line=dict(color='darkgray'),
                                    name='benchmark'),
                         row=row,
                         col=col)
        # fig.update_layout(showlegend=False,hovermode="x unified",yaxis_title='累计收益率',xaxis_title='日期',yaxis_tickformat='.2%')

        return fig

    for current_col, (name, ser) in enumerate(df1.items()):

        if current_col > (size - 1):
            break
        if current_col % cols == 0 and current_col != 0:
            row += 1
            if col >= cols:
                col = 1
        fig = _plotly_add_nav(ser, df2[name], fig, row, col)
        col += 1

    height = rows * 450 if cols == 1 else cols * 450
    fig.update_layout(height=height, showlegend=False, hovermode="x unified")
    return fig

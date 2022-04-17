'''
Author: Hugo
Date: 2022-04-16 21:46:22
LastEditTime: 2022-04-16 22:12:36
LastEditors: Please set LastEditors
Description: 画图
'''
import pandas as pd
from collections import namedtuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import (List, Tuple, Union, Dict)


def plot_ohlc(ohlc: pd.DataFrame, res: namedtuple,*,showlegend:bool=False, title: str = '',fig=None,rows:int=None,cols:int=None):
    """画k线并标记

    Args:
        ohlc (pd.DataFrame): _description_
        res (namedtuple): _description_
        title (str, optional): _description_. Defaults to ''.
    """
    def get_holidays(ohlc: pd.DataFrame) -> List:

        idx = pd.to_datetime(ohlc.index)
        begin = idx.min()
        end = idx.max()

        days = pd.date_range(begin, end)

        return [i.strftime('%Y-%m-%d') for i in set(days).difference(idx)]

    def get_marker(ohlc: pd.DataFrame, res: namedtuple) -> pd.Series:

        idx = list(res.points.values())[0][0]

        return ohlc.loc[idx, 'close']
    
    if fig is None:
        fig = go.Figure()

    fig.add_trace(go.Candlestick(x=ohlc.index,
                       open=ohlc['open'],
                       high=ohlc['high'],
                       low=ohlc['low'],
                       close=ohlc['close'],
                       increasing_line_color='red',
                       decreasing_line_color='green',
                       showlegend=False),row=rows,col=cols)
    
    marker = get_marker(ohlc, res)
    pattern_name = res.patterns

    fig.add_trace(
        go.Scatter(x=marker.index,
                   y=marker.values,
                   name=pattern_name,
                   mode='markers+lines',
                   marker_size=15,
                   marker_symbol='diamond',
                   showlegend=False,
                   line=dict(color='royalblue', width=7,dash='solid')
                   ),row=rows,col=cols)

    holidays = get_holidays(ohlc)

    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            #dict(bounds=["sat", "mon"]), # 隐藏周六、周日
            dict(values=holidays)  # 隐藏特定假期
        ])
    # showlegend=False)
    fig.update_layout(title=title, xaxis_tickformat='%Y-%m-%d',)

    return fig

def plot_subplots(res:Dict,price:pd.DataFrame):

    size = len(res)

    rows = (size // 2 + 1) if size % 2 else (size // 2)
    titles = [k + '_' + v.patterns for k, v in res.items()]
    fig = make_subplots(rows=rows, cols=2, subplot_titles=titles)

    tmp = []

    r = 1
    for i, (name, dic) in enumerate(res.items()):
        c = i % 2 + 1
        slice_df = price.query('symbol==@name').copy()
        slice_df.drop(columns=['symbol'], inplace=True)
        plot_ohlc(slice_df, dic, fig=fig, rows=r, cols=c)
        r += i % 2

    fig.update_layout(height=18000)  # ,xaxis_tickformat='%Y-%m-%d'
    return fig
'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-03-30 15:17:23
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-09 10:21:28
FilePath: \sqlalchemy_to_datad:\WorkSpace\QuantsPlaybook\C-择时类\技术分析算法框架与实战二\scr\plotting.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-03-01 15:03:00
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-03-01 15:03:37
Description: 画蜡烛图
"""
from typing import List, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from mplfinance import make_marketcolors, make_mpf_style


class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols, figsize: Tuple):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=figsize)
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None


CN_STYLE = make_mpf_style(marketcolors=make_marketcolors(up="r", down="g"))


def view_gride_chart(
    codes: Union[List, Tuple, str],
    ohlc: pd.DataFrame,
    rows: int,
    cols: int,
    figsize: Tuple,
    last_window: int = 60,
) -> plt.figure:

    gf = GridFigure(rows=rows, cols=cols, figsize=figsize)
    for code in codes:
        slice_df = ohlc.xs(code, level=1)
        mpf.plot(
            slice_df.iloc[-last_window:],
            type="candle",
            axtitle=code,
            style=CN_STYLE,
            datetime_format="%Y%m%d",
            ax=gf.next_cell(),
        )

    plt.show()
    gf.close()

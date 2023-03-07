"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-03-07 15:42:11
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-03-07 15:47:21
Description: 
"""

from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def find_stage_stock(
    pivot_price: pd.DataFrame,
    window: int,
    method: str = "high",
    offset: int = None,
) -> pd.DataFrame:
    """获取近期创新高/新低股票数量

    Args:
        pivot_price (pd.DataFrame): index-date columns-code values-price
        window (int): 窗口期
        method (str, optional): high-创新高/low-创新低. Defaults to "high".
        offset (int, optional): 是否offset. Defaults to None.

    Returns:
        pd.DataFrame: index-date columns-code values-True/False
    """
    oper: str = {"high": "ge", "low": "le"}[method]
    method: str = {"high": "max", "low": "min"}[method]

    roll_: pd.DataFrame = pivot_price.rolling(window)

    roll_df: pd.DataFrame = getattr(roll_, method)()

    if offset is not None:

        roll_df: pd.DataFrame = roll_df.shift(offset)

    return getattr(pivot_price, oper)(roll_df)


def get_ind_stage_num(pivot_num: pd.DataFrame, sw_cons_dict: Dict) -> pd.DataFrame:
    """通过个股数量统计行业创新高情况

    Args:
        pivot_num (pd.DataFrame): 个股创新高数量标记 index-date columns-code values-True/False
        sw_cons_dict (Dict): 个股所属行业字典 k-code v-industry_code

    Returns:
        pd.DataFrame: index-date columns-indutsry_code values-num
    """
    industry_num: pd.DataFrame = pivot_num.copy()
    industry_num.columns = industry_num.columns.map(sw_cons_dict)

    return industry_num.groupby(level=0, axis=1).sum()


def calc_industry_nhnl(
    pivot_price: pd.DataFrame,
    sw_cons_dict: Dict,
    window: int,
    classify_num: pd.DataFrame = None,
    tradition: bool = True,
) -> pd.DataFrame:
    """获取行业净新高占比(NHNL)

    Args:
        pivot_price (pd.DataFrame): 个股价格数据 index-date columns MultiIndex level0 fields have low|high;level1 codes
        sw_cons_dict (Dict): k-code v-indutsry_name/industry_code
        window (int): 窗口期
        tradition:True-传统构建方法;False-研报方式 使用close判断创新高/新低;默认为true

    Returns:
        pd.DataFrame: index-date columns-industry_code values-per
    """
    if classify_num is None:
        classify_num: pd.Series = pd.Series(Counter(tuple(sw_cons_dict.values())))

    h_field: str = "high"
    l_field: str = "low"
    if tradition:
        h_field, l_field = "close", "close"

    high_num: pd.DataFrame = find_stage_stock(pivot_price[h_field], window, "high", 5)
    low_num: pd.DataFrame = find_stage_stock(pivot_price[l_field], window, "low", 5)

    ind_high: pd.DataFrame = get_ind_stage_num(high_num, sw_cons_dict)
    ind_low: pd.DataFrame = get_ind_stage_num(low_num, sw_cons_dict)

    return (ind_high - ind_low).div(classify_num)


def plot_nhnl_signal(
    price: pd.Series,
    siganl: pd.Series,
    cons_num: int = None,
    title: str = "",
    align: bool = False,
) -> go.Figure:
    """画NH-NL图
    plotly >= 5.13
    Args:
        price (pd.Series): 价格数据 index-date values-price
        siganl (pd.Series): NH-NL信号 index-date values-sigbal
        cons_num (int): 行业个股上市天数超过一年的个数.Defaults is None
        title (str, optional): 标题. Defaults to "".
        align (bool, optional): 是否按照信号对齐价格数据. Defaults to False.

    Returns:
        go.Figure: 图表
    """
    fig = go.Figure()

    THRESHOLD: Dict = {
        "normal": {"贪婪": 0.3, "乐观": 0.2, "悲观": -0.2, "恐惧": -0.3},
        "other": {"贪婪": 0.4, "乐观": 0.3, "悲观": -0.3, "恐惧": -0.4},
    }

    COLOR: Dict = {
        "贪婪": {"color": "LightSeaGreen"},
        "乐观": {"color": "LightSeaGreen", "dash": "dashdot"},
        "悲观": {"color": "Crimson", "dash": "dashdot"},
        "恐惧": {"color": "Crimson"},
    }

    if align:
        siganl, price = siganl.align(price, join="inner")

    price_ax = go.Scatter(
        x=price.index,
        y=price.values,
        line=dict(color="darkgray"),
        name="close",
    )
    nhnl_ax = go.Scatter(
        x=siganl.index,
        y=siganl.values,
        line=dict(color="DarkSalmon"),
        name="NH-NL",
        yaxis="y2",
    )

    fig.add_trace(price_ax)
    fig.add_trace(nhnl_ax)

    method: str = "normal" if (cons_num > 40 or cons_num is None) else "other"
    threshold_range: Dict = THRESHOLD[method]

    for name, value in threshold_range.items():

        fig.add_trace(
            go.Scatter(
                x=price.index,
                y=np.ones(len(price)) * value,
                line=COLOR[name],
                name=name,
                yaxis="y2",
            )
        )

    fig.update_layout(
        hovermode="x unified",
        yaxis2=dict(
            title="NHNL",
            overlaying="y",
            side="right",
        ),
        title={"text": title},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig

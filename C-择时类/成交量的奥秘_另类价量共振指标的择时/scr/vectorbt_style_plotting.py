"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-10-31 11:08:01
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-12-02 13:16:06
Description: 根据backtrader的回测结果 画vectorbt风格的图表
"""

from copy import deepcopy
from typing import Dict, List, Tuple, Union

import empyrical as ep
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .timeseries import gen_drawdown_table
from .utils import max_rel_rescale, min_rel_rescale

COLORS: Dict = {
    "gray": "#7f7f7f",
    "red": "#dc3912",
    "green": "#2ca02c",
    "orange": "#ff7f0e",
    "blue": "#1f77b4",
    "purple": "#9467bd",
}

LAYOUT: Dict = dict(
    xaxis_tickformat="%Y-%m-%d",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        traceorder="normal",
    ),
    hovermode="x unified",
)


def make_figure(
    use_widgets: bool = False, *args, **kwargs
) -> Union[go.Figure, go.FigureWidget]:
    """Make new figure.

    Returns either `Figure` or `FigureWidget`, depending on `use_widgets`
    defined under `plotting` in `vectorbt._settings.settings`."""

    if use_widgets:
        return go.FigureWidget(*args, **kwargs)
    return go.Figure(*args, **kwargs)


def _plot_orders(trade_record: pd.DataFrame, fig: go.Figure = None) -> go.Figure:
    """_summary_

    Args:
        trade_record (pd.DataFrame): 2-closed 1-open
            | index | status | ref  | ticker  | dir  | datein    | pricein     | dateout   | priceout   | chng% | pnl       | pnl%    | size   | value    | cumpnl   | nbars | pnl/bar    | mfe% | mae%  |
            | :---- | :----- | :--- | :------ | :--- | :-------- | :---------- | :-------- | :--------- | :---- | :-------- | :------ | :----- | :------- | :------- | :---- | :--------- | :--- | :---- |
            | 49    | 2      | 50   | 沪深300 | long | 2016/1/26 | 3099.90996  | 2016/2/26 | 2941.80579 | -5.1  | -2.72E+07 | -0.0518 | 167756 | 5.20E+08 | 4.25E+08 | 18    | -1512174.2 | 0.91 | -8.41 |
            | 50    | 2      | 51   | 沪深300 | long | 2016/4/5  | 3211.62113  | 2016/4/21 | 3160.48392 | -1.59 | -8.61E+06 | -0.0167 | 154888 | 4.97E+08 | 4.17E+08 | 12    | -717287.81 | 2.64 | -3.25 |
            | 51    | 2      | 52   | 沪深300 | long | 2016/5/20 | 3048.084778 | 2016/6/27 | 3064.97    | 0.55  | 2.02E+06  | 0.0039  | 160280 | 4.89E+08 | 4.19E+08 | 24    | 84131.25   | 5.04 | -0.68 |
            | 52    | 2      | 53   | 沪深300 | long | 2016/8/16 | 3403.720338 | 2017/1/11 | 3355.46442 | -1.42 | -7.69E+06 | -0.0151 | 145215 | 4.94E+08 | 4.11E+08 | 98    | -78480.14  | 5.29 | -5.56 |
            | 53    | 1      | 54   | 沪深300 | long | 2017/2/20 | 3421.982164 | 2017/2/28 | 3452.81    | 0.9   | -9.71E+04 | -0.0002 | 141890 | 4.86E+08 | 4.11E+08 | NaN   | NaN        | NaN  | NaN   |
        fig (go.Figure, optional): _description_. Defaults to None.

    Returns:
        go.Figure: _description_
    """
    closed_mask: pd.DataFrame = trade_record.query("status==2")

    buy_mask: pd.DataFrame = closed_mask[["ref", "datein", "pricein", "size"]]
    sell_mask: pd.DataFrame = closed_mask[["ref", "dateout", "priceout", "size"]]
    if not buy_mask.empty:
        buy_scatter = go.Scatter(
            x=buy_mask["datein"],
            y=buy_mask["pricein"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                color=COLORS["red"],
                size=8,
                line=dict(width=1, color=COLORS["red"]),
            ),
            name="Buy",
            customdata=buy_mask.values,
            hovertemplate=f"ref: %{{customdata[0]}}"
            f"<br>Entry Timestamp: %{{x:%Y-%m-%d}}"
            f"<br>Entry Price: %{{y:.2f}}"
            f"<br>Size: %{{customdata[3]:.6f}}",
        )

        fig.add_trace(buy_scatter)

    if not buy_mask.empty:

        sell_scatter = go.Scatter(
            x=sell_mask["dateout"],
            y=sell_mask["priceout"],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                color=COLORS["green"],
                size=8,
                line=dict(width=1, color=COLORS["green"]),
            ),
            name="Sell",
            customdata=sell_mask.values,
            hovertemplate=f"ref: %{{customdata[0]}}"
            f"<br>Exit Timestamp: %{{x:%Y-%m-%d}}"
            f"<br>Exit Price: %{{y:.2f}}"
            f"<br>Size: %{{customdata[3]:.6f}}",
        )

        fig.add_trace(sell_scatter)

    active_df: pd.DataFrame = trade_record.query("status==1")[
        ["ref", "datein", "pricein", "size"]
    ]
    if not active_df.empty:

        active_scatter = go.Scatter(
            x=active_df["datein"],
            y=active_df["pricein"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                color=COLORS["orange"],
                size=8,
                line=dict(width=1, color=COLORS["orange"]),
            ),
            name="Active",
            customdata=active_df.values,
            hovertemplate=f"ref: %{{customdata[0]}}"
            f"<br>Entry Timestamp: %{{x:%Y-%m-%d}}"
            f"<br>Entry Price: %{{y:.2f}}"
            f"<br>Size: %{{customdata[3]:.6f}}",
        )

        fig.add_trace(active_scatter)

    return fig


def _plot_position(
    trade_record: pd.DataFrame,
    plot_zones: bool = True,
    xref: str = "x",
    yref: str = "y",
    fig: go.Figure = None,
) -> go.Figure:
    """_summary_

    Args:
        trade_record (pd.DataFrame):status 2-closed 1-open
            | index | status | ref  | ticker  | dir  | datein    | pricein     | dateout   | priceout   | chng% | pnl       | pnl%    | size   | value    | cumpnl   | nbars | pnl/bar    | mfe% | mae%  |
            | :---- | :----- | :--- | :------ | :--- | :-------- | :---------- | :-------- | :--------- | :---- | :-------- | :------ | :----- | :------- | :------- | :---- | :--------- | :--- | :---- |
            | 49    | 2      | 50   | 沪深300 | long | 2016/1/26 | 3099.90996  | 2016/2/26 | 2941.80579 | -5.1  | -2.72E+07 | -0.0518 | 167756 | 5.20E+08 | 4.25E+08 | 18    | -1512174.2 | 0.91 | -8.41 |
            | 50    | 2      | 51   | 沪深300 | long | 2016/4/5  | 3211.62113  | 2016/4/21 | 3160.48392 | -1.59 | -8.61E+06 | -0.0167 | 154888 | 4.97E+08 | 4.17E+08 | 12    | -717287.81 | 2.64 | -3.25 |
            | 51    | 2      | 52   | 沪深300 | long | 2016/5/20 | 3048.084778 | 2016/6/27 | 3064.97    | 0.55  | 2.02E+06  | 0.0039  | 160280 | 4.89E+08 | 4.19E+08 | 24    | 84131.25   | 5.04 | -0.68 |
            | 52    | 2      | 53   | 沪深300 | long | 2016/8/16 | 3403.720338 | 2017/1/11 | 3355.46442 | -1.42 | -7.69E+06 | -0.0151 | 145215 | 4.94E+08 | 4.11E+08 | 98    | -78480.14  | 5.29 | -5.56 |
            | 53    | 1      | 54   | 沪深300 | long | 2017/2/20 | 3421.982164 | 2017/2/28 | 3452.81    | 0.9   | -9.71E+04 | -0.0002 | 141890 | 4.86E+08 | 4.11E+08 | NaN   | NaN        | NaN  | NaN   |

        plot_zones (bool, optional): _description_. Defaults to True.
        xref (str, optional): _description_. Defaults to "x".
        yref (str, optional): _description_. Defaults to "y".
        fig (go.Figure, optional): _description_. Defaults to None.

    Returns:
        go.Figure: _description_
    """
    if fig is None:
        fig = go.Figure()

    df = trade_record.query("status==2")

    entry_hovertemplate = (
        f"ref: %{{customdata[0]}}"
        f"<br>Size: %{{customdata[1]:.2f}}"
        f"<br>Entry Timestamp: %{{x:%Y-%m-%d}}"
        f"<br>Avg Entry Price: %{{y:.2f}}"
        f"<br>Direction: %{{customdata[4]}}"
    )

    entry_scatter = go.Scatter(
        x=trade_record["datein"],
        y=trade_record["pricein"],
        mode="markers",
        marker=dict(
            symbol="square",
            color=COLORS["blue"],
            size=7,
            line=dict(width=1, color=COLORS["blue"]),
        ),
        name="Entry",
        customdata=trade_record[["ref", "size", "datein", "pricein", "dir"]],
        hovertemplate=entry_hovertemplate,
    )

    fig.add_trace(entry_scatter)

    def _plot_end_markers(mask: pd.DataFrame, name: str, color: str, **kwargs) -> None:

        exit_customdata: np.ndarray = mask.values
        exit_hovertemplate = (
            f"ref: %{{customdata[0]}}"
            f"<br>Size: %{{customdata[1]:.2f}}"
            f"<br>Exit Timestamp: %{{x:%Y-%m-%d}}"
            f"<br>Avg Exit Price: %{{y:.2f}}"
            f"<br>PnL: %{{customdata[4]:.2f}}"
            f"<br>Return: %{{customdata[5]:.2%}}"
            f"<br>Direction: %{{customdata[6]}}"
            f"<br>Duration: %{{customdata[7]}}"
        )

        scatter = go.Scatter(
            x=mask["dateout"],
            y=mask["priceout"],
            mode="markers",
            marker=dict(
                symbol="square",
                color=color,
                size=7,
                line=dict(width=1, color=COLORS[color]),
            ),
            name=name,
            customdata=exit_customdata,
            hovertemplate=exit_hovertemplate,
        )

        scatter.update(**kwargs)
        fig.add_trace(scatter)

    _plot_end_markers(
        df.query("pnl==0")[
            ["ref", "size", "dateout", "priceout", "pnl", "pnl%", "dir", "nbars"]
        ],
        "Exit",
        "gray",
    )
    _plot_end_markers(
        df.query("pnl>0")[
            ["ref", "size", "dateout", "priceout", "pnl", "pnl%", "dir", "nbars"]
        ],
        "Exit - Profit",
        "red",
    )
    _plot_end_markers(
        df.query("pnl<0")[
            ["ref", "size", "dateout", "priceout", "pnl", "pnl%", "dir", "nbars"]
        ],
        "Exit - Loss",
        "green",
    )
    _plot_end_markers(
        trade_record.query("status==1")[
            ["ref", "size", "dateout", "priceout", "pnl", "pnl%", "dir", "nbars"]
        ],
        "Active",
        "orange",
    )

    if plot_zones:
        profit_mask: pd.DataFrame = trade_record.query("pnl > 0")
        if not profit_mask.empty:
            # Plot profit zones
            for _, rows in profit_mask.iterrows():
                fig.add_shape(
                    **dict(
                        type="rect",
                        xref=xref,
                        yref=yref,
                        x0=rows["datein"],
                        y0=rows["pricein"],
                        x1=rows["dateout"],
                        y1=rows["priceout"],
                        fillcolor="red",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                    )
                )

        loss_mask = trade_record.query("pnl < 0")
        if not loss_mask.empty:
            # Plot loss zones
            for _, rows in loss_mask.iterrows():
                fig.add_shape(
                    **dict(
                        type="rect",
                        xref=xref,
                        yref=yref,
                        x0=rows["datein"],
                        y0=rows["pricein"],
                        x1=rows["dateout"],
                        y1=rows["priceout"],
                        fillcolor="green",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                    )
                )

    return fig


def plot_against(
    ser: pd.Series,
    other: pd.Series,
    trace_kwargs: Dict = None,
    other_trace_kwargs: Dict = None,
    fig=None,
) -> go.Figure:
    """Plot Series as a line against another line.

    Args:
        ser (pd.Series): 主线
        other (pd.Series, optional): 副线.
        trace_kwargs (Dict, optional): 主线参数. Defaults to None.
        other_trace_kwargs (Dict, optional): 副线参数. Defaults to None.
        fig (go.Figure, optional): Figure to add traces to. Defaults to None.

    Returns:
        go.Figure: go.Figure
    """
    if trace_kwargs is None:
        trace_kwargs: Dict = {}
    if other_trace_kwargs is None:
        other_trace_kwargs: Dict = {}

    ser_, other_ = ser.align(other, axis=0, join="left")

    pos_mask: pd.Series = ser_ > other_

    if fig is None:
        fig = go.Figure()

    if pos_mask.any():
        # Fill positive area
        pos_obj = ser_.copy()
        pos_obj[~pos_mask] = other_[~pos_mask]

        fig.add_trace(
            go.Scatter(
                x=other_.index,
                y=other_.values,
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                opacity=0,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pos_obj.index,
                y=pos_obj.values,
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                opacity=0,
                fill="tonexty",
                connectgaps=False,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )

    neg_mask: pd.Series = ser_ < other_
    if neg_mask.any():
        # Fill negative area
        neg_obj = ser_.copy()
        neg_obj[~neg_mask] = other_[~neg_mask]
        fig.add_trace(
            go.Scatter(
                x=other_.index,
                y=other_.values,
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                opacity=0,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=neg_obj.index,
                y=neg_obj.values,
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                fillcolor="rgba(0, 128, 0, 0.3)",
                opacity=0,
                fill="tonexty",
                connectgaps=False,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )

    fig.add_trace(go.Scatter(x=ser_.index, y=ser_.values, **trace_kwargs))
    if other_trace_kwargs == "hidden":
        other_trace_kwargs: Dict = dict(
            line=dict(color="rgba(0, 0, 0, 0)", width=0),
            opacity=0.0,
            hoverinfo="skip",
            showlegend=False,
            name=None,
        )
    fig.add_trace(go.Scatter(x=other_.index, y=other_.values, **other_trace_kwargs))

    return fig


def plot_position(
    ser: pd.Series,
    trade_record: pd.DataFrame,
    use_widgets: bool,
    **layout_kwargs,
) -> go.Figure:
    """画position

    Args:
        ser (pd.Series): 时序数据 index-datetime values
        trade_record (pd.DataFrame): 2-closed 1-open
            | index | status | ref  | ticker  | dir  | datein    | pricein     | dateout   | priceout   | chng% | pnl       | pnl%    | size   | value    | cumpnl   | nbars | pnl/bar    | mfe% | mae%  |
            | :---- | :----- | :--- | :------ | :--- | :-------- | :---------- | :-------- | :--------- | :---- | :-------- | :------ | :----- | :------- | :------- | :---- | :--------- | :--- | :---- |
            | 49    | 2      | 50   | 沪深300 | long | 2016/1/26 | 3099.90996  | 2016/2/26 | 2941.80579 | -5.1  | -2.72E+07 | -0.0518 | 167756 | 5.20E+08 | 4.25E+08 | 18    | -1512174.2 | 0.91 | -8.41 |
            | 50    | 2      | 51   | 沪深300 | long | 2016/4/5  | 3211.62113  | 2016/4/21 | 3160.48392 | -1.59 | -8.61E+06 | -0.0167 | 154888 | 4.97E+08 | 4.17E+08 | 12    | -717287.81 | 2.64 | -3.25 |
            | 51    | 2      | 52   | 沪深300 | long | 2016/5/20 | 3048.084778 | 2016/6/27 | 3064.97    | 0.55  | 2.02E+06  | 0.0039  | 160280 | 4.89E+08 | 4.19E+08 | 24    | 84131.25   | 5.04 | -0.68 |
            | 52    | 2      | 53   | 沪深300 | long | 2016/8/16 | 3403.720338 | 2017/1/11 | 3355.46442 | -1.42 | -7.69E+06 | -0.0151 | 145215 | 4.94E+08 | 4.11E+08 | 98    | -78480.14  | 5.29 | -5.56 |
            | 53    | 1      | 54   | 沪深300 | long | 2017/2/20 | 3421.982164 | 2017/2/28 | 3452.81    | 0.9   | -9.71E+04 | -0.0002 | 141890 | 4.86E+08 | 4.11E+08 | NaN   | NaN        | NaN  | NaN   |

    Returns:
        go.Figure:
    """
    fig = make_figure(
        use_widgets=use_widgets,
        data=[
            go.Scatter(
                x=ser.index, y=ser.values, line=dict(color=COLORS["blue"]), name="Close"
            )
        ],
    )

    if layout_kwargs:
        layout_kwargs.update(LAYOUT)
    else:
        layout_kwargs: Dict = deepcopy(LAYOUT)
        del layout_kwargs["hovermode"]

    fig.update_layout(**layout_kwargs)
    return _plot_position(trade_record, fig=fig)


def plot_orders(
    ser: pd.Series,
    trade_record: pd.DataFrame,
    use_widgets: bool,
    **layout_kwargs,
) -> go.Figure:
    """_summary_

    Args:
        ser (pd.Series): _description_
        trade_record (pd.DataFrame): 1-open 2-closed
            | index | status | ref  | ticker  | dir  | datein    | pricein     | dateout   | priceout   | chng% | pnl       | pnl%    | size   | value    | cumpnl   | nbars | pnl/bar    | mfe% | mae%  |
            | :---- | :----- | :--- | :------ | :--- | :-------- | :---------- | :-------- | :--------- | :---- | :-------- | :------ | :----- | :------- | :------- | :---- | :--------- | :--- | :---- |
            | 49    | 2      | 50   | 沪深300 | long | 2016/1/26 | 3099.90996  | 2016/2/26 | 2941.80579 | -5.1  | -2.72E+07 | -0.0518 | 167756 | 5.20E+08 | 4.25E+08 | 18    | -1512174.2 | 0.91 | -8.41 |
            | 50    | 2      | 51   | 沪深300 | long | 2016/4/5  | 3211.62113  | 2016/4/21 | 3160.48392 | -1.59 | -8.61E+06 | -0.0167 | 154888 | 4.97E+08 | 4.17E+08 | 12    | -717287.81 | 2.64 | -3.25 |
            | 51    | 2      | 52   | 沪深300 | long | 2016/5/20 | 3048.084778 | 2016/6/27 | 3064.97    | 0.55  | 2.02E+06  | 0.0039  | 160280 | 4.89E+08 | 4.19E+08 | 24    | 84131.25   | 5.04 | -0.68 |
            | 52    | 2      | 53   | 沪深300 | long | 2016/8/16 | 3403.720338 | 2017/1/11 | 3355.46442 | -1.42 | -7.69E+06 | -0.0151 | 145215 | 4.94E+08 | 4.11E+08 | 98    | -78480.14  | 5.29 | -5.56 |
            | 53    | 1      | 54   | 沪深300 | long | 2017/2/20 | 3421.982164 | 2017/2/28 | 3452.81    | 0.9   | -9.71E+04 | -0.0002 | 141890 | 4.86E+08 | 4.11E+08 | NaN   | NaN        | NaN  | NaN   |

    Returns:
        go.Figure: _description_
    """
    if layout_kwargs:
        layout_kwargs.update(LAYOUT)
    else:
        layout_kwargs: Dict = deepcopy(LAYOUT)
        del layout_kwargs["hovermode"]

    fig = make_figure(
        use_widgets=use_widgets,
        data=[
            go.Scatter(
                x=ser.index, y=ser, line=dict(color=COLORS["blue"]), name="Close"
            )
        ],
    )
    fig.update_layout(**layout_kwargs)
    return _plot_orders(trade_record, fig=fig)


def plot_cumulative(
    rets: pd.Series,
    benchmark_rets: pd.Series = None,
    start_value: float = 0.0,
    fill_to_benchmark: bool = False,
    main_kwargs: Dict = None,
    benchmark_kwargs: Dict = None,
    fig: go.Figure = None,
    use_widgets: bool = False,
    **layout_kwargs,
) -> go.Figure:

    if fig is None:
        fig = make_figure(use_widgets=use_widgets)

    if layout_kwargs:
        layout_kwargs.update(LAYOUT)
    else:
        layout_kwargs = LAYOUT

    fig.update_layout(**layout_kwargs)

    if benchmark_rets is not None:
        rets, benchmark_rets_ = rets.align(benchmark_rets, axis=0, join="left")
        if benchmark_kwargs is None:
            benchmark_kwargs: Dict = {}
        benchmark_kwargs.update(dict(line=dict(color=COLORS["gray"]), name="Benchmark"))
        benchmark_cumrets: pd.Series = ep.cum_returns(benchmark_rets_, start_value)
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumrets.index,
                y=benchmark_cumrets.values,
                **benchmark_kwargs,
            )
        )
    if main_kwargs is None:
        main_kwargs: Dict = {}
    main_kwargs.update(dict(line=dict(color=COLORS["purple"])))
    cumrets: pd.Series = ep.cum_returns(rets, start_value)
    if fill_to_benchmark:
        fig = plot_against(
            cumrets,
            benchmark_cumrets,
            trace_kwargs=main_kwargs,
            other_trace_kwargs=benchmark_kwargs,
            fig=fig,
        )
    else:
        benchmark_kwargs: str = "hidden"
        fig = plot_against(
            cumrets,
            pd.Series(index=cumrets.index, data=[start_value] * len(cumrets)),
            trace_kwargs=main_kwargs,
            other_trace_kwargs=benchmark_kwargs,
            fig=fig,
        )

    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        y0=start_value,
        x1=1,
        y1=start_value,
        line=dict(
            color="gray",
            dash="dash",
        ),
    )

    return fig


def plot_underwater(
    rets: pd.Series,
    xref: str = "x",
    yref: str = "y",
    use_widgets: bool = False,
    **layout_kwargs,
) -> go.Figure:

    if layout_kwargs:
        layout_kwargs.update(LAYOUT)
    else:
        layout_kwargs = LAYOUT

    cum_ser: pd.Series = ep.cum_returns(rets, 1)
    maxdrawdown: pd.Series = cum_ser / cum_ser.cummax() - 1

    fig = make_figure(
        use_widgets=use_widgets,
        data=[
            go.Scatter(
                x=maxdrawdown.index,
                y=maxdrawdown.values,
                fillcolor="rgba(220,57,18,0.3000)",
                line=dict(color=COLORS["red"]),
                fill="tozeroy",
                name="Drawdown",
                hovertemplate=(f"<br>Drawdown:%{{y:.2%}}" f"<br>Date:%{{x:%Y-%m-%d}}"),
            )
        ],
    )

    fig.update_layout(**layout_kwargs)

    fig.add_shape(
        type="line",
        line=dict(color="gray", dash="dash"),
        xref="paper",
        yref="y",
        x0=0,
        y0=0,
        x1=1,
        y1=0,
    )

    fig.update_layout(**layout_kwargs)
    yaxis = "yaxis" + yref[1:]
    fig.layout[yaxis]["tickformat"] = ".2%"
    return fig


def plot_pnl(
    trade_record: pd.DataFrame,
    pct_scale: bool = True,
    marker_size_range: Tuple[int, int] = (7, 14),
    opacity_range: Tuple[float, float] = (0.75, 0.9),
    fig: go.Figure = None,
    xref: str = "x",
    yref: str = "y",
    use_widgets: bool = False,
    **layout_kwargs,
) -> go.Figure:

    xaxis: str = "xaxis" + xref[1:]
    yaxis: str = "yaxis" + yref[1:]

    if fig is None:
        fig = make_figure(use_widgets=use_widgets)

    if layout_kwargs:
        layout_kwargs.update(LAYOUT)
    else:
        layout_kwargs: Dict = LAYOUT

    fig.update_layout(**layout_kwargs)

    if pct_scale:
        _layout_kwargs: Dict = {}
        _layout_kwargs[yaxis] = dict(tickformat=".2%")
        fig.update_layout(**_layout_kwargs)

    marker_size: pd.Series = min_rel_rescale(
        trade_record["pnl%"].abs(), marker_size_range
    )
    opacity: pd.Series = max_rel_rescale(trade_record["pnl%"].abs(), opacity_range)

    open_mask: pd.Series = trade_record["status"] == 1
    closed_frame: pd.Series = trade_record["status"] == 2
    closed_profit_mask: pd.Series = closed_frame & (trade_record["pnl"] > 0)
    closed_loss_mask: pd.Series = closed_frame & (trade_record["pnl"] < 0)

    def _plot_scatter(mask: pd.Series, name: str, color: str) -> None:

        hovertemplate = (
            f"ref: %{{customdata[0]}}"
            f"<br>Exit Timestamp: %{{x:%Y-%m-%d}}"
            f"<br>PnL: %{{customdata[1]:.2f}}"
            f"<br>Return: %{{customdata[2]:.2%}}"
        )
        customdata: pd.DataFrame = trade_record.loc[mask, ["ref", "pnl", "pnl%"]]
        if not mask.empty:

            scatter = go.Scatter(
                x=trade_record.loc[mask, "dateout"],
                y=trade_record.loc[mask, "pnl%"]
                if pct_scale
                else trade_record.loc[mask, "pnl"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color=color,
                    size=marker_size[mask],
                    opacity=opacity[mask],
                    line=dict(width=1, color=COLORS[color]),
                ),
                name=name,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
            fig.add_trace(scatter)

    # Plot Closed - Profit scatter
    _plot_scatter(closed_profit_mask, "Closed - Profit", "red")

    # Plot Closed - Profit scatter
    _plot_scatter(
        closed_loss_mask,
        "Closed - Loss",
        "green",
    )

    # Plot Open scatter
    _plot_scatter(open_mask, "Open", "orange")

    # Plot zeroline
    fig.add_shape(
        type="line",
        xref="paper",
        yref=yref,
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        line=dict(
            color="gray",
            dash="dash",
        ),
    )

    return fig


def plot_drawdowns(
    rets: pd.Series,
    top_n: int = 5,
    start_value: float = 0.0,
    plot_zones: bool = True,
    xref: str = "x",
    yref: str = "y",
    ts_trace_kwargs: Dict = None,
    use_widgets: bool = False,
    fig: go.Figure = None,
    **layout_kwargs,
) -> go.Figure:

    NAME_DICT: Dict = {1.0: "Nav", 0.0: "Cum"}
    if layout_kwargs:
        layout_kwargs.update(LAYOUT)
    else:
        layout_kwargs: Dict = deepcopy(LAYOUT)
        del layout_kwargs["hovermode"]

    # 计算累计收益率序列,当start_value为1时为净值序列
    line_ser: pd.Series = ep.cum_returns(rets, start_value)
    # 计算top_n的最大回撤表格
    drawdown_table: pd.DataFrame = gen_drawdown_table(rets, top_n)
    drawdown_table["id"] = np.arange(top_n) + 1
    drawdown_table["peak value"] = drawdown_table["Peak date"].map(line_ser)
    drawdown_table["valley value"] = drawdown_table["Valley date"].map(line_ser)
    drawdown_table["recovery value"] = drawdown_table["Recovery date"].map(
        lambda x: line_ser.to_dict().get(x, line_ser[-1])
    )
    drawdown_table["Net drawdown in %"] /= 100

    if ts_trace_kwargs is None:
        ts_trace_kwargs: Dict = {}
    if "name" not in ts_trace_kwargs:

        ts_trace_kwargs.update({"name": NAME_DICT.get(start_value, "Value")})

    ts_trace_kwargs.update(dict(line=dict(color=COLORS["blue"])))

    if fig is None:

        fig = make_figure(use_widgets=use_widgets)

    fig.add_trace(go.Scatter(x=line_ser.index, y=line_ser.values, **ts_trace_kwargs))
    fig.update_layout(**layout_kwargs)

    # Plot peak markers
    peak_customdata: np.ndarray = drawdown_table[["id", "Peak date"]].values
    peak_scatter = go.Scatter(
        x=drawdown_table["Peak date"],
        y=drawdown_table["peak value"],
        mode="markers",
        marker=dict(
            symbol="diamond",
            color=COLORS["blue"],
            size=7,
            line=dict(width=1, color=COLORS["blue"]),
        ),
        name="Peak",
        customdata=peak_customdata,
        hovertemplate=(
            f"<br>Top Drawdowns id:%{{customdata[0]}}"
            f"<br>Peak Date:%{{x:%Y-%m-%d}}"
            f"<br>Peak Value:%{{y:.2f}}"
        ),
    )
    fig.add_trace(peak_scatter)

    # Recovery date为nan的是还处于回撤期间的即active
    recovered_mask: pd.Series = ~drawdown_table["Recovery date"].isna()

    if recovered_mask.any():

        # Plot valley markers
        valley_customdata: np.ndarray = drawdown_table.loc[
            recovered_mask, ["id", "Net drawdown in %", "Valley Duration"]
        ].values
        valley_scatter = go.Scatter(
            x=drawdown_table.loc[recovered_mask, "Valley date"],
            y=drawdown_table.loc[recovered_mask, "valley value"],
            mode="markers",
            marker=dict(
                symbol="diamond",
                color=COLORS["red"],
                size=7,
                line=dict(width=1, color=COLORS["red"]),
            ),
            name="Valley",
            customdata=valley_customdata,
            hovertemplate=f"<br>Top Drawdowns id: %{{customdata[0]}}"
            f"<br>Valley Date: %{{x:%Y-%m-%d}}"
            f"<br>Valley Value: %{{y:.2f}}"
            f"<br>Drawdown: %{{customdata[1]:.2%}}"
            f"<br>Duration: %{{customdata[2]}}",
        )
        fig.add_trace(valley_scatter)

        if plot_zones:

            # Plot drawdown zones
            for _, rows in drawdown_table.loc[recovered_mask].iterrows():

                fig.add_shape(
                    type="rect",
                    xref=xref,
                    yref="paper",
                    x0=rows["Peak date"],
                    y0=0,
                    x1=rows["Valley date"],
                    y1=1,
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

        # Plot recovery markers
        recovery_customdata: np.ndarray = drawdown_table.loc[
            recovered_mask, ["id", "Net drawdown in %", "End Duration", "Duration"]
        ]
        recovery_scatter = go.Scatter(
            x=drawdown_table.loc[recovered_mask, "Recovery date"],
            y=drawdown_table.loc[recovered_mask, "recovery value"],
            mode="markers",
            marker=dict(
                symbol="diamond",
                color=COLORS["green"],
                size=7,
                line=dict(width=1, color=COLORS["green"]),
            ),
            name="Recovery/Peak",
            customdata=recovery_customdata,
            hovertemplate=(
                f"<br>Top Drawdowns id: %{{customdata[0]}}"
                f"<br>Recovery/Peak Date: %{{x:%Y-%m-%d}}"
                f"<br>Recovery/Peak Value: %{{y:.2f}}"
                f"<br>Return: %{{customdata[1]:.2%}}"
                f"<br>End Duration: %{{customdata[2]}}"
                f"<br>Duration: %{{customdata[3]}}"
            ),
        )
        fig.add_trace(recovery_scatter)

        if plot_zones:
            # Plot recovery zones
            for _, rows in drawdown_table.loc[recovered_mask].iterrows():
                fig.add_shape(
                    type="rect",
                    xref=xref,
                    yref="paper",
                    x0=rows["Valley date"],
                    y0=0,
                    x1=rows["Recovery date"],
                    y1=1,
                    fillcolor="green",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

    # Plot active markers
    active_mask: pd.Series = ~recovered_mask
    if active_mask.any():
        peak_date = drawdown_table.loc[active_mask, "Peak date"].values[0]
        active_date = line_ser.index[-1]
        active_customdata: pd.Series = drawdown_table.loc[
            active_mask, ["id", "Net drawdown in %", "Duration"]
        ]
        active_customdata["Duration"] = len(
            pd.date_range(peak_date, active_date, freq="B")
        )
        active_customdata: np.ndarray = active_customdata.values
        active_scatter = go.Scatter(
            x=[active_date],
            y=[drawdown_table.loc[active_mask, "recovery value"].values[0]],
            mode="markers",
            marker=dict(
                symbol="diamond",
                color=COLORS["orange"],
                size=7,
                line=dict(width=1, color=COLORS["orange"]),
            ),
            name="Active",
            customdata=active_customdata,
            hovertemplate=(
                f"<br>Top Drawdowns id: %{{customdata[0]}}"
                f"<br>Active Date: %{{x:%Y-%m-%d}}"
                f"<br>Active Value: %{{y:.2}}"
                f"<br>Return: %{{customdata[1]:.2%}}"
                f"<br>Duration: %{{customdata[2]}}"
            ),
        )

        fig.add_trace(active_scatter)

        if plot_zones:
            # Plot active drawdown zones

            fig.add_shape(
                type="rect",
                xref=xref,
                yref="paper",
                x0=pd.to_datetime(peak_date),
                y0=0,
                x1=pd.to_datetime(active_date),
                y1=1,
                fillcolor="orange",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

    return fig


def plot_annual_returns(
    returns: Union[pd.Series, List], use_widgets: bool = False
) -> go.Figure:
    """画每年累计收益

    Args:
        returns (pd.Series): index-date values-returns

    Returns:
        _type_: _description_
    """
    if isinstance(returns, pd.Series):
        ann_ret_df: pd.Series = ep.aggregate_returns(returns, "yearly")
    elif isinstance(returns, List):
        ann_ret_df: pd.Series = pd.Series(
            returns[0].analyzers._AnnualReturn.get_analysis()
        )
    else:
        raise ValueError("returns类型必须为pd.Series或bt_result.result")

    colors: List = ["crimson" if v > 0 else "#7a9e9f" for v in ann_ret_df]

    fig = make_figure(
        use_widgets=use_widgets,
        data=go.Bar(
            x=ann_ret_df.values,
            y=ann_ret_df.index,
            orientation="h",
            marker_color=colors,
        ),
    )

    fig.update_layout(
        title={"text": "Annual returns", "x": 0.5, "y": 0.9},
        yaxis_title="Year",
        xaxis_tickformat=".2%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    fig.add_vline(x=ann_ret_df.mean(), line_dash="dash")

    return fig


def plot_monthly_heatmap(returns: pd.Series, use_widgets: bool = False) -> go.Figure:
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
    monthly_ret_table: pd.Series = ep.aggregate_returns(returns, "monthly")
    monthly_ret_table: pd.Series = monthly_ret_table.unstack().round(3)

    fig = make_figure(
        use_widgets=use_widgets,
        data=go.Heatmap(
            z=monthly_ret_table.values,
            x=monthly_ret_table.columns.map(str),
            y=monthly_ret_table.index.map(str),
            text=monthly_ret_table.values,
            texttemplate="%{text:.2%}",
        ),
    )
    fig.update_layout(
        title={"text": "Monthly returns (%)", "x": 0.5, "y": 0.9},
        yaxis_title="Year",
        xaxis_title="Month",
    )
    return fig


def plot_monthly_dist(returns: pd.Series, use_widgets: bool = False) -> go.Figure:
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
    monthly_ret_table = pd.DataFrame(
        ep.aggregate_returns(returns, "monthly"), columns=["Returns"]
    )
    fig = px.histogram(monthly_ret_table, x="Returns")
    mean_returns = monthly_ret_table["Returns"].mean()
    fig.add_vline(
        x=mean_returns,
        line_dash="dash",
        annotation_text="Mean:{:.2f}".format(mean_returns),
    )
    fig.update_layout(
        hovermode="x unified",
        title={"text": "Distribution of monthly returns", "x": 0.5, "y": 0.9},
        yaxis_title="Number of months",
        xaxis_tickformat=".2%",
        xaxis_title="Returns",
    )
    return make_figure(use_widgets, fig)


def plot_table(
    df: pd.DataFrame, use_widgets: bool = False, index_name: str = ""
) -> go.Figure:

    df.index.names = [index_name]
    df: pd.DataFrame = df.reset_index()

    headerColor = "grey"
    # rowEvenColor = "lightgrey"
    # rowOddColor = "white"

    fig = make_figure(
        use_widgets=use_widgets,
        data=[
            go.Table(
                header=dict(
                    values=df.columns,
                    line_color="darkslategray",
                    fill_color=headerColor,
                    align=["left", "center"],
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=df.T.values,
                    line_color="darkslategray",
                    # 2-D list of colors for alternating rows
                    # fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
                    align=["left", "center"],
                    font=dict(color="darkslategray", size=11),
                ),
            )
        ],
    )

    return fig

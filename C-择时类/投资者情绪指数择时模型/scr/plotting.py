from typing import Dict

import pandas as pd
import plotly.graph_objects as go


def plot_industry_circle(
    data: pd.DataFrame,
    group_name: str = "industry_name",
    x: str = "relative_strength",
    y: str = "relative_strength_mom",
    title: str = "行业相对轮动",
):

    fig = go.Figure()

    colors = [
        "#096631",
        "#0f86c3",
        "#ffdf01",
        "#dd801e",
        "#b84747",
        "#9758e0",
        "#eecf85",
        "#0c424b",
        "#31baff",
        "#533527",
        "#7dc9ce",
        "#2ec2d7",
    ]
    size = len(colors)

    for i, (key, df) in enumerate(data.groupby(group_name)):

        df = df.sort_values("trade_date", ascending=False)

        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y],
                showlegend=False,
                # name=key,
                text=df[["trade_date", group_name]],
                line=dict(color=colors[i % size]),
                mode="lines",
                #    marker_symbol='line',
                #    marker_size=10,
                hovertemplate="<b>%{text[1]}</b><br><br>"
                + "<i>相对强弱的动量</i>: %{y:.2f}"
                + "<br>相对强弱: %{x:.2f}<br>"
                + "<br>日期:%{text[0]}</br><extra></extra>",
            )
        )

        fig.add_annotation(
            x=df[x].iloc[0],  # arrows' head
            y=df[y].iloc[0],  # arrows' head
            ax=df[x].iloc[1],  # arrows' tail
            ay=df[y].iloc[1],  # arrows' tail
            # xref='x',
            # yref='y',
            axref="x",
            ayref="y",
            text="",  # if you want only the arrow
            showarrow=True,
            arrowhead=4,
            arrowsize=3,
            arrowwidth=1,
            arrowcolor=colors[i % size],
        )

        fig.add_trace(
            go.Scatter(
                mode="markers",
                showlegend=True,
                x=df[x].iloc[1:],
                y=df[y].iloc[1:],
                name=key,
                marker_symbol="circle",
                marker_line_color=colors[i % size],
                marker_color="#e5ecf6",
                marker_line_width=2,
                marker_size=8,
                text=df[["trade_date", group_name]].iloc[1:],
                hovertemplate="<b>%{text[1]}</b><br><br>"
                + "<i>相对强弱的动量</i>: %{y:.2f}"
                + "<br>相对强弱: %{x:.2f}<br>"
                + "<br>日期:%{text[0]}</br><extra></extra>",
            )
        )

    annotation_para = get_industry_circle_quadrant(data, x, y)

    for name, k in annotation_para.items():

        x = k["x"]
        y = k["y"]
        fontcolor = k["fontcolor"]
        bgcolor = k["bgcolor"]

        fig.add_annotation(
            x=x,
            y=y,
            showarrow=False,
            text=name,
            font=dict(size=20, color=fontcolor),
            align="center",
            bgcolor=bgcolor,
            # opacity=0.8
        )

    fig.update_xaxes(linewidth=1, zerolinecolor="black", zeroline=True)
    fig.update_yaxes(linewidth=1, zerolinecolor="black", zeroline=True)
    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.95, "font": {"size": 20}},
        height=1000,
        showlegend=True,
        xaxis_title="相对强弱",
        yaxis_title="相对强弱的动量",
    )

    return fig


def get_industry_circle_quadrant(data: pd.DataFrame, x: str, y: str) -> Dict:

    # labels = (('领涨', 'trend > 0 and trend_mom > 0'),
    #           ('走强', 'trend < 0 and trend_mom > 0'),
    #           ('领跌', 'trend < 0 and trend_mom < 0'),
    #           ('走弱', 'trend > 0 and trend_mom < 0'))

    y_max = data[y].max()
    y_min = data[y].min()
    x_max = data[x].max()
    x_min = data[x].min()

    return {
        "走强": {"x": x_min, "y": y_max, "bgcolor": "#fff2f4", "fontcolor": "#f92d4f"},
        "领涨": {"x": x_max, "y": y_max, "bgcolor": "#fff2f4", "fontcolor": "#f92d4f"},
        "领跌": {"x": x_min, "y": y_min, "bgcolor": "#eefff6", "fontcolor": "#09ab7d"},
        "走弱": {"x": x_max, "y": y_min, "bgcolor": "#eefff6", "fontcolor": "#09ab7d"},
    }

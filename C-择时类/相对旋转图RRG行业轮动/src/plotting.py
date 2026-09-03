"""
Author: Hugo
Date: 2024-07-29 12:55:20
LastEditors: Hugo
LastEditTime: 2026-05-25 20:28:26
Description: 
"""

import math
from itertools import cycle
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyecharts.options as opts
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pyecharts.charts import HeatMap, Line
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig

# pyecharts 默认 CDN ``assets.pyecharts.org`` 常证书过期/不可达 → echarts.min.js
# 加载失败、图全空白（render() 文件、render_notebook、render_embed 全中招，与操作
# 系统无关）。在本模块 import 时把全局 ONLINE_HOST 切到稳定的 jsdelivr，覆盖**所有**
# 渲染路径（charts 在构造时读 ONLINE_HOST 定 js_host）。China 网络若 jsdelivr 受限，
# 调用前自行 ``CurrentConfig.ONLINE_HOST = "https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/"``。
_ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/dist/"
CurrentConfig.ONLINE_HOST = _ECHARTS_CDN

# 四象限调色板（领先 / 改善 / 滞后 / 疲软），matplotlib 与 echart 两后端共用：
# mpl 路径直接用 hex（Rectangle color + text color），echart 路径经
# _hex_to_rgba 派生 rgba 填充——单一事实源，两后端配色严格一致。
_QUADRANT_HEX: Tuple[str, str, str, str] = (
    "#38b48b",  # Q1 领先
    "#44617b",  # Q2 改善
    "#bf783a",  # Q3 滞后
    "#745399",  # Q4 疲软
)

# —— plot_etf_holding_heatmap（行业树版）的未分类枝色 ——
# 代表行业缺失（etf_industry 查不到）的 ETF 归「未分类」枝，用中性灰蓝标识。
_UNCLASSIFIED_COLOR = "#9aa0a6"


def _symmetric_radius(values: np.ndarray, center: float, fallback: float = 5.0) -> float:
    """对称轴半径：``max|values - center| × 1.05``，对空数组 / 全 NaN / 退化为 0
    三种边界情况返回 ``fallback``，避免 ``plt.xlim`` 被喂 NaN / 0 导致塌缩。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    m = np.max(np.abs(finite - center))
    if m < 1e-6:
        return fallback
    return float(m) * 1.05


def _axis_bounds(
    values: np.ndarray, center: float, lo: float = 0.9, hi: float = 1.1, fallback: float = 5.0
) -> Tuple[int, int]:
    """非对称轴边界 ``(floor(min*lo), ceil(max*hi))``。先滤掉非有限值再取
    min/max，对空切片 / 全 NaN 退回 ``center ± fallback``——避免
    ``math.floor(np.nanmin(全 NaN))`` 抛 ``ValueError: cannot convert float
    NaN to integer``（与 ``_symmetric_radius`` 同等防御；symmetric 路径已有
    兜底，本函数为 symmetric=False 默认路径补齐）。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.floor(center - fallback), math.ceil(center + fallback)
    return math.floor(finite.min() * lo), math.ceil(finite.max() * hi)


def _subsample_tail(
    df: pd.DataFrame, step: int = 1, tail_length: int | None = None
) -> pd.DataFrame:
    """显示层抽稀：先按 ``step`` 锚定**最新点**向前每 step 行取一个（保证最后
    一行/最新日期总在内），再取最后 ``tail_length`` 个显示点。

    日频 RRG 经 MA(smooth_window) 平滑后相邻日点距极小（实测 ~0.5），全画会
    挤成一团（研报图 2 用日频计算但**按周采样显示**，点距拉开 ~step 倍）。
    ``step<=1`` 时不抽稀（向后兼容）；``tail_length`` 作用于抽稀后的点数。
    """
    if step is not None and step > 1:
        # 反转→每 step 取一→再反转：保证最后一行（最新日期）必定保留
        df = df.iloc[::-1].iloc[::step].iloc[::-1]
    if tail_length is not None and tail_length > 0:
        df = df.iloc[-tail_length:]
    return df


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` → ``rgba(r, g, b, a)``。pyecharts 的 ItemStyleOpts.color 在
    所有 echarts 版本里都稳定接受 rgba 字符串，避免依赖 8-digit hex 的浏览器兼容。"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _rrg_tooltip_formatter(center: Tuple[float, float]) -> JsCode:
    """构造 item-trigger tooltip 的 JS formatter：标签 / 趋势 / 相对强弱 /
    相对强弱动量 / 日期 五行。

    - 趋势在 JS 端按 (x, y) vs ``center`` 现算象限，**与
      ``classify_quadrant`` 同语义**（严格不等号，恰落在中心轴上显示 "—"）。
    - 相对强弱 / 动量显示**中心化偏移** ``value - center``（正=强于基准 /
      负=弱于基准），与研报视角一致。
    - 日期取自 ``LineItem.name``（ISO ``yyyy-mm-ddT...``），``split('T')[0]``
      截到日。
    """
    cx, cy = center
    js = """
function(p){
    var cx = __CX__, cy = __CY__;
    var x = p.value[0], y = p.value[1];
    var rs = (x - cx).toFixed(2), mom = (y - cy).toFixed(2);
    var t;
    if (x > cx && y > cy) t = '领先';
    else if (x < cx && y > cy) t = '改善';
    else if (x < cx && y < cy) t = '滞后';
    else if (x > cx && y < cy) t = '疲软';
    else t = '—';
    var d = String(p.name).split('T')[0];
    return '标签：' + p.seriesName + '<br/>'
         + '趋势：' + t + '<br/>'
         + '相对强弱：' + rs + '<br/>'
         + '相对强弱动量：' + mom + '<br/>'
         + '日期：' + d;
}
""".replace("__CX__", repr(float(cx))).replace("__CY__", repr(float(cy)))
    return JsCode(js)

def _setup_cjk_font() -> None:
    """挑当前系统中**实际已安装**的第一个中文字体，避免中文显示成「口」。

    旧实现硬编码 ``["SimHei"]``：Linux 生产机有 SimHei 正常，但 macOS / 部分 CI
    无 SimHei → matplotlib 回退到无 CJK 字形的字体 → 方块。这里按候选顺序
    （SimHei 仍排第一，保持生产机行为不变）用 ``font_manager`` 探测可用字体，
    macOS / Linux / Windows 通吃。模块 import 时调用一次。
    """
    import matplotlib.font_manager as fm

    candidates = [
        "SimHei", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB",
        "Heiti SC", "Heiti TC", "STHeiti", "Songti SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


_setup_cjk_font()


def create_line_data(df: pd.DataFrame, x_name: str, y_name: str) -> opts.LineItem:
    """
    构建折线data数据

    Args:
        df (pd.DataFrame): 一个二维的DataFrame
        x_name (str): x轴的名称
        y_name (str): y轴的名称

    Returns:
        opts.LineItem: 折线数据
    """
    if df.ndim != 2:
        raise ValueError("df must be a 2D DataFrame")

    last_idx_num: int = df.shape[0] - 1
    line_items: List = []
    df: pd.DataFrame = df[[x_name, y_name]]
    for i, (idx, row) in enumerate(df.iterrows()):

        # NOTE:row[0]为x, row[1]为y

        if i == last_idx_num:
            line_item = opts.LineItem(value=row.tolist(), symbol_size=2, name=idx)
        else:
            line_item = opts.LineItem(
                value=row.tolist(), symbol="emptyCircle", symbol_size=2, name=idx
            )

        line_items.append(line_item)

    return line_items


def create_markline_items(
    line_items: List[opts.LineItem], last_n: int = 2
) -> List[opts.MarkLineItem]:
    """构造轨迹末段标记线（最后 ``last_n`` 个点之间连成的线）。

    末端点显式带 ``symbol="arrow"``，让 echarts 在尾段画出方向箭头——
    与 ``plot_rrg_chart`` 末端大点（``scatter(s=60, zorder=5)``）功能等价，
    但 echart 路径用箭头更能体现轨迹方向，符合研报图 2 / Bloomberg RRG
    既视感。其余端点 ``symbol="none"``，避免重复绘点与折线 marker 冲突。
    """
    items = line_items[-last_n:]
    n = len(items)
    return [
        opts.MarkLineItem(
            coord=item.opts["value"],
            symbol="arrow" if i == n - 1 else "none",
            symbol_size=10 if i == n - 1 else 0,
        )
        for i, item in enumerate(items)
    ]


def create_xyhline_data(center: Tuple[int, int] = None) -> List[opts.MarkLineOpts]:
    """
    创建包含水平线和垂直线的数据列表。

    Args:
        center (Tuple[int, int], optional): 中心点的坐标。默认为None。

    Returns:
        List[opts.MarkLineOpts]: 包含水平线和垂直线的数据列表。
    """
    if center is None:
        # 用于创建markarea的item项
        center: List[int] = [100, 100]

    # 当前 pyecharts 版本 MarkLineItem 不再接受 label_opts 入参（pyecharts
    # 升级带来的 API 变动）；旧代码因 v1 echart 死路径一直未暴露，v2 测试
    # 触发 show_center_line=True 后命中——随 v2 签名变更一并最小修复。
    # symbol="none" 显式禁用 axis-spanning markLine 两端的默认 circle marker
    # （否则 echarts 会在轴边界画两个圆点，与轨迹尾段箭头视觉冲突）。
    # color="#888" 中灰：旧版红色 (color="red") 与 shade_quadrants 色块共存时
    # 视觉过重；灰色既能在白底下作为弱中性参考线，又不抢色块的象限标识职能。
    yhline: opts.MarkLineItem = opts.MarkLineItem(
        y=center[1],
        symbol="none",
        linestyle_opts=opts.LineStyleOpts(type_="solid", color="#888", width=0.5),
    )
    xhline: opts.MarkLineItem = opts.MarkLineItem(
        x=center[0],
        symbol="none",
        linestyle_opts=opts.LineStyleOpts(type_="solid", color="#888", width=0.5),
    )
    return [xhline, yhline]


def create_markline_ops(
    line_items: List[opts.LineItem],
    last_n: int = 2,
    center: Union[List, Tuple] = None,
    set_center_line: bool = False,
) -> List[opts.MarkLineOpts]:
    """
    创建标记线配置。

    Args:
        line_items (List[opts.LineItem]): 标记线项列表。
        last_n (int, optional): 最后N个标记线项。默认为2。
        center (Union[List, Tuple], optional): 标记线的中心点坐标。默认为None。
        set_center_line (bool, optional): 是否设置中心线。默认为False。

    Returns:
        List[opts.MarkLineOpts]: 标记线配置列表。
    """
    # 创建markline的配置
    xyhline: List = []
    if center is None:
        # 用于创建markarea的item项
        center: List[int] = [100, 100]

    if set_center_line:
        xyhline: List[opts.MarkLineItem] = create_xyhline_data(center)

    markline_items: List[opts.MarkLineItem] = create_markline_items(line_items, last_n)

    # 当前 pyecharts MarkLineOpts 合法签名仅含
    # ``is_silent / data / symbol / symbol_size / precision / label_opts / linestyle_opts / animation_opts``，
    # 旧代码的 zlevel + emphasis 均已被上游剥离，随 v2 死路径解封一并清掉。
    return opts.MarkLineOpts(
        data=[markline_items] + xyhline,
        symbol="none",
        is_silent=True,
    )


def create_markarea_items(
    xAxismin, xAxismax, yAxismin, yAxismax, center: Tuple[int, int] = None
) -> List[opts.MarkAreaItem]:
    """构造四象限 markarea，标签置于四个角落 + 各象限专属颜色。

    视觉协议与 ``plot_rrg_chart`` 的 ``shade_quadrants=True`` 路径对齐：
    四色取自模块级 ``_QUADRANT_HEX``（mpl Rectangle alpha=0.08）；标签
    位置 ``insideTopRight / insideTopLeft / insideBottomLeft / insideBottomRight``
    把"领先 / 改善 / 滞后 / 疲软"分别钉到四个角落，与研报图 2 视觉一致。
    """
    if center is None:
        center: List[int] = [100, 100]

    c_lead, c_imp, c_lag, c_weak = _QUADRANT_HEX

    def _label(position: str, color: str) -> opts.LabelOpts:
        # alpha=0.65 让角落标签可读但不喧宾夺主——mpl 路径用 alpha=0.6 + bold
        return opts.LabelOpts(
            position=position,
            color=color,
            font_size=18,
            font_weight="bold",
        )

    def _fill(hex_color: str) -> opts.ItemStyleOpts:
        # alpha=0.08 与 plot_rrg_chart 的 Rectangle alpha=0.08 完全对齐
        return opts.ItemStyleOpts(color=_hex_to_rgba(hex_color, 0.08))

    # (name, x_range, y_range, color, label_position) 四元组按 Q1 → Q4 顺时针
    quadrants = [
        ("领先", [center[0], xAxismax], [center[1], yAxismax], c_lead, "insideTopRight"),
        ("改善", [xAxismin, center[0]], [center[1], yAxismax], c_imp, "insideTopLeft"),
        ("滞后", [xAxismin, center[0]], [center[1], yAxismin], c_lag, "insideBottomLeft"),
        ("疲软", [center[0], xAxismax], [center[1], yAxismin], c_weak, "insideBottomRight"),
    ]
    return [
        opts.MarkAreaItem(
            name=name,
            x=x_range,
            y=y_range,
            itemstyle_opts=_fill(color),
            label_opts=_label(position, color),
        )
        for name, x_range, y_range, color, position in quadrants
    ]


def plot_rrg_echart(
    rs_ratio: pd.DataFrame,
    rs_momentum: pd.DataFrame,
    *,
    # —— 前导块与 plot_rrg_chart 完全一致（同名 / 同序 / 同默认）——
    # 默认即「研报图 2 风格组合」：周采样抽稀 + 短尾迹 + 四象限色块 + 对称轴 + 无网格。
    tail_length: int | None = 10,
    step: int = 5,
    center: Tuple[float, float] = (100, 100),
    symmetric: bool = True,
    shade_quadrants: bool = True,
    show_grid: bool = False,
    # —— 以下为 pyecharts 后端特有 ——
    show_center_line: bool | None = None,
    bg_color: str = "white",
) -> Line:
    """绘制 RRG 散点轨迹的 pyecharts Line 图。

    签名前导块 ``(rs_ratio, rs_momentum, *, tail_length, step, center,
    symmetric, shade_quadrants, show_grid)`` 与 :func:`plot_rrg_chart`
    完全一致（同名 / 同序 / 同默认），两后端可无缝切换；其后为各自后端
    特有参数。接受 ``compute_rrg`` 输出的 ``(rs_ratio, rs_momentum)`` 双
    DataFrame，无需在调用方手动拼装 MultiIndex columns。

    前导块默认值即「研报图 2 风格组合」（``tail_length=10, step=5,
    symmetric=True, shade_quadrants=True, show_grid=False``）：开箱即得周采样
    短尾迹 + 四象限色块 + 对称轴的成品图。要日频全量/无色块裸图显式覆盖即可。

    :param rs_ratio: index=date, columns=industry，``compute_rs_ratio`` 输出。
    :param rs_momentum: 同形态，``compute_rs_momentum`` 输出。
    :param tail_length: 仅画每行业最后 N 个**显示点**的尾迹（默认 ``10``；
        ``None`` 或 ≤0 = 全量）。在 ``step`` 抽稀之后生效——覆盖
        ``tail_length × step`` 个交易日（默认约 10×5=50 个交易日）。
    :param step: 显示点抽稀步长（默认 ``5`` ≈ 周采样）。``1`` = 日频全画；``>1``
        时锚定最新日向前每 step 个交易日取一个显示点。日频经 MA 平滑后相邻点距
        极小（~0.5）会挤成团，``step=5`` 把点距拉开 ~5×，逼近研报图 2。
    :param center: 象限分隔与 markarea 的中心，默认 ``(100, 100)`` 与
        ``classify_quadrant`` 对齐；旧实现用数据动态均值且 ``int()`` 截断,
        造成视觉象限与算法象限不一致（spec §10 B2/B3）。
    :param symmetric: 默认 ``True`` 强制 x/y 轴以 ``center`` 为几何中心对称延展，
        半径取 ``max(|values - center|) × 1.05``，与 ``plot_rrg_chart`` 同
        参数语义一致；``False`` 走 ``floor(min*0.9) / ceil(max*1.1)`` ——
        旧实现 ``floor(min*1.1)`` 把 RS 较低的行业裁出图外（spec §10 B1）。
    :param shade_quadrants: 默认 ``True`` 填充四象限淡色背景 + 角落标签（领先 /
        改善 / 滞后 / 疲软），与 ``plot_rrg_chart`` 同名参数视觉协议一致。
    :param show_grid: 是否显示 x/y 轴 splitLine 网格。默认 ``False`` 贴近研报图 2
        风格（与 ``shade_quadrants=True`` 配合）；``True`` 叠加网格。
    :param show_center_line: 是否在象限中心画灰色十字线。
        ``None``（默认，**推荐**）= 与 ``shade_quadrants`` 互斥智能推导：
        色块存在时不画十字（视觉冗余），无色块时画十字以指示象限分界。
        因 ``shade_quadrants`` 默认 ``True``，故默认不画十字。
        ``True/False`` = 显式覆盖智能默认，例如调试时强制叠加色块+十字。
    :param bg_color: 画布背景色（透传 :class:`InitOpts.bg_color`），默认白。
        pyecharts 默认背景透明在 jupyter 暗主题下会反色，统一固定为白
        与 ``plot_rrg_chart`` 视觉一致。
    """

    # 智能默认：色块和十字都承担"指示象限分界"职能，二者同时启用视觉冗余。
    # 显式传 True/False 走 override 路径。
    if show_center_line is None:
        show_center_line = not shade_quadrants

    rs_ratio, rs_momentum = rs_ratio.align(rs_momentum)
    rs_ratio = _subsample_tail(rs_ratio, step, tail_length)
    rs_momentum = _subsample_tail(rs_momentum, step, tail_length)

    ratio_vals = rs_ratio.values
    mom_vals = rs_momentum.values
    cx, cy = center

    if symmetric:
        r_x = _symmetric_radius(ratio_vals, cx)
        r_y = _symmetric_radius(mom_vals, cy)
        xAxismin = math.floor(cx - r_x)
        xAxismax = math.ceil(cx + r_x)
        yAxismin = math.floor(cy - r_y)
        yAxismax = math.ceil(cy + r_y)
    else:
        # 旧实现 ``floor(min * 1.1)`` 对正值放大有效、对小于中心的值反而压缩区间
        # （min=80 → 88，把低 RS 行业裁出去），统一改为 *0.9 拉宽下界。
        xAxismin, xAxismax = _axis_bounds(ratio_vals, cx)
        yAxismin, yAxismax = _axis_bounds(mom_vals, cy)

    markarea_ops = None
    if shade_quadrants:
        mark_area_items: List[opts.MarkAreaItem] = create_markarea_items(
            xAxismin, xAxismax, yAxismin, yAxismax, center=center
        )
        markarea_ops = opts.MarkAreaOpts(is_silent=True, data=mark_area_items)

    line: Line = Line(init_opts=opts.InitOpts(bg_color=bg_color))
    # markarea 是 series-scoped overlay：把同一份 ops 挂给 N 个 series 会被
    # echarts 渲染为 N 倍叠加（单层 alpha=0.08 × 20 series ≈ 81% 实色浓度），
    # 盖住数据点和标签。修复：用 flag 控制只挂第一个有效 series，让 markarea
    # 等价"chart-scoped 单层背景"——视觉上与 mpl Rectangle alpha=0.08 等价。
    markarea_attached = False
    for industry in rs_ratio.columns:
        df = pd.DataFrame(
            {"RS_Ratio": rs_ratio[industry], "RS_Momentum": rs_momentum[industry]}
        ).dropna()
        if df.empty:
            continue

        line_data: opts.LineItem = create_line_data(df, "RS_Ratio", "RS_Momentum")

        line.add_yaxis(
            series_name=str(industry),
            y_axis=line_data,
            is_smooth=False,
            is_symbol_show=True,
            label_opts=opts.LabelOpts(is_show=False),
            markline_opts=create_markline_ops(
                line_data, center=center, set_center_line=show_center_line
            ),
            emphasis_opts=opts.EmphasisOpts(
                focus="series", blur_scope="coordinateSystem"
            ),
            markarea_opts=markarea_ops if not markarea_attached else None,
        )
        if markarea_ops is not None:
            markarea_attached = True

    # 旧实现 yaxis position='bottom' 与 pyecharts AxisOpts 语义冲突
    # （yaxis 合法值为 left/right），已删除走默认 left（spec §10 B4）
    splitline_opts = opts.SplitLineOpts(is_show=show_grid)
    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value",
            min_=xAxismin,
            max_=xAxismax,
            # is_on_zero=False：echarts value 轴默认把轴线钉在对面轴=0 处；当
            # symmetric=True + 右上离群把 x 区间撑到含 0（负 RS-Ratio）时，y 轴线
            # 会跑进图内部 x=0 处、与左边框的刻度标签分离。固定贴边框，符合
            # Bloomberg/StockCharts RRG 惯例（轴在四周而非穿心）。
            axisline_opts=opts.AxisLineOpts(
                is_on_zero=False,
                linestyle_opts=opts.LineStyleOpts(color="#ddd"),
            ),
            axislabel_opts=opts.LabelOpts(color="#666"),
            splitline_opts=splitline_opts,
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            min_=yAxismin,
            max_=yAxismax,
            axisline_opts=opts.AxisLineOpts(
                is_on_zero=False,
                linestyle_opts=opts.LineStyleOpts(color="#ddd"),
            ),
            axislabel_opts=opts.LabelOpts(color="#666"),
            splitline_opts=splitline_opts,
        ),
        # trigger="item"：hover 单个数据点显示该点的标签/趋势/相对强弱/动量/日期
        # （旧 trigger="axis" 是十字轴显示该 x 位置全部 series，不符研报单点研究语境）。
        # 圆角白底 + 浅蓝边框样式参照研报既视感；border_radius 未被 pyecharts
        # 暴露，用 extra_css_text 注入。
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter=_rrg_tooltip_formatter(center),
            background_color="rgba(255, 255, 255, 0.95)",
            border_color="#7fb2d6",
            border_width=1,
            textstyle_opts=opts.TextStyleOpts(color="#333", font_size=14),
            extra_css_text=(
                "border-radius: 12px; padding: 10px 16px; "
                "box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);"
            ),
        ),
        legend_opts=opts.LegendOpts(is_show=False),
    )
    return line

def plot_rrg_chart(
    rs_ratio: pd.DataFrame,
    rs_momentum: pd.DataFrame,
    *,
    # —— 前导块与 plot_rrg_echart 完全一致（同名 / 同序 / 同默认）——
    # 默认即「研报图 2 风格组合」：周采样抽稀 + 短尾迹 + 四象限色块 + 对称轴 + 无网格。
    tail_length: int | None = 10,
    step: int = 5,
    center: Tuple[float, float] = (100, 100),
    symmetric: bool = True,
    shade_quadrants: bool = True,
    show_grid: bool = False,
    # —— 以下为 matplotlib 后端特有 ——
    marker_size: float = 6,
    marker_face: str | None = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "行业相对轮动图 (RRG)",
    show_lagend: bool = True,
    save_path: str | None = None,
) -> None:
    """绘制 RRG 散点轨迹图（matplotlib 后端）。

    签名前导块 ``(rs_ratio, rs_momentum, *, tail_length, step, center,
    symmetric, shade_quadrants, show_grid)`` 与 :func:`plot_rrg_echart`
    完全一致（同名 / 同序 / 同默认），两后端可无缝切换；其后为各自后端特有参数。
    前导块默认值即「研报图 2 风格组合」（``tail_length=10, step=5,
    symmetric=True, shade_quadrants=True, show_grid=False``）；mpl 路径再叠
    ``marker_size=3, marker_face="none"`` 更贴 stockcharts。

    :param tail_length: 仅画每行业最后 N 个**显示点**的尾迹（默认 ``10``；
        ``None`` = 全部）。在 ``step`` 抽稀之后生效——覆盖 ``tail_length × step``
        个交易日。
    :param step: 显示点抽稀步长（默认 ``5`` ≈ 周采样）。``1`` = 日频全画；``>1`` 时
        锚定最新日向前每 step 个交易日取一个显示点。MA 平滑后日频相邻位移极小
        （~0.5）易视觉粘连，``step=5`` 把点距拉开 ~5× 逼近研报图 2 的 stockcharts 风格。
    :param center: 参考十字线的 (x, y) 中心，默认 (100, 100)——JdK 比率法的标准中枢。
        旧实现硬编码为 (0, 0)，会触发 matplotlib autoscale 把 0 纳入数据范围，
        让所有 RS 数据被挤到右上角；现在默认对齐研报与 stockcharts。
    :param symmetric: 默认 ``True`` 强制 x/y 轴以 ``center`` 为几何中心对称延展，半径取
        ``max(|data - center|) × 1.05``。避免单边长尾（如个别行业 RS-Ratio 冲到 145
        而下方最低只 78）把 center 推到画布角落。``False`` 保持 matplotlib autoscale。
    :param shade_quadrants: 默认 ``True`` 给四象限填充淡色背景并在四角标注名称
        （领先绿 / 改善蓝灰 / 滞后红橙 / 疲软紫——研报 §1 用语），与
        ``plot_rrg_echart`` 的 ``create_markarea_items`` 配色一致，
        方便直观区分行业所处轮动阶段。
    :param show_grid: 是否显示 matplotlib 默认网格线。默认 ``False`` 贴近研报图 2 /
        stockcharts 风格（与 ``shade_quadrants=True`` 配合）；``True`` 叠加网格。
    :param marker_size: matplotlib markersize（默认 6，stockcharts 风格建议 3）。
    :param marker_face: marker 填充色；传 ``"none"`` 变空心圆（推荐高密度尾迹场景）。
        None 表示沿用线颜色（实心）。
    """

    plt.figure(figsize=figsize)
    rs_ratio, rs_momentum = rs_ratio.align(rs_momentum)
    # 先 step 抽稀再取 tail（与 plot_rrg_echart 同语义）；tail_length=0/负数
    # 与 step<=1 均安全跳过，见 _subsample_tail。
    rs_ratio = _subsample_tail(rs_ratio, step, tail_length)
    rs_momentum = _subsample_tail(rs_momentum, step, tail_length)
    # 创建一个颜色循环
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(colors)

    for industry in rs_ratio.columns:
        color = next(color_cycle)
        plt.plot(
            rs_ratio[industry],
            rs_momentum[industry],
            marker="o",
            color=color,
            markersize=marker_size,
            markerfacecolor=marker_face if marker_face is not None else color,
        )

        last_rs_ratio = rs_ratio[industry].iloc[-1]
        last_rs_momentum = rs_momentum[industry].iloc[-1]

        plt.scatter(
            last_rs_ratio,
            last_rs_momentum,
            color=color,
            s=60,
            zorder=5,
        )

        if show_lagend:
            # 为最后一个点添加行业标签
            plt.annotate(
                industry,
                (last_rs_ratio, last_rs_momentum),
                xytext=(5, 5),
                textcoords="offset points",
                # arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            )

    # 中心十字线：与 echart 后端 show_center_line 智能默认对齐——shade_quadrants=True
    # 时四象限色块边界已标出中心、十字冗余故不画；否则画灰色（#888，与 echart
    # create_xyhline_data 同色）十字指示象限分界。旧实现恒画红线，与色块共存时过重。
    if not shade_quadrants:
        plt.axhline(center[1], color="#888", lw=0.5)
        plt.axvline(center[0], color="#888", lw=0.5)
    plt.xlabel("JdK RS-Ratio")
    plt.ylabel("JdK RS-Momentum")
    plt.title(title)
    plt.grid(show_grid)

    if symmetric:
        r_x = _symmetric_radius(rs_ratio.values, center[0])
        r_y = _symmetric_radius(rs_momentum.values, center[1])
        plt.xlim(center[0] - r_x, center[0] + r_x)
        plt.ylim(center[1] - r_y, center[1] + r_y)

    if shade_quadrants:
        ax = plt.gca()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx, cy = center
        # 命名对齐西部研报 §1 line 253-259（领先/改善/滞后/疲软）；
        # 颜色取自模块级 _QUADRANT_HEX（与 create_markarea_items rgba 同 RGB）
        c_lead, c_imp, c_lag, c_weak = _QUADRANT_HEX
        quadrants = [
            # (x, y, w, h, color, label, anchor_x, anchor_y, ha, va)
            (cx, cy, x1 - cx, y1 - cy, c_lead, "领先", 0.98, 0.98, "right", "top"),
            (x0, cy, cx - x0, y1 - cy, c_imp, "改善", 0.02, 0.98, "left", "top"),
            (x0, y0, cx - x0, cy - y0, c_lag, "滞后", 0.02, 0.02, "left", "bottom"),
            (cx, y0, x1 - cx, cy - y0, c_weak, "疲软", 0.98, 0.02, "right", "bottom"),
        ]
        for x, y, w, h, color, label, ax_x, ax_y, ha, va in quadrants:
            ax.add_patch(Rectangle((x, y), w, h, color=color, alpha=0.08, zorder=0))
            ax.text(
                ax_x, ax_y, label,
                transform=ax.transAxes,
                ha=ha, va=va,
                fontsize=14, color=color, alpha=0.6, weight="bold",
                zorder=10,
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def render_in_notebook(
    chart,
    *,
    width: str = "100%",
    height: str = "760px",
    online_host: str = _ECHARTS_CDN,
):
    """在 VS Code / JupyterLab / 经典 Notebook 中**稳健**渲染 pyecharts 图。

    一次解两个"图空白"的根因（均与操作系统无关）：

    1. **RequireJS 缺失**：``chart.render_notebook()`` 默认用 ``require(['echarts'])``
       加载 echarts；VS Code 的 notebook 渲染器无全局 ``require`` → 静默 no-op → 空白
       （cell 显示 0.0s ✓ 却一片空白，正是此症状）。本助手改用 ``render_embed()``
       （普通 ``<script src>``）base64 塞进 ``data:`` iframe——iframe 是独立浏览上下文，
       外部 script 正常加载、绕开 VS Code 对内联 ``<script>`` 的沙箱限制。

    2. **默认 CDN 挂掉**：pyecharts 默认 ``assets.pyecharts.org`` 常证书过期/不可达，
       echarts 库本身取不到 → 即便 iframe 没问题图仍空白。本助手把 ``chart.js_host``
       改到稳定 CDN（默认 jsdelivr）后再 ``render_embed``。

    对**任意** pyecharts 图实例通用（``HeatMap`` / ``Line`` / ...）：

    >>> chart = plot_signal_heatmap(score_panel, code2name=gen.code2name)
    >>> render_in_notebook(chart)          # 替代 chart.render_notebook()

    :param chart: 任意 pyecharts 图实例。
    :param width: iframe 宽（CSS，默认 ``"100%"`` 自适应 cell 宽）。
    :param height: iframe 高（CSS，默认 ``"760px"`` 略大于默认画布 720px 以容下
        底部 visualMap 图例）。
    :param online_host: echarts 资源 CDN（末尾须带 ``/``）。默认 jsdelivr；China
        网络若被限流可换 ``"https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/"``。
        传 ``None`` 则保留 chart 原 ``js_host``（不改 CDN）。
    :returns: ``IPython.display.IFrame``——notebook 中作为 cell 末表达式即渲染。
    :note: 仍需联网拉 echarts.min.js；完全离线请 ``chart.render("x.html")`` 落地后
        浏览器打开（或把 ``online_host`` 指向本地已下好的 echarts 资源目录）。
    """
    import base64

    from IPython.display import IFrame

    if online_host:
        # 换掉挂掉的默认 CDN（render_embed 的 <script src> 取自 chart.js_host）
        chart.js_host = online_host
    embed = chart.render_embed()
    b64 = base64.b64encode(embed.encode("utf-8")).decode("ascii")
    # 用 IPython 钦定的 IFrame（而非 HTML(<iframe>)，后者会触发 UserWarning）
    return IFrame(src=f"data:text/html;base64,{b64}", width=width, height=height)


def _js_single_quoted_array(items: List[str]) -> str:
    """把字符串列表渲染成**单引号** JS 数组字面量 ``['a', 'b']``。

    **不能用 ``json.dumps``（双引号）**：pyecharts 渲染 JsCode 时会把选项整体
    JSON 序列化，JsCode 内的双引号会被转义成 ``\\"`` → ``var xs = [\\"a\\"]`` 是
    非法 JS（"Invalid or unexpected token"），整段脚本中止、echarts 不初始化、
    图空白。单引号不被 JSON 转义，故安全（与 :func:`_rrg_tooltip_formatter`
    全程单引号同理）。
    """
    esc = [s.replace("\\", "\\\\").replace("'", "\\'") for s in items]
    return "[" + ", ".join(f"'{s}'" for s in esc) + "]"


def _signal_tooltip_formatter(x_dates: List[str], y_names: List[str]) -> JsCode:
    """构造持仓热力图 tooltip 的 JS formatter：行业 / 日期 / 持仓状态 三行。

    heatmap 的数据项是 ``[xIdx, yIdx, value]`` 数值索引，``p.name`` 对 heatmap
    的语义在不同 echarts 版本不稳定；故把日期 / 行业名两数组以**单引号** JS 字面量
    注入（见 :func:`_js_single_quoted_array` 为何不用 JSON 双引号），用
    ``p.value[0/1]`` 索引回标签——不依赖 ``p.name``，跨版本稳。
    """
    js = """
function(p){
    var xs = __XS__, ys = __YS__;
    var v = p.value;
    var d = xs[v[0]], ind = ys[v[1]];
    var held = v[2] > 0.5 ? '持仓' : '空仓';
    return '行业：' + ind + '<br/>日期：' + d + '<br/>状态：' + held;
}
""".replace("__XS__", _js_single_quoted_array(x_dates)).replace(
        "__YS__", _js_single_quoted_array(y_names)
    )
    return JsCode(js)


def plot_signal_heatmap(
    score_panel: pd.DataFrame,
    *,
    code2name: dict[str, str] | None = None,
    title: str = "行业轮动持仓信号",
    held_color: str = "#c23531",
    empty_color: str = "#eeeeee",
    date_fmt: str = "%Y-%m-%d",
    width: str = "1280px",
    height: str = "720px",
    bg_color: str = "white",
) -> HeatMap:
    """把 0/1 持仓信号面板渲染成「行业 × 日期」热力图（pyecharts HeatMap）。

    与本模块其它绘图函数同构：**纯渲染、零 I/O**——``code2name`` 由调用方传入
    （如 ``RRGSignalGenerator.code2name`` 或 ``fetch_rrg_dataset(...)[2]``），
    本函数不取数。0 格灰、1 格红，hover 显示行业 / 日期 / 持仓状态。

    :param score_panel: ``select_diffusion_with_rrg`` 等出的信号面板，
        index=date，columns=申万一级行业**代码**（如 ``801080.SI``），
        value ∈ {0, 1}（``NaN`` 当 0 处理）。
    :param code2name: 行业代码 → 中文简称映射；``None`` 或缺某列时该列 y 轴标签
        回退用原代码。
    :param title: 图标题。
    :param held_color: 持仓（value=1）格的颜色，默认研报红 ``#c23531``。
    :param empty_color: 空仓（value=0）格的颜色，默认浅灰 ``#eeeeee``。
    :param date_fmt: x 轴日期 ``strftime`` 格式。
    :param width: 画布宽（透传 ``InitOpts.width``）。
    :param height: 画布高。
    :param bg_color: 画布背景色，默认白（与本模块其它图一致，防 jupyter 暗主题反色）。
    :returns: pyecharts ``HeatMap``；notebook 中 ``.render_notebook()`` 渲染，
        或 ``.render("xxx.html")`` 存盘。
    """
    code2name = code2name or {}
    panel = score_panel.fillna(0)

    x_dates = [pd.Timestamp(d).strftime(date_fmt) for d in panel.index]
    y_names = [code2name.get(str(c), str(c)) for c in panel.columns]

    # echarts heatmap 数据项 = [xIdx, yIdx, value]，遍历 (日期 i, 行业 j) 全网格
    arr = panel.to_numpy()
    data = [
        [i, j, int(round(arr[i, j]))]
        for i in range(arr.shape[0])
        for j in range(arr.shape[1])
    ]

    return (
        HeatMap(init_opts=opts.InitOpts(width=width, height=height, bg_color=bg_color))
        .add_xaxis(x_dates)
        .add_yaxis(
            series_name="持仓信号",
            yaxis_data=y_names,
            value=data,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
                axislabel_opts=opts.LabelOpts(rotate=90, font_size=9, color="#666"),
            ),
            yaxis_opts=opts.AxisOpts(
                type_="category",
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
                axislabel_opts=opts.LabelOpts(font_size=11, color="#333"),
            ),
            # piecewise：0/1 离散两段直接映射灰/红，比连续 visualMap 更贴二元语义
            visualmap_opts=opts.VisualMapOpts(
                is_piecewise=True,
                pieces=[
                    {"value": 1, "label": "持仓", "color": held_color},
                    {"value": 0, "label": "空仓", "color": empty_color},
                ],
                orient="horizontal",
                pos_left="center",
                pos_top="0",
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter=_signal_tooltip_formatter(x_dates, y_names),
                background_color="rgba(255, 255, 255, 0.95)",
                border_color="#7fb2d6",
                border_width=1,
                textstyle_opts=opts.TextStyleOpts(color="#333", font_size=13),
            ),
            # 隐藏 series 默认图例（蓝块"持仓信号"）——颜色与红/灰格不符且冗余，
            # 真正的图例是底部 visualMap 的"空仓/持仓"。
            legend_opts=opts.LegendOpts(is_show=False),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
        )
    )


def _etf_representative_industry(
    etf_industry: Union[pd.Series, pd.DataFrame],
) -> dict:
    """ETF → 代表行业 sw_code 单查找，统一静态 / 动态归类（决定树枝归属）。

    - 静态 ``pd.Series``（``index=etf_code, value=sw_code``）：直取。
    - 动态 long ``pd.DataFrame``（含 ``etf_code / sw_code``）：取每 etf 的
      ``sw_code`` **众数**（平局取字典序最小，确定性）。

    :returns: ``dict[etf_code(str) -> sw_code(str)]``；缺失的 ETF 不入 dict。
    """
    if isinstance(etf_industry, pd.Series):
        return {str(k): str(v) for k, v in etf_industry.dropna().items()}

    df = etf_industry[["etf_code", "sw_code"]].dropna()
    rep: dict[str, str] = {}
    for etf, grp in df.groupby("etf_code"):
        counts = grp["sw_code"].value_counts()
        top = counts[counts == counts.max()].index
        rep[str(etf)] = str(sorted(top)[0])
    return rep


def plot_etf_holding_heatmap(
    weight_panel: pd.DataFrame,
    etf_industry: Union[pd.Series, pd.DataFrame],
    *,
    etf_name: dict[str, str] | None = None,
    industry_name: dict[str, str] | None = None,
    held_color: str = "#c23531",
    empty_color: str = "#eeeeee",
    held_threshold: float = 1e-6,
    figsize: Tuple[float, float] | None = None,
    title: str = "ETF 行业轮动持仓",
    date_fmt: str = "%Y-%m-%d",
    root_label: str = "全市场",
):
    """把 ETF 权重面板 ``w`` 画成「行业树（y 轴）+ 时间持仓热力图」(matplotlib)。

    clustermap 式双面板：**左 `ax_tree`** 手绘 ``root → 行业 → ETF`` 两级语义树
    （行业树枝 + ETF 叶子，替代图例——ETF 多时靠树枝分组导航）；**右 `ax_heat`**
    画 ETF×时间二元持仓热力图（红=持仓 / 灰=空仓），两面板逐行对齐。

    数据入参与同模块其它图同构、零 I/O：名称由调用方注入，``etf_industry``
    仅用于定每只 ETF 的**代表行业**（树枝归属）。

    :param weight_panel: ``batch_optimize→mask_listed`` 出的权重面板，
        ``index=换仓日 × columns=ts_code``。**持仓 = ``value > held_threshold``**，
        ``NaN`` 当空仓。
    :param etf_industry: 二态——静态 ``pd.Series``（``etf_code→sw_code``）或动态
        long ``pd.DataFrame``（含 ``etf_code / sw_code``，取众数定代表行业）。
    :param etf_name: ``ts_code → 简称``（y 轴叶子标签）；缺则回退代码。
    :param industry_name: ``sw_code → 中文名``（树枝标签）；缺则回退代码。
    :param held_color: 持仓格颜色（默认研报红 ``#c23531``）。
    :param empty_color: 空仓格颜色（默认浅灰 ``#eeeeee``）。
    :param held_threshold: 持仓阈值（默认 ``1e-6``，对齐 backtest）。
    :param figsize: ``None`` 时按 ``(0.55·ncols+3.5, 0.32·nrows+1.5)`` 自适应——
        ETF 多则图自动变高。
    :param title: 图标题（置于热力图上方）。
    :param date_fmt: 换仓日 ``strftime`` 格式（x 轴）。
    :param root_label: 树根标签（默认「全市场」）。
    :returns: matplotlib ``Figure``；notebook 中末行直接显示，或 ``fig.savefig(...)``。
    """
    etf_name = etf_name or {}
    industry_name = industry_name or {}
    rep_map = _etf_representative_industry(etf_industry)

    # 列名统一 str 化（与 rep_map / sorted_etfs 键一致）：避免用 str 键回查原始
    # 非 str 列导致 KeyError；后续数据访问一律走 wp（已 str 化列），不再碰原 frame。
    wp = weight_panel.copy()
    wp.columns = [str(c) for c in wp.columns]
    columns = list(wp.columns)
    dates = list(wp.index)
    _UNCLASSIFIED = "__UNCLASSIFIED__"

    def _rep(etf: str) -> str:
        return rep_map.get(etf, _UNCLASSIFIED)

    def _ind_sort_key(sw: str) -> tuple:
        # 未分类排末；其余按行业名（无则 code）确定性排序
        if sw == _UNCLASSIFIED:
            return (1, "")
        return (0, industry_name.get(sw, sw))

    sorted_etfs = sorted(columns, key=lambda e: (_ind_sort_key(_rep(e)), e))
    n, ncols = len(sorted_etfs), len(dates)

    # 二元持仓矩阵（held=1 / 空=0），向量化替代逐元素 .loc 双循环：
    # wp[sorted_etfs] 为 date×etf，``> held_threshold`` 对 NaN 恒 False（自动空仓），
    # 转置成 etf×date。一次 numpy 比较，免去 O(n·m) 标签查找与重复日/非 str 列崩溃。
    if n and ncols:
        M = (wp[sorted_etfs] > held_threshold).to_numpy().T.astype(float)
    else:
        M = np.zeros((n, ncols))

    # 行业 → 行索引（保持 sorted_etfs 出现顺序，dict 即有序）
    ind_rows: dict[str, list] = {}
    for i, etf in enumerate(sorted_etfs):
        ind_rows.setdefault(_rep(etf), []).append(i)

    if figsize is None:
        figsize = (max(8.0, 0.55 * ncols + 3.5), max(4.0, 0.32 * n + 1.5))
    # constrained_layout 按「热力图 y 刻度标签(=ETF 名)的实际像素宽度」自动预留
    # 标签列：树紧贴标签左、热力图紧贴标签右，中间只剩标签本身宽度——空白不再由
    # 手猜的 width_ratios/wspace 产生（那只会把空白搬家，不会消失）。
    fig, (ax_tree, ax_heat) = plt.subplots(
        1, 2, figsize=figsize, layout="constrained",
        gridspec_kw={"width_ratios": [1, 6]},
    )
    _engine = fig.get_layout_engine()
    if _engine is not None:
        _engine.set(w_pad=0.01, wspace=0.0)

    # 空面板（无 ETF / 无换仓日）：imshow 0 维数组 + 等值轴界会抛奇异轴告警，
    # 直接画一张「无持仓数据」占位图返回（回测窗口可能某段无任何持仓）。
    if n == 0 or ncols == 0:
        for ax in (ax_tree, ax_heat):
            ax.axis("off")
        ax_heat.text(0.5, 0.5, "无持仓数据", ha="center", va="center",
                     transform=ax_heat.transAxes, fontsize=12, color="#888")
        ax_heat.set_title(title)
        return fig

    # —— 右：二元持仓热力图 ——
    cmap = ListedColormap([empty_color, held_color])
    ax_heat.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="upper",
                   extent=(-0.5, ncols - 0.5, n - 0.5, -0.5))
    ax_heat.set_xticks(range(ncols))
    ax_heat.set_xticklabels(
        [pd.Timestamp(d).strftime(date_fmt) for d in dates], rotation=90, fontsize=8)
    # ETF 名作热力图 y 刻度标签——matplotlib 自动按其像素宽度预留标签列，
    # constrained_layout 据此把树/标签/热力图紧凑排开，杜绝可搬运的多余空白。
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels([etf_name.get(e, e) for e in sorted_etfs], fontsize=8)
    ax_heat.tick_params(left=False)  # 去刻度短线，仅留标签
    for sp in ax_heat.spines.values():
        sp.set_visible(False)
    ax_heat.set_title(title)

    # —— 左：root → 行业 → ETF 正交语义树（仅树线/节点；ETF 名由热力图 y 刻度承担）——
    # 叶子圆点落在树面板右边沿(leaf_x=1)，紧邻右侧自动预留的 ETF 标签列，按行对齐；
    # 行业(一级)标签骑在 root→行业 横线上方（居中），gid="industry" 供测试区分。
    root_x, ind_x, leaf_x = 0.0, 0.32, 1.0
    line_color = "#888"
    ax_tree.set_xlim(-0.15, 1.02)
    ax_tree.set_ylim(n - 0.5, -0.5)  # 与热力图同向（行 0 在顶），逐行对齐
    ax_tree.axis("off")

    ind_ys: list[float] = []
    for sw, rows in ind_rows.items():
        y_ind = sum(rows) / len(rows)
        ind_ys.append(y_ind)
        node_color = _UNCLASSIFIED_COLOR if sw == _UNCLASSIFIED else "#555"
        # 行业 → 各叶子：ind_x 竖线 + 每叶横线 + 叶子空心点（点在右边沿贴标签列）
        ax_tree.plot([ind_x, ind_x], [min(rows), max(rows)], color=line_color, lw=1)
        for r in rows:
            ax_tree.plot([ind_x, leaf_x], [r, r], color=line_color, lw=1)
            ax_tree.plot([leaf_x], [r], marker="o", mfc="white",
                         mec=line_color, ms=4)
        # root → 行业 横线 + 行业实心节点 + 行业名（骑在横线上方居中）
        ax_tree.plot([root_x, ind_x], [y_ind, y_ind], color=line_color, lw=1)
        ax_tree.plot([ind_x], [y_ind], marker="o", color=node_color, ms=6)
        ind_lbl = "未分类" if sw == _UNCLASSIFIED else industry_name.get(sw, sw)
        ax_tree.text((root_x + ind_x) / 2, y_ind - 0.42, ind_lbl, ha="center",
                     va="center", fontsize=9, color=node_color, gid="industry")

    # root 竖干 + 节点 + 标签（左侧）
    if ind_ys:
        ax_tree.plot([root_x, root_x], [min(ind_ys), max(ind_ys)],
                     color=line_color, lw=1)
        y_root = sum(ind_ys) / len(ind_ys)
        ax_tree.plot([root_x], [y_root], marker="o", color="#333", ms=6)
        ax_tree.text(root_x - 0.04, y_root, root_label, ha="right", va="center",
                     fontsize=9, color="#333", gid="root")

    # 不用 tight_layout（imshow 固定/auto aspect 与之不兼容会告警）；gridspec
    # 已定 width_ratios + wspace，存盘时用 savefig(bbox_inches="tight") 收边即可。
    return fig


def plot_industry_holding_panel(
    score_panel: pd.DataFrame,
    reps: pd.DataFrame,
    weight_panel: pd.DataFrame,
    *,
    industry_name: dict[str, str] | None = None,
    etf_name: dict[str, str] | None = None,
    dates=None,
    annotate: str | None = "both",
    held_threshold: float = 1e-6,
    cmap: str = "Reds",
    empty_color: str = "#eeeeee",
    selected_edge_color: str = "#c23531",
    figsize: Tuple[float, float] | None = None,
    title: str = "行业信号 → 代表 ETF 持仓（格 = 权重）",
    date_fmt: str = "%Y-%m-%d",
    name_maxlen: int = 4,
):
    """「行业 × 换仓日」单面板：格 = 该期该行业代表 ETF 拿到的 LP 权重（matplotlib）。

    把 ``plot_signal_heatmap``（行业被选中与否）与 ``plot_etf_holding_heatmap``
    （ETF 持仓）折成一张图：行数只随行业走（≤ 一级行业数），不随 ETF 数膨胀。
    沿一列往下读 = 这期哪些行业被选中、每个行业下哪只 ETF 拿到多少权重。

    三种格：**实心色阶** = 有权重（深 = 权重大）；**空心框** = 信号选中但没拿到
    权重（无合格代表 ETF，或 LP 没分配）；**浅灰** = 未选中。

    纯渲染、零 I/O，与本模块其它图同构；名称由调用方注入。

    :param score_panel: 信号面板，``index=换仓日 × columns=行业代码``，value ∈ {0,1}
        （``NaN`` 当 0）——即 ``mask.loc[月末].astype(float)``。
    :param reps: ``build_dynamic_representative_panel`` 出的代表面板，
        ``index=换仓日 × columns=行业代码, value=ts_code 或 NaN``。
    :param weight_panel: ``batch_optimize→mask_listed`` 出的权重面板，
        ``index=换仓日 × columns=ts_code``；``NaN`` 当 0。
    :param industry_name: 行业代码 → 中文名（y 轴）；缺则回退代码。
    :param etf_name: ``ts_code → 简称``（格内标注）；缺则回退代码。
    :param dates: 只画这些换仓日（任意可迭代）；``None`` = 三表共有的全部换仓日。
        45 列全窗按公式为 25.5 英寸、被 ``figsize`` 默认值封顶到 24 英寸（格子相应变窄），
        分析时常切最近 12–24 期。
    :param annotate: 格内文字——``"name"`` / ``"weight"`` / ``"both"`` / ``None``。
    :param held_threshold: 权重 > 此值才算持有（默认 ``1e-6``，对齐 backtest）。
    :param cmap: 权重色阶（默认 ``"Reds"``）。
    :param empty_color: 未选中格颜色。
    :param selected_edge_color: 空心框颜色。
    :param figsize: ``None`` 时 ``(min(0.5·ncols+3, 24), 0.35·nrows+1.5)``。
    :param title: 图标题。
    :param date_fmt: x 轴日期格式。
    :param name_maxlen: ETF **简称**截断长度（格子窄）；缺简称回退为去交易所后缀的 6 位码，不截断。
    :returns: matplotlib ``Figure``（单轴 ``fig.axes[0]``）。
    """
    _setup_cjk_font()
    industry_name = industry_name or {}
    etf_name = etf_name or {}
    if annotate not in ("name", "weight", "both", None):
        raise ValueError(f"annotate 只接受 name/weight/both/None，收到 {annotate!r}")

    score = score_panel.fillna(0.0)
    weights = weight_panel.fillna(0.0)
    idx = score.index.intersection(weights.index).intersection(reps.index)
    if dates is not None:
        idx = idx.intersection(pd.DatetimeIndex(pd.to_datetime(list(dates))))
    idx = idx.sort_values()

    sel = score.reindex(index=idx).fillna(0.0) > 0
    industries = sorted(str(c) for c in sel.columns[sel.any(axis=0)])

    # 格值：w[date, reps[date, ind]]；无代表 / 代表不在权重面板 → 0
    W = pd.DataFrame(0.0, index=idx, columns=industries)
    rep_used = pd.DataFrame(np.nan, index=idx, columns=industries, dtype=object)
    for ind in industries:
        if ind not in reps.columns:
            continue
        r = reps.reindex(index=idx)[ind]
        for d, etf in r.items():
            if isinstance(etf, str) and etf in weights.columns:
                W.at[d, ind] = float(weights.at[d, etf])
                rep_used.at[d, ind] = etf
    W = W.T                      # 行 = 行业，列 = 换仓日
    S = sel.reindex(columns=industries).fillna(False).T
    held = W > held_threshold

    nrows, ncols = W.shape
    if figsize is None:
        figsize = (min(0.5 * max(ncols, 1) + 3, 24), 0.35 * max(nrows, 1) + 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    if nrows == 0 or ncols == 0:
        ax.set_title(title + "（窗口内无行业被选中）")
        ax.set_axis_off()
        return fig

    vmax = float(W.to_numpy().max())
    vmax = vmax if vmax > held_threshold else 1.0
    data = np.ma.masked_where(~held.to_numpy(), W.to_numpy())
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(empty_color)
    ax.imshow(data, cmap=cm, vmin=0.0, vmax=vmax, aspect="auto", interpolation="nearest")

    # 选中但无权重 → 空心框
    for i in range(nrows):
        for j in range(ncols):
            if S.iat[i, j] and not held.iat[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=selected_edge_color, linewidth=1.2))

    # 格内标注
    if annotate is not None:
        for i, ind in enumerate(W.index):
            for j, d in enumerate(W.columns):
                if not held.iat[i, j]:
                    continue
                etf = rep_used.at[d, ind]
                if isinstance(etf, str) and etf in etf_name:
                    nm = etf_name[etf][:name_maxlen]          # 有简称：截断到 name_maxlen
                elif isinstance(etf, str):
                    nm = etf.split(".")[0]                    # 无简称：去后缀的 6 位码，不截断
                else:
                    nm = ""
                wv = f"{W.iat[i, j]:.2f}"
                txt = {"name": nm, "weight": wv, "both": f"{nm}\n{wv}"}[annotate]
                dark = W.iat[i, j] > 0.6 * vmax
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")

    ax.set_yticks(range(nrows))
    ax.set_yticklabels([industry_name.get(c, c) for c in W.index], fontsize=9)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([pd.Timestamp(d).strftime(date_fmt) for d in W.columns],
                       rotation=90, fontsize=8)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title)
    fig.tight_layout()
    return fig

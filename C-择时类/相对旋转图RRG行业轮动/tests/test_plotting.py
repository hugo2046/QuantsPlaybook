"""plot_rrg_echart v2 契约测试。

不依赖 compute_rrg / qlib，使用直接构造的 ``rs_ratio`` / ``rs_momentum``
合成 DataFrame 验证签名、固定中心、align 行为、tail_length 守卫。

断言策略：对结构层（series 数、轴边界）用 ``chart.options``；涉及
``opts.*`` 嵌套对象（如 MarkAreaOpts.data）的，用
``chart.dump_options_with_quotes()`` 拿到 JSON 字符串后 ``json.loads``
反序列化——pyecharts 只在 dump 时把 opts 对象展平成 dict。
注意必须用 ``_with_quotes`` 变体：tooltip 的 JsCode formatter 在普通
``dump_options()`` 里被序列化成占位符，破坏 JSON 合法性。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pyecharts.charts import HeatMap, Line

from src.plotting import (
    plot_rrg_chart,
    plot_rrg_echart,
    plot_signal_heatmap,
    render_in_notebook,
)


def _make_rs_frames(
    n_days: int = 30,
    n_industries: int = 3,
    mean_offset: float = 0.0,
    seed: int = 20260528,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造合成 rs_ratio / rs_momentum，均值围绕 ``100 + mean_offset`` 抖动。"""
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    cols = [f"IND{i + 1}" for i in range(n_industries)]
    rs_ratio = pd.DataFrame(
        100.0 + mean_offset + rng.normal(scale=2.0, size=(n_days, n_industries)),
        index=dates,
        columns=cols,
    )
    rs_momentum = pd.DataFrame(
        100.0 + mean_offset + rng.normal(scale=1.5, size=(n_days, n_industries)),
        index=dates,
        columns=cols,
    )
    return rs_ratio, rs_momentum


def test_plot_rrg_echart_smoke():
    """烟雾测试：返回 Line 实例，xAxis/yAxis 中心可见，每行业一条 series。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum)

    assert isinstance(chart, Line)
    options = chart.options
    x_axis = options["xAxis"][0]
    y_axis = options["yAxis"][0]
    # 默认 center=(100,100) 应落在轴区间内
    assert x_axis["min"] <= 100 <= x_axis["max"]
    assert y_axis["min"] <= 100 <= y_axis["max"]
    # 每个行业对应一条 series
    assert len(options["series"]) == rs_ratio.shape[1]


def test_plot_rrg_echart_center_fixed_at_100():
    """B2/B3 反向验证：即便数据均值大幅偏离 100，markarea 中心仍固定。"""
    rs_ratio, rs_momentum = _make_rs_frames(mean_offset=30.0)  # 数据中心 ~130
    chart = plot_rrg_echart(
        rs_ratio, rs_momentum, shade_quadrants=True, show_center_line=True
    )

    # markarea 在 chart.options 里还是 MarkAreaOpts 对象，序列化后才是 dict；
    # 字段为 xAxis/yAxis（pyecharts 内部把 x→xAxis 转写），且中心点可能落在
    # MarkAreaItem 的 [0] 或 [1] 端。
    dumped = json.loads(chart.dump_options_with_quotes())
    all_corners: list[tuple] = []
    for s in dumped["series"]:
        mark_area = s.get("markArea")
        if not mark_area:
            continue
        for item in mark_area.get("data", []):
            for corner in item:
                all_corners.append((corner.get("xAxis"), corner.get("yAxis")))

    # 旧实现用 int(mean(rs_ratio)) ≈ 130 作中心；v2 固定 (100, 100)
    assert all_corners, "shade_quadrants=True 应至少产生一组 markarea"
    assert (100, 100) in all_corners, (
        f"v2 应固定中心 (100,100)，实际 markarea 角点：{all_corners}"
    )
    # 反向断言：数据均值 ~130 不应作为中心出现
    assert (130, 130) not in all_corners


def test_plot_rrg_echart_align_handles_misaligned_inputs():
    """rs_ratio 与 rs_momentum 行/列不一致时，align 应不抛、按交集绘制。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    # 错开 rs_momentum 的前 5 行 + 多一列
    rs_momentum_misaligned = rs_momentum.iloc[5:].copy()
    rs_momentum_misaligned["IND_EXTRA"] = 100.0

    chart = plot_rrg_echart(rs_ratio, rs_momentum_misaligned)
    assert isinstance(chart, Line)


def test_plot_rrg_echart_tail_length_zero_equals_none():
    """tail_length=0 / 负数应等价 None（全量），与 plot_rrg_chart 同名守卫一致。"""
    rs_ratio, rs_momentum = _make_rs_frames(n_days=50)

    chart_none = plot_rrg_echart(rs_ratio, rs_momentum)
    chart_zero = plot_rrg_echart(rs_ratio, rs_momentum, tail_length=0)
    chart_neg = plot_rrg_echart(rs_ratio, rs_momentum, tail_length=-5)

    # series 数据点数应一致（全量）
    def _point_count(chart: Line) -> int:
        return sum(len(s["data"]) for s in chart.options["series"])

    assert _point_count(chart_zero) == _point_count(chart_none)
    assert _point_count(chart_neg) == _point_count(chart_none)


def test_plot_rrg_echart_symmetric_axes():
    """symmetric=True 时 x/y 轴应以 center 为几何中心对称。"""
    rs_ratio, rs_momentum = _make_rs_frames(mean_offset=10.0)  # 数据中心 ~110，偏右上
    chart = plot_rrg_echart(rs_ratio, rs_momentum, symmetric=True)

    x_axis = chart.options["xAxis"][0]
    y_axis = chart.options["yAxis"][0]
    # 100 应为 (min+max)/2，允许 ±1 误差（floor/ceil 取整）
    assert abs((x_axis["min"] + x_axis["max"]) / 2 - 100) <= 1
    assert abs((y_axis["min"] + y_axis["max"]) / 2 - 100) <= 1


def test_plot_rrg_echart_show_grid_false_hides_splitline():
    """show_grid=False 时 xAxis/yAxis 的 splitLine 都应隐藏。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum, show_grid=False)

    dumped = json.loads(chart.dump_options_with_quotes())
    assert dumped["xAxis"][0]["splitLine"]["show"] is False
    assert dumped["yAxis"][0]["splitLine"]["show"] is False


def test_plot_rrg_echart_bg_color_applied():
    """bg_color 应通过 InitOpts 落到 chart 的 backgroundColor。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum, bg_color="white")
    # pyecharts 把 InitOpts.bg_color 注入到 ``chart.bg_color`` 属性
    assert chart.bg_color == "white"


def test_plot_rrg_echart_shade_quadrants_corner_labels_and_colors():
    """shade_quadrants=True 时，4 角落标签位置 + 四象限专属颜色。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum, shade_quadrants=True)

    dumped = json.loads(chart.dump_options_with_quotes())
    # 找出含 markArea 的第一个 series（pyecharts 把 markarea 挂到 series）
    mark_area_data = None
    for s in dumped["series"]:
        if s.get("markArea"):
            mark_area_data = s["markArea"]["data"]
            break
    assert mark_area_data, "shade_quadrants=True 应产生 markArea"

    # 收集 (name, label.position, label.color) 三元组
    triples = []
    for item in mark_area_data:
        first = item[0]
        triples.append(
            (first.get("name"), first["label"].get("position"), first["label"].get("color"))
        )

    # 4 个象限的标签 ↔ 角落 ↔ 颜色映射
    expected = {
        ("领先", "insideTopRight", "#38b48b"),
        ("改善", "insideTopLeft", "#44617b"),
        ("滞后", "insideBottomLeft", "#bf783a"),
        ("疲软", "insideBottomRight", "#745399"),
    }
    assert set(triples) == expected, f"标签位置/颜色不匹配：{triples}"


def _has_center_crosshair(chart) -> tuple[bool, str | None]:
    """检查 markLine 中是否存在 axis-spanning 中心线（dict 形 item，非 pair list）。

    返回 (是否存在, 线条颜色)。颜色用于断言灰色而非红色。
    """
    dumped = json.loads(chart.dump_options_with_quotes())
    for s in dumped["series"]:
        ml = s.get("markLine")
        if not ml:
            continue
        for item in ml["data"]:
            if isinstance(item, dict) and (
                item.get("xAxis") is not None or item.get("yAxis") is not None
            ):
                return True, item.get("lineStyle", {}).get("color")
    return False, None


def test_plot_rrg_echart_center_line_default_omitted_when_shaded():
    """智能默认：shade_quadrants=True 时不画中心线（色块已指示象限）。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum, shade_quadrants=True)
    has_line, _ = _has_center_crosshair(chart)
    assert not has_line, "色块存在时智能默认应不画中心十字（视觉冗余）"


def test_plot_rrg_echart_center_line_default_shown_when_not_shaded():
    """智能默认：shade_quadrants=False 时画灰色中心线（指示象限分界）。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum, shade_quadrants=False)
    has_line, color = _has_center_crosshair(chart)
    assert has_line, "无色块时智能默认应画中心十字"
    assert color == "#888", f"中心线应为灰色 #888，实际 {color}"


def test_plot_rrg_echart_show_center_line_explicit_overrides():
    """显式 show_center_line=True 应覆盖智能默认，与 shade_quadrants 共存。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(
        rs_ratio, rs_momentum, shade_quadrants=True, show_center_line=True
    )
    has_line, _ = _has_center_crosshair(chart)
    assert has_line, "显式 show_center_line=True 应强制画中心线"


def test_plot_rrg_echart_markarea_attached_to_single_series():
    """markarea 是 series-scoped overlay——同一份 ops 挂给 N 个 series 会被
    echarts 渲染为 N 倍叠加，把 alpha=0.08 累计到完全不透明。回归测试：
    多 series 输入时，恰好只有 1 个 series 携带 markArea。"""
    # 20 个 series 是触发原 bug 的临界规模（叠加后浓度 ~81%）
    rs_ratio, rs_momentum = _make_rs_frames(n_industries=20)
    chart = plot_rrg_echart(rs_ratio, rs_momentum, shade_quadrants=True)

    dumped = json.loads(chart.dump_options_with_quotes())
    series_with_markarea = [s for s in dumped["series"] if s.get("markArea")]
    assert len(series_with_markarea) == 1, (
        f"shade_quadrants 应只挂 1 个 series，实际 {len(series_with_markarea)} 个 "
        f"(N 倍叠加 bug 复活)"
    )


def test_plot_rrg_echart_tooltip_item_trigger_and_rows():
    """tooltip 改 trigger='item' + 自定义 formatter，含五行标签与四象限术语。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum)

    tooltip = chart.options["tooltip"]
    assert tooltip.opts["trigger"] == "item"

    fmt = tooltip.opts["formatter"].js_code
    for row in ["标签", "趋势", "相对强弱", "相对强弱动量", "日期"]:
        assert row in fmt, f"tooltip formatter 缺少行：{row}"
    # 趋势用研报正术语，非图 #4 旧术语
    for term in ["领先", "改善", "滞后", "疲软"]:
        assert term in fmt
    assert "领涨" not in fmt and "走强" not in fmt, "不应复活旧术语"


# ============ plot_signal_heatmap（持仓信号热力图）============


def _make_score_panel(
    n_dates: int = 6, n_ind: int = 4, seed: int = 20260604
) -> pd.DataFrame:
    """构造合成持仓面板：index=月末日期，columns=申万一级代码，value∈{0,1}。"""
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range("2024-01-31", periods=n_dates, freq="ME")
    codes = [f"80{1010 + i * 10}.SI" for i in range(n_ind)]
    vals = rng.integers(0, 2, size=(n_dates, n_ind)).astype(float)
    return pd.DataFrame(vals, index=dates, columns=codes)


def test_plot_signal_heatmap_smoke():
    """返回 HeatMap：x 轴=日期 str，y 轴=行业名，每 cell 一个数据点。"""
    sp = _make_score_panel()
    code2name = {c: f"行业{i}" for i, c in enumerate(sp.columns)}
    chart = plot_signal_heatmap(sp, code2name=code2name)

    assert isinstance(chart, HeatMap)
    options = chart.options
    assert options["xAxis"][0]["data"] == [d.strftime("%Y-%m-%d") for d in sp.index]
    assert options["yAxis"][0]["data"] == [f"行业{i}" for i in range(sp.shape[1])]
    # 每个 (日期, 行业) cell 一个数据点
    dumped = json.loads(chart.dump_options_with_quotes())
    assert len(dumped["series"][0]["data"]) == sp.shape[0] * sp.shape[1]
    # series 默认图例（蓝块"持仓信号"）应隐藏——真正图例是底部 visualMap 的空仓/持仓
    assert dumped["legend"][0]["show"] is False


def test_plot_signal_heatmap_fallback_to_code_when_name_missing():
    """code2name 缺某列 → 该行业 y 轴标签回退用原 code。"""
    sp = _make_score_panel(n_ind=3)
    code2name = {sp.columns[0]: "电子", sp.columns[1]: "医药"}  # 第三列缺失
    chart = plot_signal_heatmap(sp, code2name=code2name)
    assert chart.options["yAxis"][0]["data"] == ["电子", "医药", sp.columns[2]]


def test_plot_signal_heatmap_none_code2name_uses_codes():
    """code2name=None 时 y 轴直接用代码（纯函数不取数）。"""
    sp = _make_score_panel(n_ind=2)
    chart = plot_signal_heatmap(sp)
    assert chart.options["yAxis"][0]["data"] == list(sp.columns)


def test_plot_signal_heatmap_piecewise_colors_and_tooltip():
    """visualMap piecewise 0 灰 / 1 红；tooltip 含行业/日期/持仓·空仓。"""
    sp = _make_score_panel()
    code2name = {c: f"行业{i}" for i, c in enumerate(sp.columns)}
    chart = plot_signal_heatmap(
        sp, code2name=code2name, held_color="#c23531", empty_color="#eeeeee"
    )

    dumped = json.loads(chart.dump_options_with_quotes())
    vm = dumped["visualMap"]
    assert vm["type"] == "piecewise"
    colors = {p["color"] for p in vm["pieces"]}
    assert {"#c23531", "#eeeeee"} <= colors

    fmt = chart.options["tooltip"].opts["formatter"].js_code
    for row in ["行业", "日期", "持仓", "空仓"]:
        assert row in fmt


def test_plot_signal_heatmap_tooltip_is_valid_js_after_render():
    """tooltip formatter 渲染后必须是合法 JS。pyecharts 把 JsCode 内的双引号
    JSON 转义成 \\" → `var xs = [\\"...\\"]` 非法（Invalid or unexpected token）→
    整段脚本中止、echarts.init 不执行、图空白。注入数组须用单引号规避。"""
    sp = _make_score_panel()
    chart = plot_signal_heatmap(
        sp, code2name={c: f"行业{i}" for i, c in enumerate(sp.columns)}
    )
    embed = chart.render_embed()
    i = embed.find('"formatter": function(p)')
    assert i != -1, "未找到 tooltip formatter"
    body = embed[i : i + 3000]
    # 被 JSON 转义的双引号 = 非法 JS 的根源
    assert '\\"' not in body, "formatter 内出现被转义的双引号 \\\" → 会破坏 JS 语法"


# ============ render_in_notebook（VS Code 稳健渲染助手）============


def test_render_in_notebook_uses_data_uri_iframe_not_requirejs():
    """render_in_notebook 用 base64 data-uri iframe 包 render_embed，绕开
    render_notebook() 的 RequireJS（VS Code 无全局 require → 静默空白）。"""
    import base64
    import re

    sp = _make_score_panel(n_ind=3)
    chart = plot_signal_heatmap(
        sp, code2name={c: f"行业{i}" for i, c in enumerate(sp.columns)}
    )
    disp = render_in_notebook(chart)

    # IPython.display.IFrame 的 HTML 串落在 _repr_html_()
    html = disp._repr_html_()
    assert "<iframe" in html
    assert 'src="data:text/html;base64,' in html
    assert "require(" not in html, "外层不应残留 RequireJS"

    # base64 解出的内嵌文档应自包含 echarts（普通 <script src>，非 requirejs）
    payload = re.search(r"base64,([A-Za-z0-9+/=]+)", html).group(1)
    decoded = base64.b64decode(payload).decode("utf-8")
    assert "echarts.min.js" in decoded
    assert "require(" not in decoded
    # echarts 必须来自可用 CDN（jsdelivr），不可是挂掉的 pyecharts 默认 CDN
    assert "cdn.jsdelivr.net" in decoded
    assert "assets.pyecharts.org" not in decoded


def test_plot_charts_use_working_cdn_not_dead_default():
    """plot_* 构造的图 js_host 应是可用 CDN——覆盖 render() 到文件路径
    （默认 assets.pyecharts.org 证书过期/不可达 → echarts 加载失败、图空白）。"""
    sp = _make_score_panel(n_ind=2)
    heatmap = plot_signal_heatmap(sp)
    rs_ratio, rs_momentum = _make_rs_frames()
    line = plot_rrg_echart(rs_ratio, rs_momentum)
    for chart in (heatmap, line):
        assert "assets.pyecharts.org" not in chart.js_host
        assert chart.js_host.startswith("https://")
        # render() 到文件的 HTML 也不应引用挂掉的默认 CDN
        assert "assets.pyecharts.org" not in chart.render_embed()


def test_render_in_notebook_works_for_rrg_line_chart_too():
    """助手对任意 pyecharts 图通用——RRG Line 图（plot_rrg_echart）同样适用。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum)
    disp = render_in_notebook(chart, height="600px")
    html = disp._repr_html_()
    assert 'src="data:text/html;base64,' in html
    assert 'height="600px"' in html


def test_plot_rrg_echart_tooltip_center_injected():
    """formatter 的象限/偏移判定基准应注入实际 center，而非硬编码 100。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum, center=(50, 50))

    fmt = chart.options["tooltip"].opts["formatter"].js_code
    assert "cx = 50" in fmt and "cy = 50" in fmt


def test_plot_functions_share_aligned_signature_prefix():
    """plot_rrg_echart 与 plot_rrg_chart 的签名前导块（数据参数 + 6 个共有
    keyword 参数）应同名 / 同序 / 同默认 / 同 kind，保证两后端无缝切换。"""
    import inspect

    echart = inspect.signature(plot_rrg_echart).parameters
    chart = inspect.signature(plot_rrg_chart).parameters

    # 1. 两个数据参数：位置-or-keyword，同名
    assert list(echart)[:2] == ["rs_ratio", "rs_momentum"]
    assert list(chart)[:2] == ["rs_ratio", "rs_momentum"]
    for fn_params in (echart, chart):
        for p in ("rs_ratio", "rs_momentum"):
            assert fn_params[p].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    # 2. 共有 keyword 前导块：同名 / 同序 / 同默认 / KEYWORD_ONLY
    common = ["tail_length", "step", "center", "symmetric", "shade_quadrants", "show_grid"]
    assert list(echart)[2 : 2 + len(common)] == common
    assert list(chart)[2 : 2 + len(common)] == common
    for name in common:
        assert echart[name].default == chart[name].default, f"{name} 默认值不一致"
        assert echart[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert chart[name].kind == inspect.Parameter.KEYWORD_ONLY


def test_subsample_tail_anchors_at_latest_and_counts_display_points():
    """_subsample_tail：step 抽稀锚定最新行，tail_length 作用于抽稀后的点数。"""
    from src.plotting import _subsample_tail

    df = pd.DataFrame(
        {"v": range(100)}, index=pd.date_range("2024-01-01", periods=100, freq="B")
    )
    # step=5：每 5 行取一个，锚定最后一行 → 最后一行必在内
    sub = _subsample_tail(df, step=5, tail_length=None)
    assert df.index[-1] in sub.index, "最新行必须保留"
    assert len(sub) == 20  # 100 / 5

    # step=5 + tail_length=10 → 抽稀后取最后 10 点（覆盖 50 交易日）
    sub2 = _subsample_tail(df, step=5, tail_length=10)
    assert len(sub2) == 10
    assert df.index[-1] in sub2.index
    # 相邻显示点的原始位置间隔应为 step
    positions = [df.index.get_loc(t) for t in sub2.index]
    assert all(b - a == 5 for a, b in zip(positions, positions[1:]))

    # step<=1 向后兼容：不抽稀
    assert len(_subsample_tail(df, step=1, tail_length=None)) == 100


def test_plot_rrg_echart_step_widens_point_spacing():
    """step>1 应减少显示点数（抽稀），step=1 全画。"""
    rs_ratio, rs_momentum = _make_rs_frames(n_days=100, n_industries=2)
    # 显式 tail_length=None 隔离 step 效应——本测专测抽稀，不受前导块默认
    # tail_length=10（研报图 2 风格组合）干扰，否则两侧都被裁到 10 点而相等。
    full = plot_rrg_echart(rs_ratio, rs_momentum, step=1, tail_length=None)
    sub = plot_rrg_echart(rs_ratio, rs_momentum, step=5, tail_length=None)

    def _pts(chart):
        return sum(len(s["data"]) for s in chart.options["series"])

    assert _pts(sub) < _pts(full), "step=5 应抽稀显示点"
    # 2 series × ceil(100/5)=20 → 40 点
    assert _pts(sub) == 40


def test_plot_rrg_echart_all_nan_input_does_not_crash():
    """全 NaN / 空有效值输入时，默认 symmetric=False 路径不应崩溃
    （回归：math.floor(np.nanmin(全 NaN)) 抛 ValueError）。"""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    cols = ["A", "B"]
    rs_ratio = pd.DataFrame(np.nan, index=dates, columns=cols)
    rs_momentum = pd.DataFrame(np.nan, index=dates, columns=cols)

    # 两条轴路径都不应抛
    chart_default = plot_rrg_echart(rs_ratio, rs_momentum, symmetric=False)
    chart_symmetric = plot_rrg_echart(rs_ratio, rs_momentum, symmetric=True)
    assert isinstance(chart_default, Line)
    assert isinstance(chart_symmetric, Line)
    # 全 NaN 退回 center±fallback，轴区间应有限且含 center
    x_axis = chart_default.options["xAxis"][0]
    assert x_axis["min"] <= 100 <= x_axis["max"]


def test_plot_rrg_echart_numpy_center_yields_valid_js():
    """center 含 numpy 标量时，formatter 注入应为合法 JS 数字而非
    'np.float64(100.0)'（回归：numpy>=2.0 的 repr 变更）。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(
        rs_ratio, rs_momentum, center=(np.float64(100.0), np.float64(100.0))
    )
    fmt = chart.options["tooltip"].opts["formatter"].js_code
    assert "np.float64" not in fmt, "center 的 numpy 标量泄漏进 JS"
    assert "var cx = 100.0" in fmt and "cy = 100.0" in fmt


def test_plot_rrg_echart_arrow_at_trajectory_end():
    """末端点（markLineItem[-1]）应带 symbol='arrow'，前端点 symbol='none'。"""
    rs_ratio, rs_momentum = _make_rs_frames()
    chart = plot_rrg_echart(rs_ratio, rs_momentum)

    dumped = json.loads(chart.dump_options_with_quotes())
    # 至少一条 series 的 markLine 第一对应该有 arrow 末端
    found_arrow = False
    for s in dumped["series"]:
        ml = s.get("markLine")
        if not ml:
            continue
        for pair in ml["data"]:
            # pair 形式 [{coord, symbol, ...}, {coord, symbol, ...}]
            if isinstance(pair, list) and len(pair) >= 2:
                if pair[-1].get("symbol") == "arrow":
                    found_arrow = True
                    break
        if found_arrow:
            break
    assert found_arrow, "末端 markLineItem 应 symbol='arrow' 形成方向箭头"

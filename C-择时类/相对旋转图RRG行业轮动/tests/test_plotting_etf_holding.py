"""plot_etf_holding_heatmap（matplotlib 树版：行业树 y 轴 + 时间热力图）契约测试。

不依赖 qlib / DolphinDB / TuShare / .env——仅用合成权重面板 `w` +
合成 etf_industry（静态 Series / 动态 long DataFrame）。

布局：左 `ax_tree` 画 root→行业→ETF 语义树，右 `ax_heat` 画 ETF×时间二元持仓
热力图（1=持仓红 / 0=空仓灰）。断言用 Figure / Axes 对象。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 无显示环境

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plotting import plot_etf_holding_heatmap


def _make_weight_panel() -> pd.DataFrame:
    """3 个换仓日 × 5 只 ETF 的权重面板（0=空仓，>0=持仓）。"""
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    etfs = ["512760.SH", "159995.SZ", "512010.SH", "159992.SZ", "512690.SH"]
    vals = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.0],
    ])
    return pd.DataFrame(vals, index=dates, columns=etfs)


def _static_mapping() -> pd.Series:
    return pd.Series({
        "512760.SH": "801080.SI", "159995.SZ": "801080.SI",   # 电子
        "512010.SH": "801150.SI", "159992.SZ": "801150.SI",   # 医药
        "512690.SH": "801120.SI",                              # 食品饮料
    })


_ETF_NAMES = {
    "512760.SH": "半导体ETF", "159995.SZ": "芯片ETF", "512010.SH": "医药ETF",
    "159992.SZ": "创新药ETF", "512690.SH": "酒ETF",
}
_IND_NAMES = {"801080.SI": "电子", "801150.SI": "医药生物", "801120.SI": "食品饮料"}
_NAME2CODE = {v: k for k, v in _ETF_NAMES.items()}


def _etf_leaf_labels(fig) -> list[str]:
    """ETF 名是热力图(axes[1]) 的 y 刻度标签，行序（行 0..n-1 自顶向下）。"""
    return [t.get_text() for t in fig.axes[1].get_yticklabels()]


def _row_etf_codes(fig) -> list[str]:
    """从 ETF 叶子标签（按行序）反推 ETF 代码。"""
    return [_NAME2CODE.get(lab, lab) for lab in _etf_leaf_labels(fig)]


def _matrix(ax_heat) -> np.ndarray:
    return np.asarray(ax_heat.images[0].get_array())


def _col_labels(ax_heat) -> list[str]:
    return [t.get_text() for t in ax_heat.get_xticklabels()]


# ============ 结构 ============


def test_returns_figure_with_two_axes():
    """返回 matplotlib Figure，含树面板 + 热力图两个 axes。"""
    w = _make_weight_panel()
    fig = plot_etf_holding_heatmap(
        w, _static_mapping(), etf_name=_ETF_NAMES, industry_name=_IND_NAMES
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    ax_heat = fig.axes[1]
    assert len(_etf_leaf_labels(fig)) == w.shape[1]  # 每只 ETF 一片叶子标签
    assert len(_col_labels(ax_heat)) == w.shape[0]    # 每个换仓日一列
    plt.close(fig)


# ============ 行序按行业聚拢 ============


def test_rows_grouped_by_representative_industry():
    """同代表行业的 ETF 在 y 行序上连续。"""
    w = _make_weight_panel()
    fig = plot_etf_holding_heatmap(
        w, _static_mapping(), etf_name=_ETF_NAMES, industry_name=_IND_NAMES
    )
    codes = _row_etf_codes(fig)
    m = _static_mapping()
    inds = [m[c] for c in codes]
    for ind in set(inds):
        pos = [i for i, x in enumerate(inds) if x == ind]
        assert pos == list(range(pos[0], pos[-1] + 1)), f"行业 {ind} 行不连续：{pos}"
    plt.close(fig)


# ============ 二元持仓矩阵 ============


def test_binary_matrix_held_vs_empty():
    """热力图矩阵：持仓格=1、空仓格=0。"""
    w = _make_weight_panel()
    fig = plot_etf_holding_heatmap(
        w, _static_mapping(), etf_name=_ETF_NAMES, industry_name=_IND_NAMES
    )
    ax_heat = fig.axes[1]
    arr = _matrix(ax_heat)
    codes = _row_etf_codes(fig)
    cols = _col_labels(ax_heat)
    ri = codes.index("512760.SH")           # 半导体ETF：01-31 持仓、03-29 空仓
    assert arr[ri, cols.index("2024-01-31")] == 1
    assert arr[ri, cols.index("2024-03-29")] == 0
    plt.close(fig)


def test_held_empty_colors_two_level_cmap():
    """二元配色：cmap 恰两段（空仓灰 / 持仓红）。"""
    w = _make_weight_panel()
    fig = plot_etf_holding_heatmap(
        w, _static_mapping(), etf_name=_ETF_NAMES, industry_name=_IND_NAMES,
        held_color="#c23531", empty_color="#eeeeee",
    )
    assert fig.axes[1].images[0].get_cmap().N == 2
    plt.close(fig)


# ============ 树面板 ============


def test_tree_panel_has_industry_labels_and_lines():
    """左树面板含各行业名文本 + root 标签，且画了树线。"""
    w = _make_weight_panel()
    fig = plot_etf_holding_heatmap(
        w, _static_mapping(), etf_name=_ETF_NAMES, industry_name=_IND_NAMES,
        root_label="全市场",
    )
    ax_tree = fig.axes[0]
    texts = {t.get_text() for t in ax_tree.texts}
    assert {"电子", "医药生物", "食品饮料"} <= texts
    assert "全市场" in texts
    assert len(ax_tree.lines) > 0, "树连接线未绘制"
    plt.close(fig)


# ============ 回退（缺名 / 缺映射）============


def test_label_fallback_to_code_when_names_missing():
    """缺 etf_name / industry_name → y 轴叶子标签 + 树枝标签回退用代码。"""
    w = _make_weight_panel()
    fig = plot_etf_holding_heatmap(w, _static_mapping())  # 不传名称
    assert set(_etf_leaf_labels(fig)) == set(w.columns), "缺名应回退用 ETF 代码"
    tree_texts = {t.get_text() for t in fig.axes[0].texts}
    assert "801080.SI" in tree_texts, "缺行业名应回退用 sw_code 树枝标签"
    plt.close(fig)


def test_held_but_unmapped_etf_grouped_unclassified_last():
    """持仓但无映射 → 归「未分类」枝、行排最末。"""
    w = _make_weight_panel()
    mapping = _static_mapping().drop("512010.SH")  # 512010 持仓但无映射
    fig = plot_etf_holding_heatmap(
        w, mapping, etf_name=_ETF_NAMES, industry_name=_IND_NAMES
    )
    codes = _row_etf_codes(fig)
    # 已分类 ETF 全部在未分类 512010.SH 之前
    assert codes.index("512010.SH") == len(codes) - 1 or all(
        codes.index("512010.SH") > codes.index(c)
        for c in mapping.index if c in codes
    )
    assert "未分类" in {t.get_text() for t in fig.axes[0].texts}
    plt.close(fig)


# ============ 动态 long 入参 ============


def test_dynamic_long_input_returns_valid_figure():
    """动态 long 入参（代表行业取众数）返回有效 Figure。"""
    w = _make_weight_panel()
    dyn = pd.DataFrame(
        [
            ("2024-01-31", "512760.SH", "801080.SI"),
            ("2024-02-29", "512010.SH", "801150.SI"),
            ("2024-03-29", "512010.SH", "801080.SI"),  # 换组：众数平局取字典序最小 801080
            ("2024-03-29", "159992.SZ", "801150.SI"),
        ],
        columns=["rebal_date", "etf_code", "sw_code"],
    )
    dyn["rebal_date"] = pd.to_datetime(dyn["rebal_date"])
    fig = plot_etf_holding_heatmap(
        w, dyn, etf_name=_ETF_NAMES, industry_name=_IND_NAMES
    )
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    plt.close(fig)


# ============ 确定性 ============


def test_deterministic_row_order():
    """同输入两次调用 → 行序一致。"""
    w = _make_weight_panel()
    f1 = plot_etf_holding_heatmap(w, _static_mapping(), etf_name=_ETF_NAMES,
                                  industry_name=_IND_NAMES)
    f2 = plot_etf_holding_heatmap(w, _static_mapping(), etf_name=_ETF_NAMES,
                                  industry_name=_IND_NAMES)
    assert _row_etf_codes(f1) == _row_etf_codes(f2)
    plt.close(f1)
    plt.close(f2)


# ============ 健壮性回归（code-review 找到的 bug）============


def test_non_string_columns_do_not_crash():
    """非字符串列标签（如 int）不应 KeyError——列名 str 化后用于查找与归类，
    数据访问也须走 str 化后的列（回归：旧版 weight_panel[etf] 用 str 键回查 int 列）。"""
    w = _make_weight_panel()
    w.columns = [100, 200, 300, 400, 500]  # 非 str 列
    mapping = pd.Series({100: "801080.SI", 200: "801080.SI",
                         300: "801150.SI", 400: "801150.SI", 500: "801120.SI"})
    fig = plot_etf_holding_heatmap(w, mapping, industry_name=_IND_NAMES)
    assert isinstance(fig, Figure)
    # 5 只 ETF 各一片叶子标签
    assert len(_etf_leaf_labels(fig)) == 5
    plt.close(fig)


def test_duplicate_dates_do_not_crash():
    """index 含重复换仓日不应触发「Series 真值歧义」ValueError（回归：旧版
    col.loc[d] 在重复日返回 Series → pd.notna(v) and v>thr 抛 ValueError）。"""
    w = _make_weight_panel()
    w.index = pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29"])  # 重复日
    fig = plot_etf_holding_heatmap(
        w, _static_mapping(), etf_name=_ETF_NAMES, industry_name=_IND_NAMES)
    assert isinstance(fig, Figure)
    assert len(fig.axes[1].get_xticklabels()) == 3  # 每个 index 行一列
    plt.close(fig)


def test_empty_weight_panel_returns_figure():
    """空权重面板（0 ETF / 0 换仓日）应安全返回 Figure，不抛错、不产生奇异轴告警。"""
    import warnings

    w = pd.DataFrame(index=pd.to_datetime([]), columns=pd.Index([], dtype=object))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 任何 UserWarning 都视为失败
        fig = plot_etf_holding_heatmap(w, pd.Series(dtype="object"))
    assert isinstance(fig, Figure)
    plt.close(fig)

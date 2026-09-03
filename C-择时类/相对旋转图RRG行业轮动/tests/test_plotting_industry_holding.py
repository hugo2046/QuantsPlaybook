"""plot_industry_holding_panel 契约测试（合成数据、零 I/O）。

图 = 行业 × 换仓日 单面板；格值 = 该期该行业代表 ETF 的 LP 权重；
信号选中但无权重 → 空心框。每条测试都写明「哪种生产改动会让它变红」。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from src import plotting

D1, D2, D3 = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
IND_A, IND_B, IND_C = "801080.SI", "801150.SI", "801750.SI"   # 电子 / 医药 / 计算机


def _fixture():
    """A: 三期都选中且都有权重；B: 第 2 期选中但无代表；C: 从未选中。"""
    score = pd.DataFrame(
        {IND_A: [1, 1, 1], IND_B: [0, 1, 0], IND_C: [0, 0, 0]}, index=[D1, D2, D3])
    reps = pd.DataFrame(
        {IND_A: ["512480.SH", "512480.SH", "159995.SZ"], IND_B: [np.nan, np.nan, np.nan]},
        index=[D1, D2, D3])
    w = pd.DataFrame(
        {"512480.SH": [0.20, 0.05, 0.0], "159995.SZ": [0.0, 0.0, 0.18]}, index=[D1, D2, D3])
    return score, reps, w


def _cells(fig):
    ax = fig.axes[0]
    return ax, ax.images[0].get_array()


def teardown_function(_):
    plt.close("all")


def test_returns_figure_single_axes():
    """让它变红：函数不存在 / 返回非 Figure / 多轴。"""
    score, reps, w = _fixture()
    fig = plotting.plot_industry_holding_panel(score, reps, w)
    assert isinstance(fig, plt.Figure) and len(fig.axes) == 1


def test_rows_are_ever_selected_industries_in_code_order():
    """行 = 窗口内至少选中过一次的行业，按代码序；从未选中的 C 不出现。
    让它变红：画全部行业 / 行序改为出现顺序。"""
    score, reps, w = _fixture()
    ax, _ = _cells(plotting.plot_industry_holding_panel(score, reps, w))
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [IND_A, IND_B]


def test_cell_value_is_rep_etf_weight():
    """格值 = w[date, reps[date, ind]]，逐格精确；无代表 → 0。
    让它变红：取错 ETF / 取错日期 / 把二元 held 当权重。"""
    score, reps, w = _fixture()
    _, arr = _cells(plotting.plot_industry_holding_panel(score, reps, w))
    got = np.asarray(arr.filled(0.0) if np.ma.isMaskedArray(arr) else arr, dtype=float)
    expect = np.array([[0.20, 0.05, 0.18],   # A：D1 512480→.20, D2 512480→.05, D3 159995→.18
                       [0.00, 0.00, 0.00]])  # B：无代表
    np.testing.assert_allclose(got, expect, atol=1e-12)


def test_selected_without_weight_draws_hollow_box_only_there():
    """双侧：B@D2 选中但无权重 → 恰好一个空心框；A（有权重）与 B@D1/D3（未选中）都没有。
    让它变红：空心框条件写反 / 对所有选中格都画框 / 不画框。"""
    score, reps, w = _fixture()
    ax, _ = _cells(plotting.plot_industry_holding_panel(score, reps, w))
    boxes = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_fill() is False]
    assert len(boxes) == 1
    x, y = boxes[0].get_xy()
    # 行 B 是第 2 行（index 1），列 D2 是第 2 列（index 1）；格子左下角 = (col-0.5, row-0.5)
    assert (round(x + 0.5), round(y + 0.5)) == (1, 1)


def test_dates_filter_limits_columns():
    """dates= 只画指定换仓日。让它变红：忽略 dates 参数。"""
    score, reps, w = _fixture()
    ax, arr = _cells(plotting.plot_industry_holding_panel(score, reps, w, dates=[D2, D3]))
    assert arr.shape[1] == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == ["2024-02-29", "2024-03-29"]


def test_labels_use_names_when_given_else_codes():
    """让它变红：名称映射不生效 / 缺名时不回退代码。"""
    score, reps, w = _fixture()
    ax, _ = _cells(plotting.plot_industry_holding_panel(
        score, reps, w, industry_name={IND_A: "电子"}, etf_name={"512480.SH": "半导体ETF"}))
    assert [t.get_text() for t in ax.get_yticklabels()] == ["电子", IND_B]   # B 缺名 → 代码
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "半导体" in texts and "159995" in texts                       # ETF 缺名 → 代码


def test_annotate_none_has_no_cell_texts():
    """让它变红：annotate=None 仍写格内文字。"""
    score, reps, w = _fixture()
    ax, _ = _cells(plotting.plot_industry_holding_panel(score, reps, w, annotate=None))
    assert len(ax.texts) == 0   # ArtistList，不能与 [] 比较


def test_all_unselected_returns_figure_without_crash():
    """让它变红：空行业集时 imshow 抛错。"""
    score, reps, w = _fixture()
    fig = plotting.plot_industry_holding_panel(score * 0, reps, w)
    assert isinstance(fig, plt.Figure)

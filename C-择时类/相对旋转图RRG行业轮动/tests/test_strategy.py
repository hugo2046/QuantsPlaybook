"""strategy 层 + to_equal_weight 单元测试（全合成数据）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import (
    distance_from_center,
    select_by_diffusion,
    select_by_quadrant,
    select_diffusion_with_rrg,
)
from src.utils import to_equal_weight


def test_to_equal_weight_rows_sum_to_1():
    mask = pd.DataFrame(
        [[True, True, False, False],
         [True, False, True, True],
         [False, False, False, False]],
        columns=list("ABCD"),
    )
    w = to_equal_weight(mask)
    assert np.allclose(w.iloc[0].to_numpy(), [0.5, 0.5, 0.0, 0.0])
    assert np.allclose(w.iloc[1].to_numpy(), [1 / 3, 0.0, 1 / 3, 1 / 3])
    assert np.allclose(w.iloc[2].to_numpy(), [0.0, 0.0, 0.0, 0.0])  # 全 False → 全 0


def test_distance_from_center_pythagoras():
    rs_ratio = pd.DataFrame({"A": [103.0], "B": [100.0]})
    rs_mom = pd.DataFrame({"A": [104.0], "B": [100.0]})
    d = distance_from_center(rs_ratio, rs_mom)
    assert d["A"].iloc[0] == pytest.approx(5.0)   # sqrt(3² + 4²)
    assert d["B"].iloc[0] == pytest.approx(0.0)


def test_select_by_diffusion_top_n():
    diffusion = pd.DataFrame(
        [[0.9, 0.8, 0.7, 0.6, 0.5],
         [0.1, 0.2, 0.3, np.nan, 0.5]],
        columns=list("ABCDE"),
    )
    sel = select_by_diffusion(diffusion, top_n=3)
    # 第 0 行：top3 = A,B,C
    assert list(sel.columns[sel.iloc[0].to_numpy()]) == ["A", "B", "C"]
    # 第 1 行：有效值 {A0.1,B0.2,C0.3,E0.5} 的 top3 = E,C,B；D 为 NaN 不选
    assert set(sel.columns[sel.iloc[1].to_numpy()]) == {"E", "C", "B"}
    assert bool(sel.iloc[1]["D"]) is False
    assert sel.dtypes.eq(bool).all()


def test_select_by_quadrant_top_n():
    # 8 个行业全在象限 1（>100,>100），距离随序号递增
    cols = [f"I{i}" for i in range(8)]
    vals = [[100.0 + i for i in range(1, 9)]]  # I0=101 ... I7=108
    rs_ratio = pd.DataFrame(vals, columns=cols)
    rs_mom = pd.DataFrame(vals, columns=cols)
    sel = select_by_quadrant(rs_ratio, rs_mom, quadrants=(1,), top_n=6)
    chosen = set(sel.columns[sel.iloc[0].to_numpy()])
    assert chosen == {f"I{i}" for i in range(2, 8)}  # 最远 6 个 = I2..I7
    assert int(sel.iloc[0].sum()) == 6


def test_select_by_quadrant_two_quadrants():
    rs_ratio = pd.DataFrame({"a": [101.0], "b": [99.0], "c": [99.0]})
    rs_mom = pd.DataFrame({"a": [101.0], "b": [101.0], "c": [99.0]})
    # a=象限1，b=象限2，c=象限3
    sel = select_by_quadrant(rs_ratio, rs_mom, quadrants=(1, 2), top_n=6)
    assert bool(sel.iloc[0]["a"]) is True
    assert bool(sel.iloc[0]["b"]) is True
    assert bool(sel.iloc[0]["c"]) is False


def test_select_diffusion_with_rrg_no_padding():
    cols = list("ABCDEF")
    diffusion = pd.DataFrame([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]], columns=cols)
    # A=象限1, B=象限2（mom>100，保留）；C,E=象限4，D,F=象限3（剔除）
    rs_ratio = pd.DataFrame([[101.0, 99.0, 101.0, 99.0, 101.0, 99.0]], columns=cols)
    rs_mom = pd.DataFrame([[101.0, 101.0, 99.0, 99.0, 99.0, 99.0]], columns=cols)
    sel = select_diffusion_with_rrg(
        diffusion, rs_ratio, rs_mom, top_n=6, keep_quadrants=(1, 2)
    )
    chosen = list(sel.columns[sel.iloc[0].to_numpy()])
    assert chosen == ["A", "B"]      # 扩散 top6 → 剔除 3/4 象限 → 仅剩 A,B，不补足到 6
    assert int(sel.iloc[0].sum()) == 2


def test_select_by_quadrant_warmup_nan_not_selected():
    rs_ratio = pd.DataFrame({"A": [np.nan], "B": [np.nan]})
    rs_mom = pd.DataFrame({"A": [np.nan], "B": [np.nan]})
    sel = select_by_quadrant(rs_ratio, rs_mom)
    assert sel.iloc[0].tolist() == [False, False]
    assert sel.dtypes.eq(bool).all()


def test_select_warmup_nan_not_selected():
    diffusion = pd.DataFrame([[np.nan, np.nan, 0.5]], columns=list("ABC"))
    sel = select_by_diffusion(diffusion, top_n=6)
    assert bool(sel.iloc[0]["A"]) is False
    assert bool(sel.iloc[0]["C"]) is True

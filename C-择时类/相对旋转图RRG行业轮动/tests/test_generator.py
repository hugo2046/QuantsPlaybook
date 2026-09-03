"""RRGSignalGenerator 编排层测试：派发 / 按需取数 / 缓存（全 monkeypatch，不碰 DB）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import generator as gen_mod


@pytest.fixture
def fake_rrg_dataset():
    """400 交易日 × 3 行业（≥ compute_rrg 最小长度 319），返回 (benchmark, industry_close, {})."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    inds = ["801010", "801030", "801040"]
    px = pd.DataFrame(
        np.cumprod(1.0 + rng.normal(0.0003, 0.012, (400, 3)), axis=0) * 100.0,
        index=dates, columns=inds,
    )
    return px.mean(axis=1), px, {}


@pytest.fixture
def fake_diffusion_dataset():
    """400 交易日 × 9 股票（3 行业各 3 只），返回 (close, membership, free_float_mv)."""
    rng = np.random.default_rng(8)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    inds = ["801010", "801030", "801040"]
    codes = [f"S{i}" for i in range(9)]
    close = pd.DataFrame(
        np.cumprod(1.0 + rng.normal(0.0003, 0.018, (400, 9)), axis=0) * 50.0,
        index=dates, columns=codes,
    )
    member_row = [ind for ind in inds for _ in range(3)]
    membership = pd.DataFrame([member_row] * 400, index=dates, columns=codes)
    mv = pd.DataFrame(rng.uniform(1e8, 1e10, (400, 9)), index=dates, columns=codes)
    return close, membership, mv


def test_generate_dispatch(monkeypatch, fake_rrg_dataset, fake_diffusion_dataset):
    monkeypatch.setattr(gen_mod, "fetch_rrg_dataset", lambda *a, **k: fake_rrg_dataset)
    monkeypatch.setattr(gen_mod, "fetch_diffusion_dataset", lambda *a, **k: fake_diffusion_dataset)
    gen = gen_mod.RRGSignalGenerator(start_date="2020-01-01", end_date="2021-07-01")
    mask = gen.generate("diffusion_rrg")
    assert isinstance(mask, pd.DataFrame)
    assert mask.dtypes.eq(bool).all()
    assert set(mask.columns) <= {"801010", "801030", "801040"}


def test_generate_unknown_signal_raises():
    gen = gen_mod.RRGSignalGenerator()
    with pytest.raises(ValueError, match="未知 signal"):
        gen.generate("nope")


def test_quadrant_skips_diffusion_fetch(monkeypatch, fake_rrg_dataset):
    monkeypatch.setattr(gen_mod, "fetch_rrg_dataset", lambda *a, **k: fake_rrg_dataset)

    def boom(*a, **k):
        raise AssertionError("quadrant 信号不应触发 fetch_diffusion_dataset")

    monkeypatch.setattr(gen_mod, "fetch_diffusion_dataset", boom)
    gen = gen_mod.RRGSignalGenerator(start_date="2020-01-01", end_date="2021-07-01")
    mask = gen.generate("quadrant", quadrants=(1, 2))
    assert isinstance(mask, pd.DataFrame)


def test_generate_caches_fetch(monkeypatch, fake_rrg_dataset, fake_diffusion_dataset):
    calls = {"rrg": 0}

    def rrg(*a, **k):
        calls["rrg"] += 1
        return fake_rrg_dataset

    monkeypatch.setattr(gen_mod, "fetch_rrg_dataset", rrg)
    monkeypatch.setattr(gen_mod, "fetch_diffusion_dataset", lambda *a, **k: fake_diffusion_dataset)
    gen = gen_mod.RRGSignalGenerator(start_date="2020-01-01", end_date="2021-07-01")
    gen.generate("diffusion_rrg")
    gen.generate("diffusion_rrg")
    assert calls["rrg"] == 1  # 第二次复用缓存，不重复取数


def test_diffusion_skips_rrg_fetch(monkeypatch, fake_diffusion_dataset):
    monkeypatch.setattr(gen_mod, "fetch_diffusion_dataset", lambda *a, **k: fake_diffusion_dataset)

    def boom(*a, **k):
        raise AssertionError("diffusion 信号不应触发 fetch_rrg_dataset")

    monkeypatch.setattr(gen_mod, "fetch_rrg_dataset", boom)
    gen = gen_mod.RRGSignalGenerator(start_date="2020-01-01", end_date="2021-07-01")
    mask = gen.generate("diffusion")
    assert isinstance(mask, pd.DataFrame)


def test_generate_exclude_removes_and_backfills():
    """exclude 在排序前剔除行业 → 被剔行业不出现，且 top_n 回填下一名（研报剔综合口径）。"""
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    cols = ["801010", "801030", "801230"]  # 801230 = 综合（研报应剔除）
    gen = gen_mod.RRGSignalGenerator(start_date="2022-01-03", end_date="2022-01-07")
    # 直接注入扩散指标缓存（跳过取数）：综合扩散最高、801010 次之、801030 最低
    gen._diffusion = pd.DataFrame([[0.5, 0.2, 0.9]] * 3, index=dates, columns=cols)

    out = gen.generate("diffusion", top_n=2)
    # 不剔除：top2 = {801010, 综合}，801030 落选
    assert out.loc[dates[0], "801230"]
    assert not out.loc[dates[0], "801030"]

    out_ex = gen.generate("diffusion", top_n=2, exclude=["801230"])
    # 剔除综合后 universe={801010,801030}，top2 两者全入选 → 801030 被回填
    assert "801230" not in out_ex.columns
    assert out_ex.loc[dates[0], "801030"]


def test_generate_exclude_ignores_absent_codes():
    """exclude 含不存在的行业代码时静默忽略，不报错、不改变结果。"""
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    cols = ["801010", "801030"]
    gen = gen_mod.RRGSignalGenerator(start_date="2022-01-03", end_date="2022-01-07")
    gen._diffusion = pd.DataFrame([[0.5, 0.2]] * 3, index=dates, columns=cols)
    out = gen.generate("diffusion", top_n=2, exclude=["801999"])  # 不存在
    assert set(out.columns) == {"801010", "801030"}


def test_generate_trims_to_window(monkeypatch, fake_rrg_dataset):
    # fake 数据从 2020-01-01 起；请求窗口起点更晚 → 输出须裁掉 warm-up 预取的早期行，
    # 保证不同 signal 的行索引一致（否则跨 signal 月末采样会 KeyError）。
    monkeypatch.setattr(gen_mod, "fetch_rrg_dataset", lambda *a, **k: fake_rrg_dataset)
    gen = gen_mod.RRGSignalGenerator(start_date="2020-06-01", end_date="2021-07-01")
    mask = gen.generate("quadrant")
    assert mask.index.min() >= pd.Timestamp("2020-06-01")
    assert mask.index.max() <= pd.Timestamp("2021-07-01")

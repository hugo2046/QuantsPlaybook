"""ETF 组合优化层单元测试（西部证券研报 §3.1–3.4 LP）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    batch_optimize,
    optimize_etf_weights,
    weights_base,
    weights_etf_capped,
    weights_hs300_enhanced,
    weights_ind_free_capped,
    weights_unconstrained,
)


def test_objective_picks_highest_exposure_etf():
    """sum=1、long-only、无行业约束 → 全压对选中行业暴露最高的 ETF。"""
    score = pd.Series({"IND1": 1.0, "IND2": 0.0})
    # ETF1 对选中行业 IND1 暴露 0.8，ETF2 暴露 0.3
    exposure = pd.DataFrame(
        {"IND1": [0.8, 0.3], "IND2": [0.1, 0.6]},
        index=["ETF1", "ETF2"],
    )
    w = optimize_etf_weights(score, exposure, gross_mode="eq", gross_target=1.0)
    assert w["ETF1"] == pytest.approx(1.0, abs=1e-5)
    assert w["ETF2"] == pytest.approx(0.0, abs=1e-5)
    assert w.sum() == pytest.approx(1.0, abs=1e-5)


def test_base_converges_to_equal_industry():
    """§3.1：行业暴露上限 1/k → 组合在各选中行业上趋近等权 1/k。"""
    # k=2 选中行业，两只纯 ETF 各 100% 落在一个行业
    score = pd.Series({"IND1": 1.0, "IND2": 1.0, "IND3": 0.0})
    exposure = pd.DataFrame(
        {"IND1": [1.0, 0.0], "IND2": [0.0, 1.0], "IND3": [0.0, 0.0]},
        index=["ETF_A", "ETF_B"],
    )
    w = optimize_etf_weights(
        score, exposure, gross_mode="eq", gross_target=1.0, industry_cap=0.5
    )
    # 行业约束 0.5 把两只 ETF 压成等权
    assert w["ETF_A"] == pytest.approx(0.5, abs=1e-5)
    assert w["ETF_B"] == pytest.approx(0.5, abs=1e-5)
    # 组合在每个选中行业的暴露 = 1/k = 0.5
    portfolio_exposure = exposure.T @ w
    assert portfolio_exposure["IND1"] == pytest.approx(0.5, abs=1e-5)
    assert portfolio_exposure["IND2"] == pytest.approx(0.5, abs=1e-5)


def test_hs300_active_band():
    """§3.4：行业偏离带 |组合暴露 − 基准| ≤ δ 必须满足（无约束时会越界）。"""
    score = pd.Series({"IND1": 1.0, "IND2": 1.0})
    # ETF_A 目标系数更高 → 无约束最优会全压 ETF_A，使 IND2 暴露=0 越界
    exposure = pd.DataFrame(
        {"IND1": [1.0, 0.0], "IND2": [0.0, 0.9]},
        index=["ETF_A", "ETF_B"],
    )
    benchmark = pd.Series({"IND1": 0.5, "IND2": 0.5})
    band = 0.05
    w = optimize_etf_weights(
        score, exposure, gross_mode="eq", gross_target=1.0,
        benchmark=benchmark, active_band=band,
    )
    portfolio_exposure = exposure.T @ w
    for ind in ["IND1", "IND2"]:
        assert abs(portfolio_exposure[ind] - benchmark[ind]) <= band + 1e-6


def test_infeasible_raises_and_nan():
    """不可行：默认抛带 status 的明确异常；on_infeasible='nan' 返回全 NaN。"""
    score = pd.Series({"IND1": 1.0})
    exposure = pd.DataFrame({"IND1": [0.5, 0.5]}, index=["ETF1", "ETF2"])
    # w_max=0.2，两只 ETF → Σw 最大 0.4 < 1；gross=1 的等式约束不可行
    with pytest.raises(ValueError, match="infeasible|不可行"):
        optimize_etf_weights(
            score, exposure, w_max=0.2, gross_target=1.0, gross_mode="eq"
        )
    w = optimize_etf_weights(
        score, exposure, w_max=0.2, gross_target=1.0, gross_mode="eq",
        on_infeasible="nan",
    )
    assert w.isna().all()
    assert list(w.index) == ["ETF1", "ETF2"]


def test_k_zero_returns_empty():
    """当日无选中行业（score 全 0）→ 返回全 0 权重（空仓），不强行满仓。"""
    score = pd.Series({"IND1": 0.0, "IND2": 0.0})
    exposure = pd.DataFrame(
        {"IND1": [0.5, 0.3], "IND2": [0.2, 0.4]}, index=["E1", "E2"]
    )
    w = optimize_etf_weights(score, exposure, gross_mode="eq", gross_target=1.0)
    assert (w == 0.0).all()
    assert list(w.index) == ["E1", "E2"]


def test_weights_base_preset_caps_industry():
    """§3.1 预设：满仓 + 每行业暴露 ≤ 1/k。"""
    score = pd.Series({"IND1": 1.0, "IND2": 1.0, "IND3": 0.0})
    exposure = pd.DataFrame(
        {"IND1": [1.0, 0.0], "IND2": [0.0, 1.0], "IND3": [0.0, 0.0]},
        index=["ETF_A", "ETF_B"],
    )
    w = weights_base(score, exposure)
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    portfolio_exposure = exposure.T @ w
    assert portfolio_exposure["IND1"] <= 0.5 + 1e-6  # 1/k, k=2
    assert portfolio_exposure["IND2"] <= 0.5 + 1e-6


def test_weights_etf_capped_preset():
    """§3.2 预设：单 ETF ≤ 20%，仓位 = min(1, 0.2k)；k=2 → 0.4。"""
    score = pd.Series({"IND1": 1.0, "IND2": 1.0})
    exposure = pd.DataFrame(
        {"IND1": [1.0, 0.0], "IND2": [0.0, 1.0]}, index=["ETF_A", "ETF_B"]
    )
    w = weights_etf_capped(score, exposure)
    assert (w <= 0.2 + 1e-6).all()
    assert w.sum() == pytest.approx(0.4, abs=1e-5)  # min(1, 0.2*2)


def test_weights_unconstrained_preset_concentrates():
    """§3.3 预设：无行业约束、Σw≤1 → 集中到目标系数最高的 ETF。"""
    score = pd.Series({"IND1": 1.0})
    exposure = pd.DataFrame({"IND1": [0.8, 0.3]}, index=["ETF_A", "ETF_B"])
    w = weights_unconstrained(score, exposure)
    assert w["ETF_A"] == pytest.approx(1.0, abs=1e-5)
    assert w["ETF_B"] == pytest.approx(0.0, abs=1e-5)


def test_weights_ind_free_capped_differs_from_both_report_variants():
    """自研变体必须同时区别于 §3.3（不退化到 1 只）与 §3.2（允许突破 1/k 行业上限）。

    这条断言是「别把自研变体当研报口径用」的护栏。构造 5 个行业、9 只纯行业 ETF，
    目标系数刻意两两不等（避免 LP 退化成任意解，那会让断言失去判别力）。
    """
    score = pd.Series({f"IND{i}": 1.0 for i in range(1, 6)})  # k=5 → gross=min(1,0.2*5)=1.0
    rows = {}
    # IND1 上 5 只，暴露 1.00 / 0.95×4——保证最高者唯一，§3.3 才有确定的单只解
    rows["A1"] = {"IND1": 1.00}
    for j in range(2, 6):
        rows[f"A{j}"] = {"IND1": 0.95}
    # 其余 4 个行业各 1 只，暴露 0.90（低于 IND1 组，故不会被优先选中）
    for i in range(2, 6):
        rows[f"B{i}"] = {f"IND{i}": 0.90}
    exposure = pd.DataFrame(rows).T.reindex(columns=score.index).fillna(0.0)

    w_rep = weights_unconstrained(score, exposure)       # §3.3
    w_cap = weights_etf_capped(score, exposure)          # §3.2
    w_free = weights_ind_free_capped(score, exposure)    # 自研

    k = int((score > 0).sum())
    exp_rep = exposure.T @ w_rep
    exp_cap = exposure.T @ w_cap
    exp_free = exposure.T @ w_free

    # 与 §3.3 的差别：不退化到单只，且受 20% 上限
    assert (w_rep > 1e-6).sum() == 1, "§3.3 应集中到单只"
    assert (w_free > 1e-6).sum() >= 5, "自研变体应被 20% 上限摊开"
    assert (w_free <= 0.2 + 1e-6).all()

    # 与 §3.2 的差别：允许单行业暴露突破 1/k（§3.2 不允许）
    assert exp_free["IND1"] > 1.0 / k + 1e-6, "自研变体应能突破 1/k 行业上限"
    assert exp_cap["IND1"] <= 1.0 / k + 1e-6, "§3.2 的 1/k 行业上限必须生效"
    assert exp_rep["IND1"] > 1.0 / k + 1e-6


def test_ind_free_capped_registered_but_flagged_as_custom():
    """自研变体已注册进 _PRESETS，且 docstring 显式声明「研报无此设定」。"""
    from src.optimizer import _PRESETS

    assert _PRESETS["ind_free_capped"] is weights_ind_free_capped
    doc = weights_ind_free_capped.__doc__ or ""
    assert "自研" in doc and "研报无此设定" in doc


def test_weights_hs300_enhanced_preset_band():
    """§3.4 预设：行业偏离带 |组合暴露 − 基准| ≤ δ。"""
    score = pd.Series({"IND1": 1.0, "IND2": 1.0})
    exposure = pd.DataFrame(
        {"IND1": [1.0, 0.0], "IND2": [0.0, 0.9]}, index=["ETF_A", "ETF_B"]
    )
    benchmark = pd.Series({"IND1": 0.5, "IND2": 0.5})
    w = weights_hs300_enhanced(score, exposure, benchmark, band=0.05)
    portfolio_exposure = exposure.T @ w
    for ind in ["IND1", "IND2"]:
        assert abs(portfolio_exposure[ind] - benchmark[ind]) <= 0.05 + 1e-6


def test_score_continuous():
    """连续打分也能求解，目标方向正确（偏向高暴露·高分的 ETF）。"""
    score = pd.Series({"IND1": 0.8, "IND2": 0.2})
    exposure = pd.DataFrame(
        {"IND1": [1.0, 0.0], "IND2": [0.0, 1.0]}, index=["ETF_A", "ETF_B"]
    )
    w = optimize_etf_weights(score, exposure, gross_mode="eq", gross_target=1.0)
    # a = [0.8, 0.2] → 全压 ETF_A
    assert w["ETF_A"] == pytest.approx(1.0, abs=1e-5)


def test_batch_optimize_lenient_nan_on_infeasible():
    """batch：逐换仓日；strict=False 时不可行日返回 NaN 行 + warning。"""
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    scores = pd.DataFrame(
        {"IND1": [1.0, 1.0], "IND2": [1.0, 1.0]}, index=dates
    )
    exposures = {
        # 第 1 日：两只纯 ETF，base(k=2, cap=0.5) 可行 → w=[0.5,0.5]
        dates[0]: pd.DataFrame(
            {"IND1": [1.0, 0.0], "IND2": [0.0, 1.0]}, index=["E1", "E2"]
        ),
        # 第 2 日：仅一只 100% IND1 的 ETF，k=2/cap=0.5/满仓 → 不可行
        dates[1]: pd.DataFrame({"IND1": [1.0], "IND2": [0.0]}, index=["E1"]),
    }
    with pytest.warns(UserWarning, match="优化失败|换仓日"):
        result = batch_optimize(scores, exposures, dates, preset="base")
    assert list(result.index) == list(dates)
    assert {"E1", "E2"} <= set(result.columns)
    assert result.loc[dates[0]].sum() == pytest.approx(1.0, abs=1e-5)
    assert result.loc[dates[1]].isna().all()  # 不可行 → NaN 行


def test_batch_optimize_strict_raises():
    """batch strict=True：第一处不可行即中断。"""
    dates = pd.to_datetime(["2024-01-31"])
    scores = pd.DataFrame({"IND1": [1.0], "IND2": [1.0]}, index=dates)
    exposures = {dates[0]: pd.DataFrame({"IND1": [1.0], "IND2": [0.0]}, index=["E1"])}
    with pytest.raises(ValueError):
        batch_optimize(scores, exposures, dates, preset="base", strict=True)


def test_optimizer_truncates_numerical_residue():
    """内点法在宽池上返回的微小分量必须被截断为 0——否则会被撮合成真实交易。

    实测：研报口径宽池下 unconstrained 的中心解产生 12934 笔成交，
    而同策略顶点解仅约 45 笔，差额几乎全是残渣。
    """
    n = 60
    score = pd.Series({"IND1": 1.0})
    # 60 只 ETF 对同一行业暴露递减，LP 内点法易返回带微小分量的中心解
    exposure = pd.DataFrame(
        {"IND1": np.linspace(0.9, 0.1, n)},
        index=[f"ETF{i:03d}" for i in range(n)],
    )
    w = optimize_etf_weights(score, exposure, w_max=0.2, gross_target=1.0,
                             gross_mode="eq", industry_cap=None)
    residue = w[(w.abs() > 0) & (w.abs() < 1e-6)]
    assert residue.empty, f"仍有 {len(residue)} 个残渣分量: {residue.to_dict()}"


def test_optimizer_truncation_does_not_renormalize():
    """只截断不重归一化——重归一化可能把某些权重推过 w_max。

    截断损失上界 = n × residue_tol，n=60 时约 6e-5，故 Σw 仍应贴近 gross_target。
    """
    n = 60
    score = pd.Series({"IND1": 1.0})
    exposure = pd.DataFrame(
        {"IND1": np.linspace(0.9, 0.1, n)},
        index=[f"ETF{i:03d}" for i in range(n)],
    )
    w = optimize_etf_weights(score, exposure, w_max=0.2, gross_target=1.0,
                             gross_mode="eq", industry_cap=None)
    assert (w <= 0.2 + 1e-9).all(), "截断后不得有权重突破 w_max"
    assert w.sum() <= 1.0 + 1e-9
    assert w.sum() >= 1.0 - n * 1e-6 - 1e-9


def test_optimizer_residue_tol_zero_disables_truncation():
    """residue_tol=0 保留逃生舱：不做任何截断，返回求解器原始解。

    用 Task 2 实测稳定产生 55 个残渣分量的 n=60 linspace 宽池输入——2 标的
    小输入求解器本就返回顶点解、无残渣可放行，逃生舱验证形同虚设。
    """
    n = 60
    score = pd.Series({"IND1": 1.0})
    exposure = pd.DataFrame(
        {"IND1": np.linspace(0.9, 0.1, n)},
        index=[f"ETF{i:03d}" for i in range(n)],
    )
    w0 = optimize_etf_weights(score, exposure, w_max=0.2, gross_target=1.0,
                              gross_mode="eq", industry_cap=None, residue_tol=0.0)
    assert w0.notna().all()
    assert w0.sum() == pytest.approx(1.0, abs=1e-6)
    # 未截断的原始解确含残渣分量（实测 55 个）——证明逃生舱真的放行了中心解
    assert ((w0.abs() > 0) & (w0.abs() < 1e-6)).any()


# ---------------------------------------------------------------------------
# §3.11 ②：零目标价值的 ETF 不得持仓；等式仓位按可投 ETF 封顶
# ---------------------------------------------------------------------------
import warnings as _w


def _exp(rows: dict, industries=("I1", "I2")) -> pd.DataFrame:
    return pd.DataFrame(rows, index=list(industries)).T.astype(float)


_ETF_CAPPED = dict(w_max=0.2, gross_target=0.4, gross_mode="eq", industry_cap=0.5)  # k=2


def test_zero_objective_etfs_get_zero_weight_and_gross_capped():
    """选中 I1、I2；只有 A 对 I1 有暴露，I2 整列为 0 → a=[0.7,0,0]。
    旧实现：Σw==0.4 等式迫使 0.2 的「无目标价值」权重被撒到 B/C（复现 2025-06-30 整列 0.02）。
    要求：A=0.2，B=C=0，并 warn 仓位由 0.4 降为 0.2。
    让它变红：不排除零目标 ETF / 不封顶等式仓位。"""
    E = _exp({"A": [0.7, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0]})
    score = pd.Series({"I1": 1.0, "I2": 1.0})
    with pytest.warns(UserWarning, match="仓位.*降"):
        w = optimize_etf_weights(score, E, **_ETF_CAPPED)
    assert w["A"] == pytest.approx(0.2, abs=1e-6)
    assert w["B"] == pytest.approx(0.0, abs=1e-6) and w["C"] == pytest.approx(0.0, abs=1e-6)


def test_all_zero_objective_returns_flat_with_warning():
    """选中的 I2 在 E 里是全零列（无 ETF 承接）→ 空仓 + warn。
    旧实现：selected.any() 为真 → 目标恒 0 → 任意可行解。让它变红：删掉可投守卫。"""
    E = _exp({"A": [0.7, 0.0], "B": [0.3, 0.0]})
    score = pd.Series({"I2": 1.0})
    with pytest.warns(UserWarning, match="空仓|无.*承接"):
        w = optimize_etf_weights(score, E, **_ETF_CAPPED)
    assert (w.abs() < 1e-9).all()


def test_nonzero_objective_unchanged_and_silent():
    """双侧：三只 ETF 都有目标价值时，行为与旧实现相同、不发本修复的 warning。
    让它变红：守卫误伤正常路径（错误地降仓或告警）。"""
    E = _exp({"A": [0.7, 0.0], "B": [0.0, 0.6], "C": [0.3, 0.0]})
    score = pd.Series({"I1": 1.0, "I2": 1.0})
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        w = optimize_etf_weights(score, E, **_ETF_CAPPED)
    assert not [r for r in rec if "仓位" in str(r.message) or "空仓" in str(r.message)]
    assert w.sum() == pytest.approx(0.4, abs=1e-6)
    assert (w <= 0.2 + 1e-6).all() and (w > 0).sum() >= 2


def test_benchmark_mode_keeps_zero_objective_etfs_for_tracking():
    """双侧：hs300_enhanced 路径（benchmark + active_band）里零目标 ETF 用来跟踪基准，
    不得被排除——B 对选中行业 I1 零暴露，但偏离带迫使它承接权重。
    让它变红：把可投子集逻辑也套到基准跟踪模式。"""
    E = _exp({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    score = pd.Series({"I1": 1.0})
    bench = pd.Series({"I1": 0.5, "I2": 0.5})
    w = optimize_etf_weights(score, E, w_max=1.0, gross_target=1.0, gross_mode="eq",
                             benchmark=bench, active_band=0.1)
    assert w["B"] > 0.3   # 满仓 1 − A(≤0.6) → B 至少 0.4

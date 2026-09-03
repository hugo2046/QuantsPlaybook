"""
RRG ETF 组合优化层（CVXPY 线性规划，纯函数）。

按西部证券 2026-05-25《指数化配置系列研究（6）》研报 §3.1–3.4 复现。
- 公式参考：spec docs/superpowers/specs/2026-05-28-rrg-etf-optimizer-design.md
- 数据契约：score(Series, index=industry) + exposure(DataFrame, index=etf,
  columns=industry, 值=ETF 对行业的持仓权重 w_{f,i})；输出 ETF 权重 Series。
- 研报四变体共用同一线性目标 ``max a^T w``（a = exposure @ score），仅约束集不同。
- 另有 1 个**自研变体** ``ind_free_capped``（研报无此设定，见其 docstring）。
- 纯函数：不碰 I/O，暴露矩阵 E 由调用者/数据层传入。
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable

import cvxpy as cp
import numpy as np
import pandas as pd


def optimize_etf_weights(
    score: pd.Series,
    exposure: pd.DataFrame,
    *,
    w_max: float = 1.0,
    gross_target: float = 1.0,
    gross_mode: str = "eq",
    industry_cap: float | None = None,
    benchmark: pd.Series | None = None,
    active_band: float | None = None,
    residue_tol: float = 1e-6,
    on_infeasible: str = "raise",
) -> pd.Series:
    """
    求解单换仓日的 ETF 配置权重 LP。

    :param score: 行业打分，index=industry，值 ∈ {0,1}（研报）或连续分。
    :param exposure: ETF 行业暴露矩阵，index=etf, columns=industry, 值=w_{f,i}。
    :param w_max: 单个 ETF 权重上限。
    :param gross_target: 仓位约束右端值。**等式模式下可能被封顶**：见 note。
    :param gross_mode: ``"eq"`` → Σw=gross_target；``"le"`` → Σw≤gross_target。
    :param industry_cap: 每个选中行业的组合暴露上限（如 1/k）；None=不约束。
    :param benchmark: §3.4 基准行业权重，index=industry；与 active_band 同用。
    :param active_band: §3.4 行业偏离阈值 δ；要求选中行业 |组合暴露−基准| ≤ δ。
    :param on_infeasible: 不可行/求解失败时——``"raise"`` 抛 ValueError（含
        solver status）；``"nan"`` 返回全 NaN Series（供 batch 宽容模式用）。
    :param residue_tol: 绝对值小于该阈值的权重分量截断为 0，默认 ``1e-6``；
        ``0`` 关闭截断。内点法（cvxpy 默认 CLARABEL）在宽候选池上会返回大量
        接近零但非零的分量，这些残渣会被 ``run_vbt_backtest`` 当作真实持仓变动
        撮合并计手续费（实测宽池下 12934 笔 vs 顶点解约 45 笔）。**只截断不
        重归一化**——重归一化可能突破 ``w_max``；截断后 ``Σw`` 相对
        ``gross_target`` 的偏差至多 ``n × residue_tol``（截断负残渣使 Σw 增加、
        正残渣使其减少，方向不定）。
    :returns: ETF 配置权重 Series，index=exposure.index。
    :raises ValueError: ``on_infeasible="raise"`` 且 LP 不可行/未求出最优解。
    :note:
        **可投集与等式仓位封顶（2026-09-02，偏离分析 §3.11 ②）**：``active_band``
        为 None 时只对目标系数 ``a_f = (E·s)_f > 0`` 的 ETF 建变量（零目标 ETF 对
        ``max aᵀw`` 没有价值，留在 LP 里只会让求解器把多出仓位均匀撒上去）；
        若 ``gross_mode="eq"`` 且 ``w_max·n_elig < gross_target ≤ w_max·n``，
        ``gross_target`` 降为 ``w_max·n_elig`` 并 ``UserWarning``。全体 ETF 都装不下
        （``w_max·n < gross_target``）**不封顶**，仍按 ``on_infeasible`` 处理——
        这不是 optimizer spec 2026-05-28 §10 否决的「自动放松不可行约束」，
        那条 ``relax`` 档仍未实现。可投集为空 → warn + 全零（空仓持币）。
        ``active_band`` 模式保留全变量集（零目标 ETF 用于拟合基准）。
    """
    etfs = exposure.index
    n = len(etfs)
    E = exposure.to_numpy()  # [n_etf × n_industry]
    # 目标系数 a_f = ETF f 对各行业暴露 · 行业打分（行业对齐到 exposure.columns）
    s = score.reindex(exposure.columns).fillna(0.0).to_numpy()
    a = E @ s
    selected = s > 0  # 选中行业掩码

    # 当日无选中行业 → 空仓（全 0），不强行满仓
    if not selected.any():
        return pd.Series(0.0, index=etfs)

    # 2026-09-02（偏离分析 §3.11 ②）：目标系数 a_f = 0 的 ETF 对 max aᵀw 没有任何价值，
    # 留在 LP 里只会制造平坦方向——等式仓位一旦超过「有价值 ETF × w_max」，多出的仓位
    # 会被求解器均匀撒到这些零价值 ETF 上（实测 2025-06-30：1 只 0.2 + 22 只 0.004，
    # 归一后 0.50 + 22 × 0.02）。故：只对 a_f > 0 的 ETF 建变量，等式仓位按可投上限封顶。
    # 例外：基准跟踪模式（active_band）里零目标 ETF 用于拟合基准，保留全变量集。
    eligible = a > 0
    if active_band is None:
        n_elig = int(eligible.sum())
        if n_elig == 0:
            warnings.warn(
                "选中行业在暴露矩阵中无任何 ETF 承接（目标系数全 0）→ 空仓持币",
                UserWarning, stacklevel=2,
            )
            return pd.Series(0.0, index=etfs)
        cap_total = w_max * n_elig
        # 只在「可投 ETF 装不下、但全体 ETF 装得下」时封顶——这正是旧实现会把多出仓位
        # 撒到零目标 ETF 的情形。全体都装不下（w_max·n < gross_target）仍交给 LP 判不可行
        # → raise / NaN，保住「绝不静默返回诡异权重」的契约（test_infeasible_raises_and_nan）。
        if gross_mode == "eq" and cap_total < gross_target - 1e-12 <= w_max * n:
            warnings.warn(
                f"可投 ETF 仅 {n_elig} 只（单只上限 {w_max}），仓位由 {gross_target:.3f} "
                f"降为 {cap_total:.3f}——多出的仓位不会撒到零目标 ETF 上",
                UserWarning, stacklevel=2,
            )
            gross_target = cap_total
        E = E[eligible]
        a = a[eligible]
        etfs_var = etfs[eligible]
    else:
        etfs_var = etfs
    n = len(etfs_var)

    w = cp.Variable(n)
    constraints = [w >= 0, w <= w_max]
    gross = cp.sum(w)
    if gross_mode == "eq":
        constraints.append(gross == gross_target)
    else:
        constraints.append(gross <= gross_target)
    if industry_cap is not None:
        # 组合在每个选中行业的暴露 (E.T @ w)[selected] ≤ industry_cap
        constraints.append(E[:, selected].T @ w <= industry_cap)
    if active_band is not None:
        # §3.4 行业偏离带：|组合暴露 − 基准| ≤ δ（选中行业）
        b = benchmark.reindex(exposure.columns).fillna(0.0).to_numpy()
        constraints.append(
            cp.abs(E[:, selected].T @ w - b[selected]) <= active_band
        )

    problem = cp.Problem(cp.Maximize(a @ w), constraints)
    problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate"):
        if on_infeasible == "nan":
            return pd.Series(np.nan, index=etfs)   # 全 etfs（含被排除的零目标 ETF）
        raise ValueError(
            f"ETF 权重 LP 求解失败 (solver status: {problem.status})；"
            f"检查约束是否互斥不可行（如 w_max·n < gross_target）。"
        )
    out = pd.Series(np.asarray(w.value).ravel(), index=etfs_var).reindex(etfs, fill_value=0.0)
    if residue_tol > 0:
        # 只截断不重归一化：重归一化需把被截断的质量重新分配，而分配后可能
        # 突破 w_max，引入新的约束违反。损失上界 n × residue_tol（n=447 时约 4.5e-4）。
        out[out.abs() < residue_tol] = 0.0
    return out


# ——— 研报四变体预设（内部按 k=(score>0).sum() 推导约束参数）———

def weights_base(score: pd.Series, exposure: pd.DataFrame) -> pd.Series:
    """§3.1：满仓 Σw=1，单 ETF ≤1，每行业暴露 ≤1/k。"""
    k = int((score > 0).sum())
    cap = 1.0 / k if k > 0 else None
    return optimize_etf_weights(
        score, exposure, w_max=1.0, gross_target=1.0, gross_mode="eq",
        industry_cap=cap,
    )


def weights_etf_capped(score: pd.Series, exposure: pd.DataFrame) -> pd.Series:
    """§3.2：单 ETF ≤0.2，仓位 Σw=min(1, 0.2k)，每行业暴露 ≤1/k。

    :note:
        k 小而选中行业只有很少 ETF 承接时，等式仓位会被 :func:`optimize_etf_weights`
        封顶为 ``0.2·n_elig``（并 warn）而非撒到零目标 ETF；2026-09-02 前的所有
        ``etf_capped`` 回测又经 ``mask_listed`` 归一成满仓，数字已作废（偏离分析 §3.11）。
    """
    k = int((score > 0).sum())
    cap = 1.0 / k if k > 0 else None
    return optimize_etf_weights(
        score, exposure, w_max=0.2, gross_target=min(1.0, 0.2 * k),
        gross_mode="eq", industry_cap=cap,
    )


def weights_unconstrained(score: pd.Series, exposure: pd.DataFrame) -> pd.Series:
    """§3.3：无行业约束，单 ETF ≤1，仓位 Σw≤1（可不满仓）。

    :note:
        **退化到 1–2 只 ETF 是研报的设计意图，不是 bug**（2026-08-29 经 NotebookLM
        调取研报原文确认）。研报原文约束即 ``0<w_f≤1`` + ``Σw_f≤1``——无单标的上限、
        无行业上限、允许留现金；正文明述该变体「多数时期仅持有 1 到 2 只 ETF，极大
        提高了持仓集中度」，定位就是「高弹性高波变体」。

        线性目标 ``max aᵀw`` 在这组约束下最优解必落在单纯形顶点，故全仓压单只是
        LP 的必然结果。**不要**为了「让它看起来合理」而私自补 ``w_max``——那会变成
        另一个策略（若需要该策略，用 :func:`weights_ind_free_capped`）。
    """
    return optimize_etf_weights(
        score, exposure, w_max=1.0, gross_target=1.0, gross_mode="le",
        industry_cap=None,
    )


def weights_hs300_enhanced(
    score: pd.Series,
    exposure: pd.DataFrame,
    benchmark: pd.Series,
    band: float = 0.05,
) -> pd.Series:
    """§3.4：满仓 Σw=1，行业偏离带 |组合暴露−基准| ≤ band（无 1/k 行业上限）。"""
    return optimize_etf_weights(
        score, exposure, w_max=1.0, gross_target=1.0, gross_mode="eq",
        benchmark=benchmark, active_band=band,
    )


# ——— 自研变体（研报无此设定，勿与上面四个复现变体混淆）———

def weights_ind_free_capped(score: pd.Series, exposure: pd.DataFrame) -> pd.Series:
    """**自研变体**：删行业上限，但保留单 ETF ≤0.2（研报无此设定）。

    介于 §3.2 与 §3.3 之间——像 §3.3 那样放开行业权重上限以释放弹性，又像 §3.2
    那样用单标的上限阻止 LP 退化到全仓单只，实测持仓约 4–5 只。

    :warning:
        **这不是研报任何一节的复现。** 研报 §3.3 明确不含单标的上限（见
        :func:`weights_unconstrained`）。本变体是 2026-08-29 对照研报图 16 时自研
        的中间形态，引入动机与实测见
        ``docs/RRG复现偏离分析_20260618.md`` §3.7。做「复现保真度」相关论证时
        **不得**用本变体的数字冒充研报口径。

    :param score: 行业打分，index=industry。
    :param exposure: ETF 行业暴露矩阵，index=etf, columns=industry。
    :returns: ETF 配置权重 Series。
    """
    k = int((score > 0).sum())
    return optimize_etf_weights(
        score, exposure, w_max=0.2, gross_target=min(1.0, 0.2 * k),
        gross_mode="eq", industry_cap=None,
    )


_PRESETS = {
    "base": weights_base,
    "etf_capped": weights_etf_capped,
    "unconstrained": weights_unconstrained,
    "hs300_enhanced": weights_hs300_enhanced,
    # 自研，非研报变体
    "ind_free_capped": weights_ind_free_capped,
}


def batch_optimize(
    scores: pd.DataFrame,
    exposures: dict[pd.Timestamp, pd.DataFrame],
    dates: Iterable[pd.Timestamp],
    *,
    preset: str = "base",
    strict: bool = False,
    **preset_kwargs,
) -> pd.DataFrame:
    """
    按调用者给定的换仓日逐日求解 ETF 权重（不内置任何「月末」逻辑）。

    :param scores: 行业打分宽表，index=date, columns=industry。
    :param exposures: ``{换仓日: 暴露矩阵 DataFrame[etf × industry]}``（时变持仓）。
    :param dates: 调用者指定的换仓日序列。
    :param preset: ``"base"|"etf_capped"|"unconstrained"|"hs300_enhanced"``。
    :param strict: True→任一换仓日不可行即抛错中断；False→该日返回 NaN 行 + warning。
    :param preset_kwargs: 透传给预设（如 hs300_enhanced 的 ``benchmark`` / ``band``）。
    :returns: ETF 权重宽表，index=dates，columns=各日 ETF 并集。
    """
    fn = _PRESETS[preset]
    rows: dict[pd.Timestamp, pd.Series] = {}
    for d in dates:
        exposure = exposures[d]
        try:
            rows[d] = fn(scores.loc[d], exposure, **preset_kwargs)
        except ValueError as exc:
            if strict:
                raise
            warnings.warn(
                f"换仓日 {d} 优化失败，该日返回 NaN：{exc}",
                UserWarning,
                stacklevel=2,
            )
            rows[d] = pd.Series(np.nan, index=exposure.index)
    return pd.DataFrame(rows).T

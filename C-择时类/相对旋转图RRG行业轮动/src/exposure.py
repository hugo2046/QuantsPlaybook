"""
ETF→申万一级行业暴露矩阵 E 的构建层（真实持仓驱动版）。

把 TuShare ``fund_portfolio`` 真实持仓 + 申万成分股 membership 聚合成
**连续行业权重**，替代 ``script/build_etf_sw_mapping.py`` 关键词法的 0/1 近似，
作为 ``optimizer`` 暴露矩阵 ``E`` 的精确入口。

- ``aggregate_industry_weights``：纯函数，持仓按申万一级行业聚合成权重 long 表
  （默认 ``denominator="nav"``：``stk_mkv_ratio``/100，即**占基金净值比**；
  ``"disclosed"`` 为已废弃的缺陷口径，仅供复核历史结果）。
- ``ETFExposureBuilder``：把「取持仓 → 取 membership → 聚合」串成可内存缓存、
  可 ``reset`` 的单一类，避免重复慢取数。

PIT 责任分层：本层只在 ``anchor`` 时点（默认持仓报告期 ``end_date``）确定个股行业归类；
「``ann_date ≤ 换仓日`` 才可见」的前视防护在 ``backtest.build_exposure_panel`` 阶段把关。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.store import RRGStore

__all__ = ["aggregate_industry_weights", "reconcile_mapping", "ETFExposureBuilder"]


def aggregate_industry_weights(
    portfolio: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    anchor: str = "end_date",
    denominator: str = "nav",
) -> pd.DataFrame:
    """
    持仓个股按申万一级行业聚合成 ETF 行业权重（long 表）。

    :param portfolio: ``data_provider.fetch_etf_portfolio`` / ``store.RRGStore.get_portfolio``
        输出（``_PORTFOLIO_COLS`` 列契约），至少含
        ``code`` (ETF) / ``symbol`` (股) / ``ann_date`` / ``end_date`` / ``mkv``。
    :param membership: 个股→申万一级行业（时变），含 ``code`` (股) / ``trade_date`` /
        ``industry_code``（即 ``store.RRGStore.get_membership`` 返回列）。
    :param anchor: 个股行业归类的时点锚，``'end_date'``（默认，与 mkv 快照同期，自洽）
        或 ``'ann_date'``（公告日归类）。两者均 PIT 安全。
    :param denominator: 权重分母口径，``"nav"``（默认）/ ``"disclosed"``。
        ``"nav"`` 依赖 ``portfolio`` 的 ``stk_mkv_ratio`` 列（占基金净值**百分比**）。
    :returns: long ``DataFrame``，列 ``[code, end_date, ann_date, sw_code, weight]``：

        - ``sw_code`` 为历史列名，实际承载**当前 classification 的一级行业代码**
          （``sw`` → ``801xxx.SI``，``zx`` → ``CI005xxx.CI``）。改名会波及 backtest.py
          （28 处）与 plotting.py（10 处）近 40 处而收益为零，故保留。
        - ``weight``：由 ``denominator`` 决定——
          ``"nav"``（默认）= 该行业持仓的 ``stk_mkv_ratio`` 之和 / 100，即**占基金净值比**；
          ``"disclosed"`` = 该行业 mkv / 该 (ETF, end_date) 全部**已披露**持仓 mkv。
        - ⚠️ ``"disclosed"`` 是**已知缺陷口径**，仅供复核历史结果：它让只披露前十大重仓的
          ETF（实测 23.3% 的报告期披露不足净值 50%）权重虚高到 1.0，与全披露的纯行业 ETF
          无法区分，导致 LP 最优面退化。详见
          ``docs/superpowers/specs/2026-08-30-nav-denominator-and-lp-residue-design.md``。
        - ``"nav"`` 口径下 ``Σ_i weight_i ≈ 披露覆盖率``，**不恒 ≤ 1**（实测 15.36% 的报告期
          略超 1，最大 1.0165，系披露值四舍五入）；**不要**归一化。
        - ``ann_date`` 取每 ``(code, end_date)`` 内最大公告日（最后一次更正落地 =
          整张持仓完全可见的时点），保证下游 ``merge_asof`` 每报告期单一可见日。
    :note:
        行业归类用 ``merge_asof(direction='backward')`` 取「≤ anchor 的最近 ``trade_date``」，
        天然兼容停牌 / 季末非交易日（取前一交易日的分类）。
    """
    if denominator not in ("nav", "disclosed"):
        raise ValueError(
            f"denominator 必须是 'nav' / 'disclosed'，收到 {denominator!r}"
        )
    # 缺列校验前置：在 sort+merge 之前拦截，错误路径上省一次无谓计算
    if denominator == "nav" and "stk_mkv_ratio" not in portfolio.columns:
        raise ValueError(
            "portfolio 缺 stk_mkv_ratio 列——nav 口径依赖它（占基金净值百分比）。"
            "若确要用旧口径请显式传 denominator='disclosed'。"
        )

    left = portfolio.rename(columns={"symbol": "_sym"}).sort_values(anchor)
    right = (
        membership.rename(columns={"code": "_sym"})
        .sort_values("trade_date")
    )

    joined = pd.merge_asof(
        left,
        right[["_sym", "trade_date", "industry_code"]],
        left_on=anchor,
        right_on="trade_date",
        by="_sym",
        direction="backward",
    )

    if denominator == "nav":
        # 全空的 (code, end_date) 整期丢弃：混口径比缺几行危险得多
        ratio_ok = joined.groupby(["code", "end_date"])["stk_mkv_ratio"].transform(
            lambda s: s.notna().any()
        )
        n_dropped = int(
            joined.loc[~ratio_ok, ["code", "end_date"]].drop_duplicates().shape[0]
        )
        if n_dropped:
            warnings.warn(
                f"{n_dropped} 个 (ETF, 报告期) 的 stk_mkv_ratio 全为空，"
                f"在 nav 口径下已整期丢弃（不回退到 disclosed，避免混口径不可比）",
                UserWarning,
                stacklevel=2,
            )
        joined = joined[ratio_ok]

    # 分母：每 (ETF, 报告期) 全持仓 mkv（含无行业持仓）；ann_date 取期内最大公告日
    grp_period = joined.groupby(["code", "end_date"])
    joined = joined.assign(
        _total=grp_period["mkv"].transform("sum"),
        _ann_eff=grp_period["ann_date"].transform("max"),
    )

    covered = joined.dropna(subset=["industry_code"])
    if denominator == "nav":
        out = (
            covered.groupby(["code", "end_date", "industry_code"], as_index=False)
            .agg(_numer=("stk_mkv_ratio", "sum"), ann_date=("_ann_eff", "first"))
            .assign(weight=lambda d: d["_numer"] / 100.0)
            .rename(columns={"industry_code": "sw_code"})
        )
    else:
        out = (
            covered.groupby(["code", "end_date", "industry_code"], as_index=False)
            .agg(_numer=("mkv", "sum"), ann_date=("_ann_eff", "first"),
                 _total=("_total", "first"))
            .assign(weight=lambda d: d["_numer"] / d["_total"])
            .rename(columns={"industry_code": "sw_code"})
        )
    return out[["code", "end_date", "ann_date", "sw_code", "weight"]]


def reconcile_mapping(
    industry_weights: pd.DataFrame,
    keyword_mapping: pd.DataFrame,
    *,
    min_weight: float = 0.5,
    code2name: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    用真实持仓**主导行业**校正关键词 ``sw_code`` → drop-in 校正映射（最近报告期快照）。

    .. deprecated:: legacy-for-A/B
        **静态校正路径**——用**最近报告期**持仓把 ``sw_code`` 冻成单张快照，对历史
        换仓日带前视。生产路径请用 ``backtest.build_dynamic_representative_panel``
        （每换仓日按当期 PIT 持仓动态归类，无前视）。本函数仅保留供静态 vs 动态 A/B 对比。

    持仓置信（主导行业占全持仓 NAV ≥ ``min_weight``）时以持仓为准覆盖关键词，
    低置信 / 无持仓则回退关键词。**不改候选池成员**，仅校正其 ``sw_code`` 分类
    （scope a）；下游 ``filter_top_per_industry`` 按 ``sw_code`` 分组即随之改变。

    :param industry_weights: ``aggregate_industry_weights`` 产出 long 表
        ``[code, end_date, ann_date, sw_code, weight]``。
    :param keyword_mapping: ``etf_sw_mapping.csv`` 关键词映射（含 ``etf_code`` /
        ``sw_code`` / ``sw_name`` / ``track_code`` / ``match_status`` 等全列）。
    :param min_weight: 主导行业置信阈值（默认 0.5）；``dominant_weight < min_weight``
        或无持仓 → 回退关键词。
    :param code2name: ``sw_code → 中文名`` 映射，用于按校正后 ``sw_code`` 重填
        ``sw_name``；``None`` 退化用 ``keyword_mapping`` 自身 ``(sw_code, sw_name)``
        对（持仓新引入的行业名可能留空，脚本应传完整 ``code2name``）。
    :returns: ``keyword_mapping`` 的副本——``sw_code`` ← 校正后、``sw_name`` ← 对应名，
        并追加审计列 ``sw_code_keyword`` / ``sw_code_holdings`` /
        ``dominant_weight`` / ``source`` (``holdings`` | ``keyword``) /
        ``agreement`` (持仓主导 == 关键词)。行数 / 其它列原样保留（drop-in）。
    """
    # 1) 每 (code, end_date) 权重 argmax → 主导行业
    idx = industry_weights.groupby(["code", "end_date"])["weight"].idxmax()
    dom = (
        industry_weights.loc[idx, ["code", "end_date", "sw_code", "weight"]]
        .rename(columns={"sw_code": "dominant_sw", "weight": "dominant_weight"})
    )
    # 2) 每只 ETF 取最近报告期快照
    dom_latest = (
        dom.sort_values("end_date")
        .groupby("code", as_index=False)
        .tail(1)[["code", "dominant_sw", "dominant_weight"]]
        .rename(columns={"code": "etf_code"})
    )

    # 3) merge 关键词 + 校正（置信→持仓覆盖；低置信/无持仓→回退关键词）
    out = keyword_mapping.copy().rename(columns={"sw_code": "sw_code_keyword"})
    out = out.merge(dom_latest, on="etf_code", how="left")
    confident = out["dominant_weight"] >= min_weight        # NaN >= x → False
    out["sw_code"] = out["dominant_sw"].where(confident, out["sw_code_keyword"])
    out["sw_code_holdings"] = out["dominant_sw"]
    out["source"] = confident.map({True: "holdings", False: "keyword"})
    out["agreement"] = out["dominant_sw"] == out["sw_code_keyword"]

    # sw_name 按校正后 sw_code 重填（code2name 优先，否则用关键词自身映射）
    name_map = (
        dict(code2name)
        if code2name is not None
        else keyword_mapping.dropna(subset=["sw_code"])
        .drop_duplicates("sw_code")
        .set_index("sw_code")["sw_name"]
        .to_dict()
    )
    out["sw_name"] = out["sw_code"].map(name_map)

    return out.drop(columns=["dominant_sw"])


class ETFExposureBuilder:
    """
    真实持仓 → ETF 行业权重的可缓存编排器。

    把「取持仓（``RRGStore``）→ 取 membership（``RRGStore.get_membership``）→
    聚合（``aggregate_industry_weights``）」三步串成单一入口，避免重复慢取数。

    缓存策略：

    - 持仓由 ``RRGStore``（DuckDB）增量缓存：coverage 记录已取季度，miss 才远程抓。
    - membership 由 ``RRGStore``（DuckDB）水位增量缓存，不再落地 parquet。
    - 派生结果（行业权重）**不落缓存**，按需重算（亚秒级）。

    :param codes: ETF ``ts_code`` 列表（带交易所后缀）。
    :param start_date: 回测 / 信号起始日（``'YYYY-MM-DD'``）。
    :param end_date: 结束日。
    :param cache_dir: **已废弃，保留仅为兼容旧调用方，不再使用**。membership 现由
        ``RRGStore`` 管理，此参数无任何效果。
    :param lookback_periods: 报告期起点在 ``start_date`` 前多取几个季度，保证回测首
        换仓日有已披露持仓可 PIT（默认 2）。
    :param membership_anchor: 个股行业归类时点，``'end_date'``（默认）/ ``'ann_date'``。
    :param classification: 个股行业归类体系，``"sw"``（申万一级，默认）/ ``"zx"``（中信一级）。
        信号链（``RRGSignalGenerator``）与暴露链（``ETFExposureBuilder``）必须使用同一
        ``classification``，否则 ``optimizer`` 的行业打分与暴露矩阵列无法对齐，
        会静默退化为空仓。
    :param denominator: 行业权重分母口径，``"nav"``（默认，占基金净值比）/
        ``"disclosed"``（旧缺陷口径，仅供 A/B 复核）。
    :param store: ``RRGStore`` 实例；``None`` 时用默认路径构造。
    """

    def __init__(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        *,
        cache_dir: str | Path = "data/etf_exposure",  # 已废弃，保留签名兼容性
        lookback_periods: int = 2,
        membership_anchor: str = "end_date",
        classification: str = "sw",
        denominator: str = "nav",
        store: RRGStore | None = None,
    ) -> None:
        self.codes = sorted(codes)
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir  # 废弃 no-op：持仓+成分均由 RRGStore 管，仅保留字段兼容旧调用方
        self.lookback_periods = lookback_periods
        self.membership_anchor = membership_anchor
        self.classification = classification
        self.denominator = denominator
        self._owns_store = store is None
        self.store = store if store is not None else RRGStore()

        # 报告期起点：start_date 前移 lookback_periods 个季度
        self._period_start = (
            pd.Timestamp(start_date) - pd.DateOffset(months=3 * lookback_periods)
        ).strftime("%Y-%m-%d")

        # 内存缓存
        self._portfolio: pd.DataFrame | None = None
        self._membership: pd.DataFrame | None = None
        self._industry_weights: pd.DataFrame | None = None

    # ── 三阶段 ──────────────────────────────────────────────────────────
    def load_portfolio(self, *, refresh: bool = False) -> pd.DataFrame:
        """阶段①：从 RRGStore 增量取 ETF 季度持仓（前移 lookback_periods 个季度）。"""
        if self._portfolio is not None and not refresh:
            return self._portfolio
        self._portfolio = self.store.get_portfolio(
            self.codes, self._period_start, self.end_date, refresh=refresh
        )
        return self._portfolio

    def load_membership(self, *, refresh: bool = False) -> pd.DataFrame:
        """阶段②：从 RRGStore 取全市场申万一级成分（universe 无关、水位增量）。"""
        if self._membership is not None and not refresh:
            return self._membership
        self._membership = self.store.get_membership(
            self._period_start, self.end_date, refresh=refresh,
            classification=self.classification,
        )
        return self._membership

    def build_industry_weights(self, *, refresh: bool = False) -> pd.DataFrame:
        """阶段③：聚合成行业权重 long 表（``[code, end_date, ann_date, sw_code, weight]``）。

        派生结果不落缓存——原始持仓/成分均由 store（DuckDB）缓存，本步按需重算（亚秒级）。
        """
        portfolio = self.load_portfolio(refresh=refresh)
        membership = self.load_membership(refresh=refresh)
        self._industry_weights = aggregate_industry_weights(
            portfolio, membership, anchor=self.membership_anchor,
            denominator=self.denominator,
        )
        return self._industry_weights

    # ── 复位 ────────────────────────────────────────────────────────────
    def reset(self, *, clear_cache: bool = False) -> None:
        """清空内存缓存。``clear_cache`` 已无 parquet 可删（持仓+成分均由 store 管），
        保留参数仅兼容旧调用方；如需清 DuckDB 缓存请用 ``store`` 自身接口。"""
        self._portfolio = self._membership = self._industry_weights = None

    def close(self) -> None:
        """关闭本实例自建的 RRGStore（外部传入的 store 由调用方负责关闭）。"""
        if self._owns_store:
            self.store.close()

    def __enter__(self) -> "ETFExposureBuilder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

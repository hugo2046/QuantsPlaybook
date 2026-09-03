"""ETF 轮动回测层（vectorbt 包装 + 时变流动性预筛 + 上市日 mask）。

设计契约：DataFrame in / Portfolio (or DataFrame) out 的纯函数，与
``src.factor_algo`` / ``src.optimizer`` 同构。不碰 I/O——所有数据由调用方
（数据层 + 信号层 + 优化层）按管线提供。

研报 §3 复现路径：
    1. industry_weights(PIT 持仓) + ETF amount → 每月末
       ``build_dynamic_representative_panel`` 按当期主导行业 argmax 动态选代表 ETF
       （提纯门 ``min_dominant_weight`` + 每 SW 行业 ``top_n`` 流动性）。
       （legacy 静态路径 ``build_representative_panel`` + ``filter_top_per_industry``
       仍保留供 A/B 对比，但带前视、非生产口径）
    2. 代表池 → 时变暴露矩阵 ``exposures: dict[date, E_t]``
    3. ``batch_optimize`` 月末求解 → 权重表
    4. ``mask_listed``（上市日防呆）→ ``run_vbt_backtest``（vbt LP+组合）
    5. ``portfolio_stats`` 对齐研报表 13 字段

回测层固化 ``cash_sharing=True + group_by=True``——多标的共享现金、组合视角。
"""
from __future__ import annotations

import pandas as pd
import vectorbt as vbt

from src.utils import to_equal_weight

# 年化按 A 股交易日口径（252/年）：freq="D" 下 vbt 默认 year_freq="365 days" 会把
# ~243 个交易日 bar 当作 243/365 年，年化收益/波动/夏普/Calmar 被高估约 1.2–1.5x。
vbt.settings.returns["year_freq"] = "252 days"

__all__ = [
    "build_representative_panel",
    "filter_top_per_industry",
    "build_exposure_panel",
    "build_dynamic_representative_panel",
    "build_report_universe_panel",
    "build_overseas_share",
    "build_dynamic_etf_industry",
    "mask_listed",
    "run_vbt_backtest",
    "run_signal_backtest",
    "plot_cumulative_vs_benchmark",
    "portfolio_stats",
]


def _rolling_avg_amount(amount: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """过去 ``lookback`` 日均成交额（``min_periods=1``，新上市 ETF 不被前导 NaN 吞掉）。

    两阶段流动性预筛（``build_representative_panel`` / ``filter_top_per_industry``）
    共用此口径——单一定义，避免 ``lookback`` 在两处漂移。
    """
    return amount.rolling(lookback, min_periods=1).mean()


def build_representative_panel(
    amount: pd.DataFrame,
    mapping: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    *,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    每月末按"过去 lookback 日均成交额最高"为每个跟踪指数挑代表 ETF（研报 §3.2 第 1) 条）。

    .. deprecated:: legacy-for-A/B
        **静态映射路径**——与 ``filter_top_per_industry`` 配套的两步预筛，吃静态
        ``mapping`` 的 ``track_code`` / ``sw_code``（其中 ``sw_code`` 经
        ``reconcile_mapping`` 冻成最近报告期快照，对历史换仓日带前视）。**生产路径请用
        ``build_dynamic_representative_panel``**（当期 PIT 持仓动态归类，无前视）。
        本函数仅保留供静态 vs 动态 A/B 对比，验证动态更优后可整体退役。

    :param amount: 成交额宽表，``index=date × columns=ts_code``（单位元，
        ``fetch_etf_daily`` 返回的 ``amount``）。
    :param mapping: ETF↔SW 映射，至少含 ``etf_code`` / ``track_code`` /
        ``sw_code`` 列（剔除 ``non-industry`` 后的 matched + ambiguous 子集）。
    :param rebal_dates: 换仓日序列（如月末）。
    :param lookback: 回看天数（研报默认 20）。
    :returns: 代表面板 ``DataFrame(index=rebal_dates × columns=track_code,
        value=ts_code 或 NaN)``。NaN 表示该 track 在该日所有候选 ETF
        要么未上市要么无成交（数据完全缺失）。
    :note:
        - 代表是**时变**的——新 ETF 上市后只要进入候选池就可能成为代表；
          ``batch_optimize`` 的 ``exposures: dict`` 接口正是为此设计。
        - 上市日 > t 的 ETF 在 ``amount`` 里全 NaN → 自动排除（idxmax 跳过 NaN）。
    """
    rolled = _rolling_avg_amount(amount, lookback)
    code2track = mapping.set_index("etf_code")["track_code"]

    rows: dict[pd.Timestamp, pd.Series] = {}
    for d in rebal_dates:
        if d not in rolled.index:
            continue
        avg = rolled.loc[d]
        # 仅保留 mapping 覆盖的 ETF + 过去窗口有任意成交（非全 NaN）
        cand = avg.dropna()
        cand = cand[cand.index.isin(code2track.index)]
        if cand.empty:
            continue
        # 按 track_code 分组取 idxmax —— groupby idxmax 自动跳过全 NaN 组
        tracks = code2track.reindex(cand.index)
        rows[d] = cand.groupby(tracks).idxmax()
    return pd.DataFrame(rows).T.sort_index()


def filter_top_per_industry(
    reps_panel: pd.DataFrame,
    amount: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    top_n: int = 1,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    在代表面板上按 SW 一级行业二次预筛——每行业保留过去 lookback 日均成交额 top_n。

    .. deprecated:: legacy-for-A/B
        **静态映射路径**——与 ``build_representative_panel`` 配套。生产路径请用
        ``build_dynamic_representative_panel``，本函数仅保留供 A/B 对比。

    研报 §3.2 第 1) 条原口径是按"同一指数"挑代表（已由 ``build_representative_panel``
    完成）。本步骤是其上的工程补丁：当多个 ``track_code`` 都映射到**同一 SW 一级
    行业**时（如"中证半导"+"科创芯片"两条 track 都归 801080 电子），LP 在 0/1
    暴露下两条 track 的代表 ETF 目标系数 ``a_f`` 完全相等 → 退化平摊。把它们再
    按 SW 收一道即可消除该退化。

    :param reps_panel: ``build_representative_panel`` 输出 (rebal_dates × track_code,
        value=ts_code)。
    :param amount: 成交额宽表，与 ``build_representative_panel`` 同源——用于
        SW 内 top_n 排序。
    :param mapping: 至少含 ``etf_code`` / ``sw_code`` 列。
    :param top_n: 每个 SW 行业保留的代表 ETF 数（默认 1，最贴研报）。
    :param lookback: 与 ``build_representative_panel`` 一致的回看天数（默认 20）。
    :returns: 同 shape 的 ``reps_panel``——落选的 track 在该日置 NaN。
    :note:
        - 输出格式与 ``build_exposure_panel`` 兼容（其内部 ``row.dropna().unique()``
          自动跳过 NaN，无需修改下游）。
        - 退化抑制顺序：track 维度（``build_representative_panel``）→ SW 行业维度
          （本函数）→ LP（``batch_optimize``）。三层并用持仓数从 ~170 压到 ~30。
    """
    rolled = _rolling_avg_amount(amount, lookback)
    code2sw = mapping.set_index("etf_code")["sw_code"]

    filtered = reps_panel.copy()
    for d, row in reps_panel.iterrows():
        if d not in rolled.index:
            continue
        present = row.dropna()
        if present.empty:
            continue
        # 组装当日候选：(track, etf, sw, avg_amount)
        df = pd.DataFrame({
            "track": present.index,
            "etf": present.values,
        })
        df["sw"] = code2sw.reindex(df["etf"]).values
        df["amt"] = rolled.loc[d].reindex(df["etf"]).values
        df = df.dropna(subset=["sw", "amt"])
        if df.empty:
            continue
        # 每 SW 行业按 amt 降序取 top_n
        winners = (
            df.sort_values("amt", ascending=False)
              .groupby("sw", as_index=False)
              .head(top_n)
        )
        losers = list(set(df["track"]) - set(winners["track"]))
        if losers:
            filtered.loc[d, losers] = pd.NA
    return filtered


def _pit_latest_holdings(
    industry_weights: pd.DataFrame, date: pd.Timestamp
) -> pd.DataFrame:
    """``ann_date <= date`` 中每只 ETF 取最近报告期（``end_date`` 最大）的持仓快照。

    "选代表 ETF" 与 "算暴露 E_t" 共用此 PIT 选期规则，保证两处同期、不错位。

    :param industry_weights: ``[code, end_date, ann_date, sw_code, weight]`` long 表。
    :param date: PIT 截止日（换仓日）。
    :returns: 行子集；无任何 ``ann_date <= date`` 时返回空 DataFrame（同列）。
    """
    elig = industry_weights[industry_weights["ann_date"] <= date]
    if elig.empty:
        return elig
    latest = elig.groupby("code")["end_date"].transform("max")
    return elig[elig["end_date"] == latest]


def build_dynamic_representative_panel(
    industry_weights: pd.DataFrame,
    amount: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    *,
    lookback: int = 20,
    min_dominant_weight: float = 0.5,
    top_n: int = 1,
) -> pd.DataFrame:
    """
    每换仓日用 PIT 持仓主导行业动态选代表 ETF（取代静态 mapping 两步预筛）。

    对每个换仓日 ``t``：

    1. PIT 快照：``_pit_latest_holdings`` 取 ``ann_date <= t`` 最近报告期持仓。
    2. 主导行业 = 该 ETF 行业权重 ``argmax``；``dominant_weight`` = 该 max 权重。
    3. 提纯门：``dominant_weight >= min_dominant_weight`` 才算行业 ETF
       （宽基/债基持仓分散，主导权益占比低 → 出局）。
    4. 流动性：过去 ``lookback`` 日均成交额（``_rolling_avg_amount``，与旧路径同口径）。
    5. 按主导行业分组取 ``top_n`` 最活跃 → 当期代表。

    :param industry_weights: ``ETFExposureBuilder.build_industry_weights`` 产出
        long 表 ``[code, end_date, ann_date, sw_code, weight]``。
    :param amount: 成交额宽表 ``index=date × columns=ts_code``（``fetch_etf_daily``）。
    :param rebal_dates: 换仓日序列（如月末）。
    :param lookback: 流动性回看天数（默认 20）。
    :param min_dominant_weight: 主导行业占比下限（默认 0.5）——universe 准入 +
        分类置信门二合一；债/货 ETF 主导权益占比 ≈ 0 自然出局。
    :param top_n: 每 SW 一级行业保留的代表 ETF 数（默认 1）。
    :returns: ``DataFrame(index=rebal_dates × columns=sw_code, value=ts_code 或 NaN)``；
        ``top_n>1`` 时同行业第 2+ 只列名为 ``f"{sw_code}#{rank}"``。无可见持仓 /
        无任何合格 ETF 的换仓日不出现在 index（只要有任一行业有代表，该换仓日仍在
        index，缺代表的行业列填 NaN）。
    :note:
        - **诚实兜底**：``ann_date <= t`` 无持仓的 ETF 当日缺席（PIT 真相，不回退关键词）。
        - 主导行业平局时取 sw_code 字典序最小者（确定性）。
        - 输出兼容 ``build_exposure_panel``（其仅抽每行 ``dropna().unique()`` 的 ETF 代码，
          列名语义无关）。
    """
    rolled = _rolling_avg_amount(amount, lookback)

    rows: dict[pd.Timestamp, pd.Series] = {}
    for d in rebal_dates:
        if d not in rolled.index:
            continue
        snap = _pit_latest_holdings(industry_weights, d)
        if snap.empty:
            continue
        # 每只 ETF 主导行业（argmax 权重）；先按 sw_code 排序使平局确定性
        # （idxmax 取首个最大值出现位置，排序后"首个"= sw_code 字典序最小者）
        snap_sorted = snap.sort_values("sw_code")
        idx = snap_sorted.groupby("code")["weight"].idxmax()
        dom = snap_sorted.loc[idx, ["code", "sw_code", "weight"]]
        # 提纯门
        dom = dom[dom["weight"] >= min_dominant_weight]
        if dom.empty:
            continue
        # 附过去 lookback 日均成交额：按 code 值 reindex 取对应成交额，
        # 再 to_numpy 剥离索引避免错位；丢无成交（amt NaN）的 ETF
        dom = dom.assign(amt=rolled.loc[d].reindex(dom["code"]).to_numpy())
        dom = dom.dropna(subset=["amt"])
        if dom.empty:
            continue
        # 每 SW 行业按成交额降序取 top_n
        winners = (
            dom.sort_values("amt", ascending=False)
               .groupby("sw_code", as_index=False)
               .head(top_n)
        )
        # 列名：top 用纯 sw_code；同行业第 2+ 用 f"{sw}#{rank}"
        rank = winners.groupby("sw_code").cumcount()
        col = winners["sw_code"].where(
            rank == 0, winners["sw_code"] + "#" + rank.astype(str)
        )
        rows[d] = pd.Series(winners["code"].to_numpy(), index=col.to_numpy())

    return pd.DataFrame(rows).T.sort_index()


def build_overseas_share(
    portfolio: pd.DataFrame,
    *,
    suffixes: tuple[str, ...] = (".HK",),
) -> pd.DataFrame:
    """每 (ETF, 报告期) 的**境外持仓市值占比**，用于把境外 ETF 排除出候选池。

    :param portfolio: ``store.RRGStore.get_portfolio`` 输出，含 ``code`` (ETF) /
        ``symbol`` (标的) / ``ann_date`` / ``end_date`` / ``mkv``。
    :param suffixes: 视为境外的标的代码后缀，默认 ``(".HK",)``（港股）。
    :returns: long ``DataFrame[code, end_date, ann_date, overseas_share]``，
        ``overseas_share`` ∈ [0, 1]；``ann_date`` 取期内最大公告日（与
        :func:`build_exposure_panel` 的 PIT 口径一致）。

    :note:
        分母是该报告期**已披露持仓的 mkv 之和**，即「持仓内部的构成比例」——
        刻意不用基金净值，使本判定与
        ``exposure.aggregate_industry_weights`` 的分母口径之争解耦：无论那边取
        已披露 mkv 还是净值，「这只 ETF 主要投港股吗」的答案都不变。

        为什么要剔：港股 ETF 的持仓全是 ``.HK`` 标的，在 A 股行业成分表里查不到，
        因而算不出任何行业权重、必然被 :func:`build_exposure_panel` 丢弃。留在池子里
        只会虚高池子规模，还会白占掉一个跟踪指数的槽位（见
        ``docs/RRG复现偏离分析_20260618.md`` §3.7 的丢弃审计：128 只港股 ETF、
        27324 条持仓 100% 为 ``.HK``，占全部丢弃的 41.1%）。

        当前**没有港股 / 美股的行业分类与行情数据**，故只能剔除；待数据补齐后
        可改为按境外行业体系单独建模。

        **与 ``script/build_etf_sw_mapping.py`` 的 ``overseas`` 关键词表的关系**：
        那里按 ETF **名称**关键词剔境外，只服务已废弃的关键词映射 A/B 臂
        （``etf_sw_mapping.csv``），不在生产路径上。实测两法在 1246 只 ETF 上一致
        136 只，但关键词法**漏剔 18 只**（「沪港深互联网/创新药/科技龙头」等，港股占比
        61–91%，因表里只精确匹配了 ``沪港深300``/``沪港深500``）、**误剔 5 只**
        （``广发恒生A股电网设备``、``华宝标普中国A股红利机会`` 等，港股占比 **0%**）。
        误剔的根因是「恒生」「标普」是**指数编制机构**而非投资地域——用名字判地域必然
        两头出错。本函数按**真实持仓**判定，不受命名影响。详见
        ``docs/RRG复现偏离分析_20260618.md`` §3.7。
    """
    sym = portfolio["symbol"].astype(str)
    is_overseas = sym.str.endswith(suffixes)
    tmp = portfolio.assign(_ovs_mkv=portfolio["mkv"].where(is_overseas, 0.0))
    out = (
        tmp.groupby(["code", "end_date"], as_index=False)
        .agg(_ovs=("_ovs_mkv", "sum"), _tot=("mkv", "sum"),
             ann_date=("ann_date", "max"))
        .assign(overseas_share=lambda d: d["_ovs"] / d["_tot"].where(d["_tot"] != 0))
    )
    return out[["code", "end_date", "ann_date", "overseas_share"]]


def build_report_universe_panel(
    meta: pd.DataFrame,
    amount: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    *,
    lookback: int = 20,
    overseas_share: pd.DataFrame | None = None,
    overseas_threshold: float = 0.5,
) -> pd.DataFrame:
    """研报 §3.1 口径的 ETF 候选池：同一跟踪指数只留过去 ``lookback`` 日 ADV 最高者。

    与 :func:`build_dynamic_representative_panel` 的**根本区别**——本函数
    **不做**「每行业选代表」，也**不设**主导权重门槛：全池按跟踪指数去重后
    整体交给 LP，由优化器自行取舍。这是研报 §3.1 原文口径：

        「配置对象为已上市的所有行业和主题 ETF；若有多个 ETF 跟踪同一指数，
        只保留过去 20 日日均成交额最高的那个。」

    两种口径对持仓集中的变体影响巨大：§3.3 每期全仓压 1–2 只时，「从全池约 447 只
    里挑暴露最高的」与「从约 30 只行业代表里挑」是两个不同的策略。详见
    ``docs/RRG复现偏离分析_20260618.md`` §3.7。

    :param meta: :func:`~src.data_provider.fetch_etf_meta` 输出，**必须含
        ``index_code`` 列**（2026-08-29 起返回）。
    :param amount: ETF 日成交额宽表，``index=date × columns=ts_code``（单位元）。
    :param rebal_dates: 换仓日序列（如月末）。
    :param lookback: ADV 回看交易日数，默认 20（研报口径）。
    :param overseas_share: :func:`build_overseas_share` 输出；``None``（默认）=
        不做境外剔除，行为与本参数引入前完全一致。
    :param overseas_threshold: 境外持仓占比**严格大于**该值的 ETF 被剔出候选池，
        默认 ``0.5``。仅在 ``overseas_share`` 非 ``None`` 时生效。
    :returns: ``DataFrame(index=rebal_dates × columns=index_code, value=ts_code
        或 NaN)``。形状契约与 :func:`build_representative_panel` 一致，可直接喂
        :func:`build_exposure_panel`。
    :raises ValueError: ``meta`` 缺 ``index_code`` 列，或 ``meta`` 与 ``amount``
        无交集 ETF。

    :note:
        - **无前视**：ADV 由 ``_rolling_avg_amount``（backward rolling）算出，
          换仓日 ``d`` 那行只含 ``≤ d`` 的成交额。
        - **平局确定性**：ADV 相同时取 ``ts_code`` 字典序最小者——先按 ``ts_code``
          升序、再用 stable sort 按 ADV 降序，故结果不依赖输入顺序。
        - ADV 为 0 或 NaN 的 ETF 不入选；某指数全组无成交则该格 NaN。
        - **境外剔除发生在同指数去重之前**：否则一只港股 ETF 若是某指数的 ADV 冠军，
          先去重再剔除会让该指数当期直接空缺，而正确顺序下同指数的 A 股亚军可以递补。
        - 境外判定同样 **PIT**：只用 ``ann_date <= d`` 中最近报告期的占比；该 ETF
          在 ``d`` 尚无任何已公告报告期时**不剔除**（信息还不存在）。
        - ``lookback`` 窗口经 ``min_periods=1`` 回退：新上市不足 ``lookback`` 日的
          ETF，其 ADV 为上市以来部分窗口均值，与研报「过去 20 日日均成交额」存在
          口径偏差——系与 ``build_dynamic_representative_panel`` 共用
          ``_rolling_avg_amount`` 的有意单源选择。
    """
    if "index_code" not in meta.columns:
        raise ValueError(
            "meta 缺 index_code 列——请用 fetch_etf_meta（2026-08-29 起返回该列）。"
            "缺列时若静默跳过去重，池子会从约 447 只膨胀到约 1544 只而不报错。"
        )

    code2index = meta["index_code"].dropna()
    cols = [c for c in amount.columns if c in code2index.index]
    if not cols:
        raise ValueError("meta 与 amount 无交集 ETF；检查 ts_code 后缀是否一致")

    rolled = _rolling_avg_amount(amount[cols], lookback)

    rows: dict[pd.Timestamp, pd.Series] = {}
    for d in rebal_dates:
        if d not in rolled.index:
            continue
        avg = rolled.loc[d]
        cand = pd.DataFrame({
            "ts_code": avg.index,
            "adv": avg.to_numpy(),
            "index_code": code2index.reindex(avg.index).to_numpy(),
        }).dropna(subset=["adv", "index_code"])
        cand = cand[cand["adv"] > 0]
        if overseas_share is not None and not cand.empty:
            # PIT：每只 ETF 取 ann_date <= d 中最近报告期的境外占比
            vis = overseas_share[overseas_share["ann_date"] <= d]
            if not vis.empty:
                latest = vis.loc[vis.groupby("code")["end_date"].idxmax()]
                drop = set(
                    latest.loc[latest["overseas_share"] > overseas_threshold, "code"]
                )
                if drop:
                    cand = cand[~cand["ts_code"].isin(drop)]
        if cand.empty:
            continue
        # 平局取 ts_code 字典序最小：先按 ts_code 升序，再 stable 按 adv 降序
        cand = cand.sort_values("ts_code").sort_values(
            "adv", ascending=False, kind="mergesort"
        )
        rows[d] = cand.drop_duplicates("index_code", keep="first").set_index(
            "index_code"
        )["ts_code"]

    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=None))
    return pd.DataFrame(rows).T.sort_index()


def build_dynamic_etf_industry(
    industry_weights: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    *,
    min_dominant_weight: float = 0.5,
) -> pd.DataFrame:
    """每换仓日按当期 PIT 持仓 ``argmax`` 主导行业 → 动态 etf→行业 **long 表**。

    与 :func:`build_dynamic_representative_panel` 共享 PIT 快照 + argmax + 提纯门
    口径（第 1–3 步），但**不做** ``top_n`` 流动性选——保留全部过门 ETF，产出
    长表而非代表面板。是「动态 etf→行业归类」的**单一事实源**：
    :func:`plot_etf_holding_heatmap` 的动态 ``etf_industry`` 入参、
    ``script/ab_static_vs_dynamic.py`` 物化的审计 CSV、example/notebook 均调它，
    避免各处手搓重复且脱离测试覆盖。

    :param industry_weights: ``ETFExposureBuilder.build_industry_weights`` 产出
        long 表 ``[code, end_date, ann_date, sw_code, weight]``。
    :param rebal_dates: 换仓日序列。
    :param min_dominant_weight: 主导行业占比下限（默认 0.5）；低于此门的 ETF
        （宽基/债基持仓分散）出局。
    :returns: long ``DataFrame[rebal_date, etf_code, sw_code, dominant_weight]``；
        无任何可见持仓时返回**列齐全的空表**（下游可安全消费）。平局取 ``sw_code``
        字典序最小（先按 ``sw_code`` 排序使 ``idxmax`` 取首个=字典序最小，确定性）。
    """
    rows: list[pd.DataFrame] = []
    for d in rebal_dates:
        snap = _pit_latest_holdings(industry_weights, d)
        if snap.empty:
            continue
        idx = snap.sort_values("sw_code").groupby("code")["weight"].idxmax()
        dom = snap.loc[idx, ["code", "sw_code", "weight"]]
        dom = dom[dom["weight"] >= min_dominant_weight]
        if dom.empty:
            continue
        rows.append(dom.assign(rebal_date=d).rename(
            columns={"code": "etf_code", "weight": "dominant_weight"}))
    cols = ["rebal_date", "etf_code", "sw_code", "dominant_weight"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(rows, ignore_index=True)[cols]


def build_exposure_panel(
    reps_panel: pd.DataFrame,
    industry_weights: pd.DataFrame,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    每月末代表 ETF 池 + 真实持仓行业权重 → 当月暴露矩阵 ``E_t``（连续权重 + PIT）。

    替代旧 0/1 关键词近似版：``E_t`` 的每行是该 ETF 的**连续行业权重向量**
    （来自 ``exposure.aggregate_industry_weights``），与 ``optimizer`` 的
    ``a = E @ score`` / ``|E·w − b| ≤ δ`` 数学契约一致。

    PIT：每个换仓日 ``t`` 对每只代表 ETF，取 ``ann_date ≤ t`` 中**最近报告期**
    （``end_date`` 最大）的那期持仓权重——已披露才可见，无前视。

    :param reps_panel: ``build_representative_panel`` / ``filter_top_per_industry``
        输出（rebal_dates × track_code，值=当期代表 ETF ``ts_code``）。
    :param industry_weights: ``exposure.ETFExposureBuilder.build_industry_weights``
        输出 long 表，含 ``code`` (ETF) / ``end_date`` / ``ann_date`` / ``sw_code`` / ``weight``。
    :returns: ``{换仓日: E_t}``，``E_t: index=ts_code(本月代表) × columns=sw_code,
        值=行业权重``。无任何 PIT 可见持仓的换仓日被跳过（不入 dict）。喂
        ``optimizer.batch_optimize`` 的 ``exposures`` 参数。
    """
    iw = industry_weights.sort_values(["code", "end_date"])
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for d, row in reps_panel.iterrows():
        reps = pd.unique(row.dropna().to_numpy())
        if len(reps) == 0:
            continue

        # PIT：取最近已披露报告期（_pit_latest_holdings），再限定本期代表
        snap = _pit_latest_holdings(iw, d)
        sel = snap[snap["code"].isin(reps)]
        if sel.empty:
            continue

        out[d] = sel.pivot_table(
            index="code", columns="sw_code", values="weight", fill_value=0.0
        )
    return out


def mask_listed(weights: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    按 ETF 上市日把"上市前"的权重强制 0，被清零的权重按原行和重摊（行和不变）。

    :param weights: 权重宽表，``index=date × columns=ts_code``。
    :param meta: ``fetch_etf_meta`` 返回（``index=ts_code, columns=[list_date, ...]``）。
    :returns: 同 shape 权重表；上市日 > 当日的格子已置 0，被清零的权重按**原行和**
        重摊到该行其余 ETF（行和保持不变，全空行保持全 0）。
    :note:
        - 在时变流动性预筛已起作用的前提下，本步骤是**双保险**——
          未上市 ETF 的 amount 全 NaN，本就不会被选为代表；但若数据层缺数
          导致漏过滤，此处兜底。
        - **不把行归一到 1**（2026-09-02 修，偏离分析 §3.11 ①）：旧实现无条件
          ``w / row_sum`` 会把 ``etf_capped`` 的半仓 ``Σw = min(1, 0.2k)`` 抹成满仓
          （k=3 → 0.2 各变 0.33，k=1 → 1.00）。现在只在真有格子被清零时按
          清零前的行和重摊，各预设自己的仓位语义原样透传。
    """
    list_date = meta["list_date"].reindex(weights.columns)
    # listed[t, f] = (date_t >= list_date_f)
    listed = pd.DataFrame(
        weights.index.to_numpy()[:, None] >= list_date.to_numpy(),
        index=weights.index, columns=weights.columns,
    )
    before = weights.sum(axis=1)
    w = weights.where(listed, 0.0)
    after = w.sum(axis=1)
    # 清零后行和变小的行按原行和重摊；没清零的行 scale=1 原样返回；全空行保持 0
    scale = (before / after).where(after > 1e-9, 1.0)
    return w.mul(scale, axis=0)


def run_vbt_backtest(
    target_weights: pd.DataFrame,
    adjclose: pd.DataFrame,
    adjopen: pd.DataFrame | None = None,
    *,
    fees: float = 3e-4,
    slippage: float = 0.0,
    init_cash: float = 1e7,
) -> vbt.Portfolio:
    """
    多标的 ETF 组合回测——目标权重路径 + 后复权估值（close）+ 后复权开盘成交（price）。

    :param target_weights: 月末换仓权重宽表，``index=换仓日 × columns=ts_code``
        （非换仓日的行不传，下游自动 reindex 到日频 + ffill 持仓不变）。
    :param adjclose: 后复权收盘价宽表，与 target_weights columns 一致 + 全交易日 index。
        用作 vbt ``close=``——按当日**收盘**计组合净值与日收益率。
    :param adjopen: 后复权开盘价宽表，shape 同 ``adjclose``。用作 vbt ``price=``——
        换仓**按开盘价成交**（信号基于换仓日收盘、次日开盘执行——本函数内部已把
        目标权重 ``shift(1)`` 顺延一个交易日，消除日内前视）。
        **必须与 ``adjclose`` 同一后复权基**（同 ``S_DQ_ADJFACTOR``）。``None``（默认）
        → 退化为按收盘成交（``price=adjclose``），仅供无开盘数据的单元测试 / 简化场景。
    :param fees: 单边手续费率（默认 0.03% = 研报单边万 3）。百分比费率与价基无关。
    :param slippage: 单边滑点率（默认 0.0；研报仅设单边万 3 费率，
        故全部计入 fees、slippage 置 0，避免双重计费偏离研报口径）。
    :param init_cash: 初始资金。
    :returns: ``vbt.Portfolio``——多资产共享现金组合。
        ``pf.value()`` = 单条组合净值曲线；``pf.stats()`` = 组合绩效。
    :note:
        - **``close`` 与 ``price`` 是"估值 vs 成交"之分，不是"复权 vs 原价"**：vbt
          ``from_orders`` 用 ``close`` 计组合净值并反推 targetpercent 目标股数，``price``
          是成交价；二者**必须同一复权基**。本回测 ``close=adjclose``（收盘估值）、
          ``price=adjopen``（开盘成交）——同为后复权，``adjclose/adjopen`` 仅是真实日内
          涨跌，**不注入复权因子**。⚠️ 历史 bug：曾用 ``price=原始收盘价`` 与 ``adjclose``
          混基，每笔成交注入 = 复权因子(后复权/原价) 的幻收益，换仓日 ±20–40% 跳变——
          **成交价绝不可换成原价/异基序列**。
        - **固化 ``cash_sharing=True + group_by=True``**——所有 ETF 共用现金账户、
          作为单一组合结算。不暴露这两个参数避免误退回到"N 个独立账户"模式。
        - target_weights 在非换仓日为 NaN，vbt 解释为"不调仓、维持当前股数"。
    """
    # 月末换仓权重 → 日频对齐（非换仓日 NaN = 持仓不动）
    weights_daily = target_weights.reindex(adjclose.index)
    # 顺延 1 个交易日执行：换仓日 d 的目标由 d 日收盘信号算出，故在 d+1 开盘成交，
    # 消除日内前视。原实现把目标挂在 d 并按 adjopen[d] 成交 = 用 d 日收盘信息在 d 日
    # 开盘（9:30）成交，与本函数「信号基于昨收、次日开盘执行」的口径不符。
    weights_daily = weights_daily.shift(1)
    # 成交价：默认按收盘（adjopen=None），传入 adjopen 则按开盘成交（贴实盘）
    price = adjclose if adjopen is None else adjopen
    return vbt.Portfolio.from_orders(
        close=adjclose,
        price=price,  # 与 close 必须同一后复权基——绝不可换成原价/异基，否则注入复权因子幻收益
        size=weights_daily,
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        freq="D",
    )


def _pivot_sw_field(sw_price: pd.DataFrame, field: str) -> pd.DataFrame:
    """申万行业指数长表 ``[trade_date, code, close, open]`` → 宽表 ``index=date × columns=code``。

    ``ffill`` 补行业指数偶发缺日（vbt 估值/成交需连续价格序列；leading NaN 无前值
    可填仍保留，但对应行业当日信号通常为 False、权重 0，不持仓）。
    """
    return (
        sw_price.pivot(index="trade_date", columns="code", values=field)
        .sort_index()
        .ffill()
    )


def _rebalance_rows_only(weights: pd.DataFrame) -> pd.DataFrame:
    """只保留权重相对前一日发生变化的行；其余日 vbt 收到 NaN → 持仓不动。

    首行与全 NaN 的 ``shift(1)`` 比较恒为 "变化" → 保留（建仓日）。持仓集合从
    有仓位切到全 0（空仓）也算变化 → 保留该全 0 行 → vbt 清仓持现金。
    """
    changed = (weights != weights.shift(1)).any(axis=1)
    return weights[changed]


def run_signal_backtest(
    signal: pd.DataFrame,
    sw_price: pd.DataFrame,
    *,
    rebalance: str = "on_change",
    fees: float = 3e-4,
    slippage: float = 0.0,
    init_cash: float = 1e7,
) -> vbt.Portfolio:
    """行业指数 boolean 信号 → 等权组合 → vbt 回测（研报 §2 行业轮动落地）。

    把 ``RRGSignalGenerator.generate`` 出的**日频 boolean 信号**（``True``=持有 /
    ``False``=空仓）按**等权**配置成申万一级行业指数组合，复用 ``run_vbt_backtest``
    的同一 vbt 口径：``close=收盘`` 估值/算每日盈亏，``price=open`` 开盘成交，
    ``fees``/``slippage``/``init_cash`` 默认与 ETF 路径一致。是 §3 ETF 路径
    （``run_vbt_backtest``）在 §2 行业指数层的平行入口。

    等权由本函数负责（调用方只给 boolean 信号）：``to_equal_weight`` 把每行 ``True``
    数均分，全 ``False`` 行 → 全 0 = 空仓持现金。

    :param signal: ``index=date × columns=industry_code`` 的 boolean 信号
        （``RRGSignalGenerator.generate`` 输出）。列名须为申万一级 ``industry_code``
        （如 ``'801010.SI'``），与 ``sw_price`` 的 ``code`` 列同口径才对得齐。
    :param sw_price: ``get_sw_industry_data`` 长表，至少含
        ``[trade_date, code, close, open]`` 四列。
    :param rebalance: ``"on_change"``（默认，仅持仓集合/权重变化日换仓，贴"持有"
        语义）或 ``"daily"``（每日把权重拉回等权，即使持仓不变也因价格漂移产生每日
        微再平衡 + 手续费空耗）。
    :param fees: 单边手续费率（默认 ``3e-4`` = 研报单边万 3，与 ETF 路径一致）。
    :param slippage: 单边滑点率（默认 ``0.0``，与 ETF 路径一致）。
    :param init_cash: 初始资金（默认 ``1e7``）。
    :returns: ``vbt.Portfolio``——多行业共享现金组合；``pf.value()`` 组合净值、
        ``portfolio_stats(pf)`` 绩效摘要。
    :raises ValueError: ``rebalance`` 取值非法，或 ``signal`` 行业列与 ``sw_price``
        的 ``code`` 列无交集。
    :note:
        - 列对齐：取 ``signal.columns ∩ close.columns``，等权在交集上重算
          （某 ``True`` 行业若无价格则不计入分母，避免静默漏配）。
        - 内部委托 ``run_vbt_backtest``——非换仓日 reindex 成 NaN = 持仓不动，
          ``cash_sharing=True + group_by=True`` 组合视角固化。
    """
    if rebalance not in ("on_change", "daily"):
        raise ValueError(f"rebalance 须为 'on_change' / 'daily'，收到 {rebalance!r}")

    close = _pivot_sw_field(sw_price, "close")
    open_ = _pivot_sw_field(sw_price, "open")

    # 列对齐：信号行业 ∩ 有价格的行业（二者均应为申万一级 industry_code）
    cols = signal.columns.intersection(close.columns)
    if cols.empty:
        raise ValueError(
            "signal 行业列与 sw_price 的 code 列无交集——请确认二者同为申万一级 "
            "industry_code（如 '801010.SI'）"
        )
    signal = signal[cols]
    close, open_ = close[cols], open_[cols]

    # 等权目标权重：每行 True 数均分，全 False 行 → 全 0（空仓持现金）
    weights = to_equal_weight(signal.astype(bool))
    if rebalance == "on_change":
        weights = _rebalance_rows_only(weights)

    return run_vbt_backtest(
        weights, close, open_, fees=fees, slippage=slippage, init_cash=init_cash
    )


def plot_cumulative_vs_benchmark(
    pf: vbt.Portfolio,
    benchmark_price: pd.Series,
    *,
    start_value: float = 1.0,
    fill_to_benchmark: bool = True,
    **layout_kwargs,
):
    """组合累计收益曲线 vs **自定义基准**（传入一条价格 Series）。

    vbt 1.0 的 ``Portfolio.plot`` 默认基准是"全标的等权 buy&hold"（从 ``close`` 自动
    推导，无法注入）。本助手绕开它——把策略收益率与**你给的基准价格序列**喂进
    ``returns accessor`` 的 ``plot_cumulative(benchmark_rets=...)``，画出策略（Value）
    对照自定义基准（Benchmark）的累计收益图（绿/红填充=超额/跑输）。

    :param pf: ``run_signal_backtest`` / ``run_vbt_backtest`` 返回的组合（grouped，
        ``pf.returns()`` 为单条组合收益率 Series）。
    :param benchmark_price: 基准**价格**序列（``index=date``，如中证 1000 收盘、或行业
        等权 NAV ``ind_avg``）。内部 ``reindex(策略收益率 index).ffill().pct_change()``
        对齐成与策略同 index 的基准收益率——**价格而非收益率传入即可**。
    :param start_value: 两条曲线的起点（默认 ``1.0``，即归一化累计净值）。
    :param fill_to_benchmark: ``True``（默认）填充策略与基准之间的区域（绿=跑赢/
        红=跑输）；``False`` 则填充到起点水平线。
    :param layout_kwargs: 透传 plotly ``update_layout``（如 ``title=...``、``width=...``）。
    :returns: plotly ``Figure``——``.show()`` 显示、``.write_html(...)`` 存盘。
    :raises ValueError: ``benchmark_price`` 与策略收益率 index 无任何重合日。
    :note:
        - 基准曲线**不计费用/滑点**（buy&hold 价格序列本身的收益），与策略含费净值对照
          是行业惯例。
        - 若 ``benchmark_price`` 覆盖不全策略区间，缺口由 ``ffill`` 兜底（停牌/缺日按
          前值持平），头部无前值的缺口收益率记 0。
    """
    strat_rets = pf.returns().rename("Value")  # grouped → 单条组合收益率；命名对齐图例
    aligned = benchmark_price.reindex(strat_rets.index)
    if aligned.notna().sum() == 0:
        raise ValueError(
            "benchmark_price 与策略收益率 index 无重合日——请确认基准价格序列的日期"
            "覆盖回测区间且为交易日"
        )
    benchmark_rets = aligned.ffill().pct_change().fillna(0.0)
    fig = strat_rets.vbt.returns.plot_cumulative(
        benchmark_rets=benchmark_rets,
        start_value=start_value,
        fill_to_benchmark=fill_to_benchmark,
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig


def portfolio_stats(pf: vbt.Portfolio) -> pd.Series:
    """
    组合绩效摘要——累计 / 年化 / MDD / 波动率 / 夏普 / Calmar / 总成交笔数。

    :param pf: ``run_vbt_backtest`` 返回。
    :returns: 中文标签 Series，便于直接 print 或拼报表。
    :note:
        研报表 13 含「双边换手率」字段，本摘要暂以 ``总成交笔数``
        （``pf.orders.count()``）替代；精确年化换手率需从 orders records 自行求和。
    """
    return pd.Series({
        "累计收益率": float(pf.total_return()),
        "年化收益率": float(pf.annualized_return()),
        "最大回撤": float(pf.max_drawdown()),
        "年化波动率": float(pf.annualized_volatility()),
        "夏普比率": float(pf.sharpe_ratio()),
        "Calmar 比率": float(pf.calmar_ratio()),
        "总成交笔数": int(pf.orders.count()),
    })

"""ETF-RRG 原始数据 DuckDB 缓存层。

单一 ``data/etf_rrg.duckdb`` 增量缓存原始实体（phase 1：ETF 季度持仓）；缓存 miss
时回退调 ``data_provider`` 远程取数 → upsert → 出表。派生矩阵（industry_weights /
E_t）不缓存，由上层按需重算。详见
``docs/superpowers/specs/2026-06-04-duckdb-cache-layer-design.md``。
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

#: 部分填充判定阈值：缺口内有数据的工作日占比低于此值即 warn。工作日比真实交易日
#: 多约 4%（每年 ~10 个节假日），0.6 远在噪声之外；SW 2021 那种 1/250 稳稳命中。
_PARTIAL_FILL_RATIO = 0.6

from src.data_provider import (
    _PORTFOLIO_COLS,
    fetch_etf_portfolio_range,
    fetch_sw_membership,
    fetch_zx_membership,
)

__all__ = ["RRGStore"]

DEFAULT_DB_PATH = "data/etf_rrg.duckdb"

#: classification → (成分表名, 水位表名, 取数原子)。两套缓存共用同一段增量逻辑，
#: 只换表名与 fetch——**不要复制并行代码**，否则两边的增量语义会漂移。
#: 取数原子用惰性 lambda 包装：调用时才解析模块级 ``fetch_*_membership`` 名字，
#: 使 ``monkeypatch.setattr(store, "fetch_sw_membership", ...)`` 仍可注入测试桩。
_MEMBERSHIP_BACKENDS = {
    "sw": ("sw_membership", "sw_membership_coverage",
           lambda s, e: fetch_sw_membership(s, e)),
    "zx": ("zx_membership", "zx_membership_coverage",
           lambda s, e: fetch_zx_membership(s, e)),
}


class RRGStore:
    """DuckDB 原始数据缓存管理器（持仓 + 行业成分（SW / 中信双体系））。

    :param db_path: DuckDB 文件路径，默认 ``data/etf_rrg.duckdb``；测试可传临时路径。
    :param read_only: ``True``（默认）以**只读**打开——离线版只读快照、从不写库，只读打开
        才能支持「快照放只读介质 / chmod 444」的部署，且多个只读连接可跨进程并存
        （读写连接是独占的，任何别的进程打开该库都会互相锁死）。只读时跳过
        ``_init_schema``（完整快照表已在，且只读连接不能跑 DDL）。
        需要建库 / 写缓存（如生成快照、测试造 fixture）时显式传 ``False``。
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, *, read_only: bool = True) -> None:
        self.db_path = str(db_path)
        self.read_only = read_only
        if not read_only and self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(self.db_path, read_only=read_only)
        if not read_only:
            self._init_schema()

    def _init_schema(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS etf_portfolio (
                code TEXT, symbol TEXT, ann_date DATE, end_date DATE,
                mkv DOUBLE, amount DOUBLE, stk_mkv_ratio DOUBLE, stk_float_ratio DOUBLE,
                PRIMARY KEY (code, end_date, symbol)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS etf_portfolio_coverage (
                code TEXT, period DATE, fetched_at TIMESTAMP,
                PRIMARY KEY (code, period)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS sw_membership (
                trade_date DATE, stock_code TEXT, industry_code TEXT,
                PRIMARY KEY (trade_date, stock_code)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS sw_membership_coverage (
                id INTEGER PRIMARY KEY DEFAULT 1,
                min_date DATE, max_date DATE, fetched_at TIMESTAMP
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS zx_membership (
                trade_date DATE, stock_code TEXT, industry_code TEXT,
                PRIMARY KEY (trade_date, stock_code)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS zx_membership_coverage (
                id INTEGER PRIMARY KEY DEFAULT 1,
                min_date DATE, max_date DATE, fetched_at TIMESTAMP
            )
            """
        )

    @staticmethod
    def _naive(*ts: object) -> tuple[pd.Timestamp, ...]:
        """把入参统一成 tz-naive ``pd.Timestamp``（防 tz-aware 与 DuckDB 静默错位）。"""
        out = []
        for t in ts:
            t = pd.Timestamp(t)
            out.append(t.tz_localize(None) if t.tz is not None else t)
        return tuple(out)

    @contextmanager
    def _txn(self):
        """BEGIN/COMMIT 包一段写；异常 ROLLBACK 后重抛——崩溃不留半写瞬态。"""
        self._con.execute("BEGIN TRANSACTION")
        try:
            yield
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise

    def _insert_select(self, sql: str, df: pd.DataFrame) -> None:
        """把 ``df`` 注册成 ``_ins`` 跑 ``INSERT … SELECT … FROM _ins``，无论成败必注销。"""
        self._con.register("_ins", df)
        try:
            self._con.execute(sql)
        finally:
            self._con.unregister("_ins")

    def get_portfolio(
        self, codes, start, end, *, refresh: bool = False
    ) -> pd.DataFrame:
        """增量取 ETF 季度持仓（与 ``data_provider.fetch_etf_portfolio`` 同形长表）。

        只抓缺口（新 code / 新季度），缓存命中走 SQL；``ann_date≤t`` 的 PIT 由上层把关。

        **取数粒度：每个缺口 code 一次 ``fetch_etf_portfolio_range``**（``fund_portfolio``
        支持 ``start_date`` / ``end_date`` 一次返回区间全部报告期），再把结果按 ``end_date``
        分发到各缺口季度——较旧版逐 (code, period) 调用省约 N 倍 API 往返。⚠️ 该接口
        ``start_date`` / ``end_date`` 过滤的是**公告日 ann_date**，故 range 端点取"今天"
        （``frontier``）以捕获全部已披露报告期（含尾期年报晚于 ``end`` 的公告）；请求的
        报告期窗口由本地 ``needed`` 季度集裁剪，与逐期取数**逐行等价**（已实测）。

        每个 code 的 DELETE+INSERT+coverage（其全部缺口季度）包在单事务里——崩溃不留
        "有 data 无 coverage" 的瞬态。未披露的尾期返回空 → 仍写 coverage（缓存为空，
        ``refresh`` 重取），与旧逐期行为一致。

        :param codes: ETF ``ts_code`` 列表。
        :param start: 区间起始（``'YYYY-MM-DD'`` / ``pd.Timestamp``）。
        :param end: 区间结束。
        :param refresh: ``True`` 绕过 coverage 强制重抓（数据更正用）。
        :returns: long ``DataFrame``（``_PORTFOLIO_COLS``）；``ann_date``/``end_date``
            为 ``datetime64``，``end_date`` ∈ [start, end] 且 ``code`` ∈ codes。
        """
        codes = sorted(set(codes))
        start, end = self._naive(start, end)
        periods = pd.date_range(start=start, end=end, freq="QE-DEC")
        if len(periods) == 0 or not codes:
            return pd.DataFrame(columns=_PORTFOLIO_COLS)

        self._con.register("_req", pd.DataFrame({"code": codes}))
        try:
            needed = {(c, p.normalize()) for c in codes for p in periods}
            covered: set = set()
            if not refresh:
                cov = self._con.execute(
                    "SELECT code, period FROM etf_portfolio_coverage "
                    "WHERE code IN (SELECT code FROM _req)"
                ).df()
                covered = {(r.code, pd.Timestamp(r.period).normalize())
                           for r in cov.itertuples(index=False)}

            # 缺口按 code 聚合：每 code 一次区间取数，再分发到其缺口季度
            missing_by_code: dict[str, list[pd.Timestamp]] = {}
            for code, period in sorted(needed - covered):
                missing_by_code.setdefault(code, []).append(period)

            # ann_date 上界："今天"——捕获截至目前全部已披露报告期（与 end 无关，PIT 在上层）
            frontier = pd.Timestamp.today().normalize()

            for code, miss_periods in missing_by_code.items():
                api_start = min(miss_periods)  # ann_date ≥ 最早缺口季度，足以覆盖其公告
                rows = fetch_etf_portfolio_range(code, api_start, frontier)
                # 空表 end_date 可能为 object dtype（.dt 访问器会报错）→ 保留空分支守卫
                if not rows.empty:
                    by_period = {p: g for p, g in
                                 rows.groupby(rows["end_date"].dt.normalize())}
                else:
                    by_period = {}
                with self._txn():
                    for period in miss_periods:
                        pydate = period.to_pydatetime()
                        self._con.execute(
                            "DELETE FROM etf_portfolio WHERE code = ? AND end_date = ?",
                            [code, pydate],
                        )
                        sub = by_period.get(period)
                        if sub is not None and not sub.empty:
                            self._insert_select(
                                "INSERT INTO etf_portfolio SELECT * FROM _ins",
                                sub[_PORTFOLIO_COLS],
                            )
                        self._con.execute(
                            "INSERT INTO etf_portfolio_coverage VALUES (?, ?, now()) "
                            "ON CONFLICT (code, period) DO UPDATE SET fetched_at = excluded.fetched_at",
                            [code, pydate],
                        )

            out = self._con.execute(
                "SELECT p.code, p.symbol, p.ann_date, p.end_date, p.mkv, p.amount, "
                "       p.stk_mkv_ratio, p.stk_float_ratio "
                "FROM etf_portfolio p JOIN _req r ON p.code = r.code "
                "WHERE p.end_date BETWEEN ? AND ? "
                "ORDER BY p.end_date, p.code, p.symbol",
                [start.to_pydatetime(), end.to_pydatetime()],
            ).df()
        finally:
            self._con.unregister("_req")

        # 强制 datetime64[ns]——DuckDB 默认返回 us，与 data_provider.fetch_etf_portfolio
        # 的 ns 出参对齐，保持 byte drop-in（下游 merge_asof 不再因分辨率不一致报错）。
        out["ann_date"] = pd.to_datetime(out["ann_date"]).astype("datetime64[ns]")
        out["end_date"] = pd.to_datetime(out["end_date"]).astype("datetime64[ns]")
        return out[_PORTFOLIO_COLS].reset_index(drop=True)

    def get_membership(
        self, start, end, *, industry_codes=None, refresh: bool = False,
        classification: str = "sw",
    ) -> pd.DataFrame:
        """全市场一级行业成分（SW / 中信，按 ``classification``；drop-in for
        旧 datacenter 成分接口的 ``[code, trade_date, industry_code]``）。

        单一连续水位 ``[min, max]`` 增量：请求超出水位时只补两头缺口；始终全市场抓，
        ``industry_codes`` 仅查询层过滤。重构期 ``(trade_date, stock)`` 多行业在入库
        ``drop_duplicates(keep="first")`` 去重（复刻旧 ``pivot aggfunc="first"``）。

        :param start: 区间起始（``'YYYY-MM-DD'`` / ``pd.Timestamp``）。
        :param end: 区间结束。
        :param industry_codes: ``None`` 或空列表 = 全部行业；非空 = 只返回这些
            ``industry_code`` 的行（查询层过滤，底层 fetch 仍全市场）。
        :param refresh: ``True`` 强制重抓 ``[start, end]``。``refresh`` 仅重新 fetch
            ``[start, end]``；源有返回则覆盖该区间数据，**源返回空则保留原有缓存不删**
            （只 warn）。水位取现有覆盖与本次请求的并集（不清除区间外缓存）。
        :param classification: ``"sw"``（申万一级，默认）或 ``"zx"``（中信一级）。两套各自
            独立的表与水位，互不影响；现有 sw 缓存零迁移。
        :returns: long ``DataFrame[code, trade_date, industry_code]``，
            ``trade_date`` 为 ``datetime64[ns]``。
        """
        # 表名 / 取数原子来自内部白名单常量（非用户输入），SQL 拼接无注入风险；
        # 日期与行业码仍走参数绑定。
        try:
            table, cov_table, fetch_fn = _MEMBERSHIP_BACKENDS[classification]
        except KeyError:
            raise ValueError(
                f"classification 必须是 {sorted(_MEMBERSHIP_BACKENDS)}，收到 {classification!r}"
            ) from None
        start, end = self._naive(start, end)

        row = self._con.execute(
            f"SELECT min_date, max_date FROM {cov_table}"
        ).fetchone()
        have_cov = row is not None
        cmin = pd.Timestamp(row[0]) if have_cov else None
        cmax = pd.Timestamp(row[1]) if have_cov else None

        gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if refresh or not have_cov:
            gaps.append((start, end))
        else:
            if start < cmin:
                gaps.append((start, cmin - pd.Timedelta(days=1)))
            if end > cmax:
                gaps.append((cmax + pd.Timedelta(days=1), end))

        empty_gaps: list[tuple] = []
        partial_gaps: list[str] = []
        gap_results: list[tuple] = []
        for g_start, g_end in gaps:
            batch = fetch_fn(g_start, g_end)
            # 过滤到当前 gap 区间内（防止 fetch 返回区间外日期引发 PK 冲突）
            batch = batch[
                (batch["trade_date"] >= g_start) & (batch["trade_date"] <= g_end)
            ]
            batch = batch.drop_duplicates(["trade_date", "code"], keep="first")
            gap_results.append(((g_start, g_end), not batch.empty))
            if batch.empty:
                empty_gaps.append((g_start.date(), g_end.date()))
            else:
                # 部分填充检测（无日历的廉价代理：工作日数 ≈ 交易日数上界，多约 4%）。
                # 缺口只回来零星几天时水位仍会推到缺口末端，这里把它从静默变成可见。
                expected = len(pd.bdate_range(g_start, g_end))
                actual = int(batch["trade_date"].nunique())
                if expected and actual < _PARTIAL_FILL_RATIO * expected:
                    partial_gaps.append(
                        f"[{g_start.date()}, {g_end.date()}] 仅 {actual}/{expected} 个工作日有数据"
                        f"（实际数据范围 {batch['trade_date'].min().date()} ~ "
                        f"{batch['trade_date'].max().date()}）"
                    )
            # 没有替换数据就不删：refresh 撞上空返回时保住已入库的行
            if not batch.empty:
                with self._txn():
                    self._con.execute(
                        f"DELETE FROM {table} WHERE trade_date BETWEEN ? AND ?",
                        [g_start.to_pydatetime(), g_end.to_pydatetime()],
                    )
                    self._insert_select(
                        f"INSERT INTO {table}(trade_date, stock_code, industry_code) "
                        "SELECT trade_date, code, industry_code FROM _ins",
                        batch[["trade_date", "code", "industry_code"]],
                    )

        if empty_gaps:
            if refresh:
                tail = "已**保留**原有缓存未删，水位不变。"
            else:
                tail = "这些区间不会计入 coverage 水位，下次请求会重新尝试。"
            warnings.warn(
                f"{classification} 成分取数在以下区间**未返回任何数据**：{empty_gaps}——{tail}"
                f"若反复出现，说明上游数据源在该区间确实缺失（而非缓存问题）",
                UserWarning,
                stacklevel=2,
            )
        if partial_gaps:
            warnings.warn(
                f"{classification} 成分取数**部分填充**：{'；'.join(partial_gaps)}。"
                f"水位仍按缺口边界推进（单一 [min,max] 区间不追踪内部空洞），"
                f"缺失段下次**不会**重抓——请核对上游数据源在该区间是否稀疏",
                UserWarning,
                stacklevel=2,
            )

        # 水位**逐缺口**推进：只有确实取到数据的缺口才把水位扩到它的请求边界。
        #
        # 2026-08-31 修复：原实现用 min(cmin, start) / max(cmax, end) 无条件推进，
        # 于是「源里没有数据」被静默记成「已覆盖」，后续请求命中空缓存、永不再抓
        # （§3.10 的 SW 成分缺口 2014–2020 全空即被这样掩盖，水位被空跑推了 7.9 年）。
        #
        # 为什么按缺口粒度而非「表内实际数据范围」：后者会让水位缩到首末数据点，
        # 于是起止日落在非交易日（如元旦）时每次请求都重抓一个空缺口。缺口粒度
        # 同时满足两边——边缘节假日所在缺口有数据故正常推进，整段全空的缺口不推进。
        #
        # 区间内部的空洞仍按既有 YAGNI 取舍不追踪（phase-1 spec 明确不做区间集代数）：
        # 整段无数据的缺口会 warn 且不推进水位——非 refresh 场景下它保持未覆盖、下次会重试；
        # refresh 场景下水位本就覆盖它，只保留旧缓存、**不会**自动重试（须再显式 refresh）。
        # **部分填充的缺口仍会把水位推到缺口末端**（缺失段不再重抓），只以上面的「部分填充」
        # warn 暴露出来。
        # 2026-09-02：refresh 撞上空返回不再先删后不写（见循环内 `if not batch.empty`）。
        filled = [(gs, ge) for (gs, ge), had in gap_results if had]
        if filled:
            new_min = min([gs for gs, _ in filled] + ([cmin] if have_cov else []))
            new_max = max([ge for _, ge in filled] + ([cmax] if have_cov else []))
            self._con.execute(
                f"INSERT INTO {cov_table}(id, min_date, max_date, fetched_at) "
                "VALUES (1, ?, ?, now()) "
                "ON CONFLICT (id) DO UPDATE SET "
                "min_date = excluded.min_date, max_date = excluded.max_date, "
                "fetched_at = excluded.fetched_at",
                [new_min.to_pydatetime(), new_max.to_pydatetime()],
            )

        params = [start.to_pydatetime(), end.to_pydatetime()]
        if industry_codes:
            self._con.register(
                "_ind", pd.DataFrame({"industry_code": list(industry_codes)})
            )
            try:
                out = self._con.execute(
                    f"SELECT m.stock_code AS code, m.trade_date, m.industry_code "
                    f"FROM {table} m JOIN _ind i ON m.industry_code = i.industry_code "
                    "WHERE m.trade_date BETWEEN ? AND ? "
                    "ORDER BY m.trade_date, m.stock_code",
                    params,
                ).df()
            finally:
                self._con.unregister("_ind")
        else:
            out = self._con.execute(
                f"SELECT stock_code AS code, trade_date, industry_code FROM {table} "
                "WHERE trade_date BETWEEN ? AND ? "
                "ORDER BY trade_date, stock_code",
                params,
            ).df()

        out["trade_date"] = pd.to_datetime(out["trade_date"]).astype("datetime64[ns]")
        return out[["code", "trade_date", "industry_code"]].reset_index(drop=True)

    def close(self) -> None:
        """关闭 DuckDB 连接。"""
        self._con.close()

    def __enter__(self) -> "RRGStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

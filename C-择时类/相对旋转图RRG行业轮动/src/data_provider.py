"""ETF-RRG 离线数据读取层：只读 data/etf_rrg.duckdb 快照，零外部依赖。

替代源项目的在线取数层（数据库 / 行情 API）；对外函数名与返回契约与源项目一致，
使 generator/exposure/notebook 零改动。缺数据或触发联网路径时抛 OfflineDataError。
"""
from __future__ import annotations

import pandas as pd
import duckdb
from pathlib import Path

from src.factor_algo import equal_weight_benchmark
from src.utils import datetime2str  # noqa: F401  (保留以对齐源模块公共面)

__all__ = [
    "fetch_rrg_dataset", "fetch_diffusion_dataset", "fetch_etf_meta",
    "fetch_etf_daily", "get_sw_industry_data", "fetch_zx_industry_data",
    "resolve_benchmark_exclude", "fetch_etf_portfolio_range",
    "fetch_sw_membership", "fetch_zx_membership", "OfflineDataError",
]

# 锚定到项目根（src/ 的上一级），与调用方 cwd 无关——notebook 以 example/ 为 cwd
# 运行时，相对路径 "data/etf_rrg.duckdb" 会错解析到 example/data/（迁移项目无此目录）。
DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "etf_rrg.duckdb")
_SW_DEFAULT_EXCLUDE = ("801230.SI",)
ZX_DEFAULT_EXCLUDE = ("CI005029.CI", "CI005030.CI")
_PORTFOLIO_COLS = [
    "code", "symbol", "ann_date", "end_date",
    "mkv", "amount", "stk_mkv_ratio", "stk_float_ratio",
]


class OfflineDataError(RuntimeError):
    """离线快照缺数据 / 触发了需联网的取数路径。"""


def _connect(db_path: str | None = None):
    path = db_path or DEFAULT_DB_PATH
    if path != ":memory:" and not Path(path).exists():
        raise OfflineDataError(
            f"离线快照 {path} 不存在；请从百度网盘下载 etf_rrg.duckdb 放到 data/ 下"
            f"（见 data/说明.md）"
        )
    # 只读打开：本层只发 SELECT，且离线快照从不写。RRGStore 亦默认 read_only=True，
    # 两边配置一致（DuckDB 禁止同进程内对同一文件混用 read_only=True/False）。
    # 只读还使快照可放只读介质，且多个只读连接可跨进程并存、互不锁死。
    return duckdb.connect(path, read_only=True)


def _warmup_start(con, end_date, count: int) -> str:
    """等价源 get_trade_days(end_date=end_date, count=count)[0]：<=end_date 的倒数第 count 个交易日。"""
    days = pd.DatetimeIndex(pd.to_datetime(
        con.execute("SELECT trade_date FROM trade_days ORDER BY trade_date").df()["trade_date"]
    ))
    upto = days[days <= pd.Timestamp(end_date)]
    if len(upto) < count:
        raise OfflineDataError(
            f"trade_days 快照不足以从 {end_date} 前移 {count} 个交易日（现有 {len(upto)} 天）"
        )
    return upto[-count].strftime("%Y-%m-%d")


def _read_industry(table: str, start_dt, end_dt, fields) -> pd.DataFrame:
    if isinstance(fields, str):
        fields = (fields,)
    col_sql = ", ".join(fields)  # 表名/字段来自内部白名单，非用户输入
    with _connect() as con:
        df = con.execute(
            f"SELECT trade_date, code, industry_name, {col_sql} FROM {table} "
            "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date, code",
            [pd.Timestamp(start_dt).to_pydatetime(), pd.Timestamp(end_dt).to_pydatetime()],
        ).df()
    if df.empty:
        raise OfflineDataError(f"{table} 在 {start_dt} ~ {end_dt} 无数据；快照窗口不足")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.reset_index(drop=True)


def get_sw_industry_data(start_dt, end_dt, fields="close") -> pd.DataFrame:
    """申万一级行业指数日线（drop-in 源项目同名接口）。"""
    return _read_industry("sw_industry_daily", start_dt, end_dt, fields)


def fetch_zx_industry_data(start_dt, end_dt, fields=("close", "open"), level: int = 1) -> pd.DataFrame:
    """中信一级行业指数日线（drop-in get_sw_industry_data）。"""
    if level != 1:
        raise ValueError(f"离线版仅支持 level=1，收到 {level!r}")
    return _read_industry("zx_industry_daily", start_dt, end_dt, fields)


def resolve_benchmark_exclude(benchmark_exclude, classification):
    if classification not in ("sw", "zx"):
        raise ValueError(f"classification 必须是 'sw' / 'zx'，收到 {classification!r}")
    if benchmark_exclude != "auto" and isinstance(benchmark_exclude, str):
        raise ValueError(f"benchmark_exclude 只接受 'auto' / tuple / None，收到 {benchmark_exclude!r}")
    if benchmark_exclude != "auto":
        return benchmark_exclude
    return _SW_DEFAULT_EXCLUDE if classification == "sw" else ZX_DEFAULT_EXCLUDE


def fetch_rrg_dataset(benchmark_code="ind_avg", start_date="2022-01-01", end_date="2025-12-31",
                      *, benchmark_exclude="auto", classification="sw"):
    """RRG 基准 + 行业宽表 + 代码映射（离线仅支持 ind_avg 等权基准）。"""
    if benchmark_code != "ind_avg":
        raise OfflineDataError(
            "离线快照仅支持 benchmark_code='ind_avg'（等权行业基准）；"
            "外部指数基准需联网，未纳入快照"
        )
    if classification == "sw":
        industry_df = get_sw_industry_data(start_dt=start_date, end_dt=end_date, fields="close")
    elif classification == "zx":
        industry_df = fetch_zx_industry_data(start_dt=start_date, end_dt=end_date, fields=("close",))
    else:
        raise ValueError(f"classification 必须是 'sw' / 'zx'，收到 {classification!r}")
    industry_close = industry_df.pivot(index="trade_date", columns="code", values="close").ffill()
    code2name = (industry_df.dropna(subset=["industry_name"])
                 .set_index("code")["industry_name"].to_dict())
    exclude = resolve_benchmark_exclude(benchmark_exclude, classification)
    cols = industry_close if exclude is None else industry_close.drop(columns=list(exclude), errors="ignore")
    benchmark = equal_weight_benchmark(cols)
    return benchmark, industry_close, code2name


def fetch_diffusion_dataset(industry_codes=None, start_date="2022-01-01", end_date="2025-12-31",
                            *, lookback=220, smooth_window=20, classification="sw"):
    """扩散指标个股三宽表 (close, membership, free_float_mv)，index=date × columns=stock_code。"""
    with _connect() as con:
        begin_dt = _warmup_start(con, start_date, lookback + smooth_window)

    from src.store import RRGStore  # 延迟 import 避免 data_provider ↔ store 循环
    # 显式传锚定路径：store 自身的 DEFAULT_DB_PATH 仍是相对 cwd（store.py 逐字节保留源版），
    # 不传会与本模块解析到不同文件。
    with RRGStore(db_path=DEFAULT_DB_PATH) as _store:
        cons = _store.get_membership(begin_dt, end_date, industry_codes=industry_codes,
                                     classification=classification)
    if cons.empty:
        raise OfflineDataError(
            f"get_membership 在 {begin_dt} ~ {end_date} 无成分（classification={classification!r}）；"
            f"检查快照 membership 覆盖"
        )
    membership = cons.pivot_table(index="trade_date", columns="code",
                                  values="industry_code", aggfunc="first")

    with _connect() as con:
        cdf = con.execute(
            "SELECT date, stock_code, close FROM ashares_close WHERE date BETWEEN ? AND ?",
            [pd.Timestamp(begin_dt).to_pydatetime(), pd.Timestamp(end_date).to_pydatetime()],
        ).df()
        mdf = con.execute(
            "SELECT date, stock_code, float_mv FROM ashares_float_mv WHERE date BETWEEN ? AND ?",
            [pd.Timestamp(begin_dt).to_pydatetime(), pd.Timestamp(end_date).to_pydatetime()],
        ).df()
    if cdf.empty:
        raise OfflineDataError(f"ashares_close 在 {begin_dt} ~ {end_date} 无数据；快照不足")
    cdf["date"] = pd.to_datetime(cdf["date"])
    mdf["date"] = pd.to_datetime(mdf["date"])
    close = cdf.pivot(index="date", columns="stock_code", values="close")
    free_float_mv = mdf.pivot(index="date", columns="stock_code", values="float_mv")
    membership = membership.reindex(index=close.index, columns=close.columns)
    free_float_mv = free_float_mv.reindex(index=close.index, columns=close.columns)
    return close, membership, free_float_mv


def fetch_etf_meta(codes=None, *, only_listed=True, only_etf=True, drop_qdii=False, drop_money=False):
    """ETF 静态元信息（离线读 etf_meta 表；过滤逻辑与源版一致）。"""
    with _connect() as con:
        df = con.execute("SELECT * FROM etf_meta").df()
    if df.empty:
        raise OfflineDataError("快照 etf_meta 为空")
    if only_listed:
        df = df[df["list_status"] == "L"]
    if only_etf:
        df = df[df["ts_code"].str.endswith((".SH", ".SZ"))]
    if drop_qdii:
        df = df[df["etf_type"] != "QDII"]
    if drop_money:
        df = df[df["index_code"].notna()]
    if codes is not None:
        df = df[df["ts_code"].isin(codes)]
    df = df.copy()
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df["setup_date"] = pd.to_datetime(df["setup_date"], errors="coerce")
    return df.set_index("ts_code")[
        ["index_code", "index_name", "list_date", "list_status", "etf_type", "setup_date", "mgt_fee"]
    ]


def fetch_etf_daily(codes, start_date, end_date, *, lookback=20):
    """ETF 日行情 (adjclose, adjopen, amount)；amount 快照已转元。"""
    with _connect() as con:
        begin_dt = _warmup_start(con, start_date, lookback + 1)
        con.register("_codes", pd.DataFrame({"code": list(codes)}))
        try:
            long = con.execute(
                "SELECT d.date, d.code, d.adjclose, d.adjopen, d.amount "
                "FROM etf_daily d JOIN _codes c ON d.code = c.code "
                "WHERE d.date BETWEEN ? AND ? ORDER BY d.date, d.code",
                [pd.Timestamp(begin_dt).to_pydatetime(), pd.Timestamp(end_date).to_pydatetime()],
            ).df()
        finally:
            con.unregister("_codes")
    if long.empty:
        raise OfflineDataError(f"etf_daily 在 {begin_dt} ~ {end_date} 对 {len(list(codes))} 只无数据")
    long["date"] = pd.to_datetime(long["date"])
    adjclose = long.pivot(index="date", columns="code", values="adjclose").sort_index()
    adjopen = long.pivot(index="date", columns="code", values="adjopen").sort_index()
    amount = long.pivot(index="date", columns="code", values="amount").sort_index()
    return adjclose, adjopen, amount


def _offline_guard(name: str):
    raise OfflineDataError(
        f"{name} 触发了联网取数路径——离线快照缺该数据。请确认 data/etf_rrg.duckdb "
        f"覆盖所请求的时间窗与标的（见 data/说明.md）"
    )


def _dedup_latest_announcement(long: pd.DataFrame) -> pd.DataFrame:
    """
    同一 ``(code, end_date, symbol)`` 多次披露时仅保留最新 ``ann_date`` 一条。

    基金定期报告 + 后续更正会让同一持仓（同 ETF、同报告期、同个股）出现多条
    ``ann_date`` 不同的记录。下游 ``E`` 暴露矩阵每键只能留一条，否则聚合 ``mkv``
    时会重复计数。保留**最新** ``ann_date`` = 取最终更正后版本（PIT 过滤在
    ``build_exposure_panel`` 阶段另按 ``ann_date ≤ 换仓日`` 把关，二者不冲突）。

    :param long: 持仓 long，至少含 ``code`` / ``end_date`` / ``symbol`` / ``ann_date``。
    :returns: 去重后的 ``DataFrame``（行序不保证，由调用方 ``sort_values`` 收口）。
    """
    return (
        long.sort_values("ann_date")
        .drop_duplicates(["code", "end_date", "symbol"], keep="last")
    )


def fetch_etf_portfolio_range(code, start, end, *, ts=None):
    """离线 guard：store 缓存命中时不会调到；miss 则 fail-loud。"""
    _offline_guard("fetch_etf_portfolio_range")


def fetch_sw_membership(start_date, end_date):
    _offline_guard("fetch_sw_membership")


def fetch_zx_membership(start_date, end_date):
    _offline_guard("fetch_zx_membership")

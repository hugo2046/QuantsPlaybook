"""ETF 暴露层单测——真实持仓 → 申万行业权重 + PIT 的纯函数。

全部合成数据，不依赖 TuShare/DB/qlib/.env。覆盖：
- ``_dedup_latest_announcement``：持仓 PIT 去重（保留最新 ann_date）
- ``aggregate_industry_weights``：持仓按申万一级行业聚合成权重（默认 nav 口径
  = ``stk_mkv_ratio``/100 占基金净值比；``disclosed`` 为旧缺陷口径）
- ``ETFExposureBuilder``：三阶段编排 + 内存缓存 + reset + 经 ``RRGStore``
  取持仓与 membership（DuckDB 增量缓存，已退役 parquet）
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_provider import _PORTFOLIO_COLS, _dedup_latest_announcement
from src.exposure import ETFExposureBuilder, aggregate_industry_weights, reconcile_mapping


def _keyword_mapping():
    """4 只 ETF 的关键词映射（drop-in 全列）。"""
    return pd.DataFrame({
        "etf_code": ["A", "B", "C", "D"],
        "etf_name": ["电子ETF", "医药ETF", "环保ETF", "银行ETF"],
        "track_code": ["t1", "t2", "t3", "t4"],
        "track_name": ["电子指数", "医药指数", "环保指数", "银行指数"],
        "sw_code": ["801080.SI", "801080.SI", "801160.SI", "801780.SI"],
        "sw_name": ["电子", "电子", "环保", "银行"],
        "match_status": ["matched", "matched", "ambiguous", "matched"],
        "match_evidence": ["", "", "", ""],
    })


def _industry_weights():
    """A 两期(取最新)、B 低置信、C 高置信覆盖、D 无持仓。"""
    return pd.DataFrame({
        "code": ["A", "A", "A", "B", "B", "C", "C"],
        "end_date": pd.to_datetime(
            ["2023-12-31", "2024-06-30", "2024-06-30", "2024-06-30",
             "2024-06-30", "2024-06-30", "2024-06-30"]),
        "ann_date": pd.to_datetime(["2024-08-30"] * 7),
        "sw_code": ["801080.SI", "801080.SI", "801770.SI", "801150.SI",
                    "801080.SI", "801730.SI", "801160.SI"],
        "weight": [0.9, 0.8, 0.15, 0.45, 0.40, 0.72, 0.20],
    })


_CODE2NAME = {"801080.SI": "电子", "801150.SI": "医药", "801770.SI": "通信",
              "801160.SI": "环保", "801730.SI": "电力设备", "801780.SI": "银行"}


def test_reconcile_mapping_override_fallback_agreement():
    """高置信覆盖 / 低置信回退 / 一致 / 无持仓回退 四分支。"""
    out = reconcile_mapping(_industry_weights(), _keyword_mapping(),
                            min_weight=0.5, code2name=_CODE2NAME).set_index("etf_code")
    # A：持仓主导电子 0.8 高置信、与关键词一致 → holdings、801080
    assert out.loc["A", "sw_code"] == "801080.SI"
    assert out.loc["A", "source"] == "holdings"
    assert bool(out.loc["A", "agreement"]) is True
    # B：持仓主导医药 0.45 低置信 → 回退关键词电子 801080、keyword、不一致
    assert out.loc["B", "sw_code"] == "801080.SI"
    assert out.loc["B", "source"] == "keyword"
    assert bool(out.loc["B", "agreement"]) is False
    # C：持仓主导电力设备 0.72 高置信、关键词=环保 → 覆盖为 801730、holdings
    assert out.loc["C", "sw_code"] == "801730.SI"
    assert out.loc["C", "sw_name"] == "电力设备"
    assert out.loc["C", "source"] == "holdings"
    # D：无持仓 → 回退关键词银行 801780、keyword、dominant_weight 缺失
    assert out.loc["D", "sw_code"] == "801780.SI"
    assert out.loc["D", "source"] == "keyword"
    assert pd.isna(out.loc["D", "dominant_weight"])


def test_reconcile_mapping_latest_snapshot():
    """同 ETF 多报告期 → 取最近一期定主导（A 取 2024-06-30 电子 0.8）。"""
    out = reconcile_mapping(_industry_weights(), _keyword_mapping(),
                            code2name=_CODE2NAME).set_index("etf_code")
    assert out.loc["A", "sw_code_holdings"] == "801080.SI"
    assert out.loc["A", "dominant_weight"] == pytest.approx(0.8)


def test_reconcile_mapping_preserves_schema():
    """输出含关键词全列 + 审计列，行数=关键词行数（drop-in）。"""
    km = _keyword_mapping()
    out = reconcile_mapping(_industry_weights(), km, code2name=_CODE2NAME)
    assert len(out) == len(km)
    assert set(km.columns).issubset(out.columns)
    for col in ["sw_code_keyword", "sw_code_holdings", "dominant_weight",
                "source", "agreement"]:
        assert col in out.columns
    # sw_code_keyword 保留原关键词值（C 原为环保 801160）
    assert out.set_index("etf_code").loc["C", "sw_code_keyword"] == "801160.SI"


def test_dedup_keeps_latest_announcement_per_holding():
    """同一 (code,end_date,symbol) 被多次披露 → 仅保留最新 ann_date 那条（含其更正后 mkv）。"""
    long = pd.DataFrame({
        "code": ["510300.SH", "510300.SH", "510300.SH"],
        "symbol": ["600000.SH", "600000.SH", "000001.SZ"],
        "ann_date": pd.to_datetime(["2024-04-30", "2024-08-30", "2024-04-30"]),
        "end_date": pd.to_datetime(["2023-12-31", "2023-12-31", "2023-12-31"]),
        "mkv": [100.0, 120.0, 50.0],  # 600000 被 8 月更正为 120
    })
    out = _dedup_latest_announcement(long)

    # 600000 两条只剩 1 条（最新 ann_date），000001 保留 → 共 2 条
    assert len(out) == 2
    row = out[out["symbol"] == "600000.SH"].iloc[0]
    assert row["ann_date"] == pd.Timestamp("2024-08-30")
    assert row["mkv"] == 120.0


def test_dedup_keeps_distinct_periods():
    """不同 end_date 的同一持仓互不影响——各报告期独立保留。"""
    long = pd.DataFrame({
        "code": ["510300.SH", "510300.SH"],
        "symbol": ["600000.SH", "600000.SH"],
        "ann_date": pd.to_datetime(["2024-04-30", "2024-08-30"]),
        "end_date": pd.to_datetime(["2023-12-31", "2024-06-30"]),  # 不同报告期
        "mkv": [100.0, 110.0],
    })
    out = _dedup_latest_announcement(long)
    assert len(out) == 2  # 两个报告期都在


# ── aggregate_industry_weights ─────────────────────────────────────────────


def test_aggregate_weights_sum_to_one_when_all_classified_disclosed():
    """[旧口径 disclosed] 持仓股全部有申万行业 → 各行业权重 = 该行业 mkv / 全持仓 mkv，和=1。"""
    portfolio = pd.DataFrame({
        "code": ["X", "X", "X"],
        "symbol": ["s1", "s2", "s3"],
        "ann_date": pd.to_datetime(["2024-04-30"] * 3),
        "end_date": pd.to_datetime(["2023-12-31"] * 3),
        "mkv": [60.0, 30.0, 10.0],
    })
    membership = pd.DataFrame({
        "code": ["s1", "s2", "s3"],
        "trade_date": pd.to_datetime(["2023-12-31"] * 3),
        "industry_code": ["801080.SI", "801080.SI", "801150.SI"],
    })
    out = aggregate_industry_weights(portfolio, membership, denominator="disclosed")

    w = out.set_index("sw_code")["weight"]
    assert w["801080.SI"] == pytest.approx(0.9)   # (60+30)/100
    assert w["801150.SI"] == pytest.approx(0.1)   # 10/100
    assert w.sum() == pytest.approx(1.0)
    # 携带 PIT 字段供下游 merge_asof
    assert set(out.columns) >= {"code", "end_date", "ann_date", "sw_code", "weight"}


def test_aggregate_uncovered_holdings_reduce_weight_sum_disclosed():
    """[旧口径 disclosed] 无申万一级行业的持仓（债/海外/现金）仍入分母，但不进任何行业 → 权重和 < 1。"""
    portfolio = pd.DataFrame({
        "code": ["X", "X"],
        "symbol": ["s1", "s_bond"],
        "ann_date": pd.to_datetime(["2024-04-30"] * 2),
        "end_date": pd.to_datetime(["2023-12-31"] * 2),
        "mkv": [90.0, 10.0],
    })
    membership = pd.DataFrame({  # 只有 s1 有行业，s_bond 缺失
        "code": ["s1"],
        "trade_date": pd.to_datetime(["2023-12-31"]),
        "industry_code": ["801080.SI"],
    })
    out = aggregate_industry_weights(portfolio, membership, denominator="disclosed")
    w = out.set_index("sw_code")["weight"]
    assert w["801080.SI"] == pytest.approx(0.9)   # 90/100，分母含 s_bond
    assert w.sum() == pytest.approx(0.9)          # 10% 未覆盖被丢弃


def test_aggregate_membership_anchor_switches_classification():
    """重分类窗口内 anchor=end_date vs ann_date 给出不同归类。"""
    portfolio = pd.DataFrame({
        "code": ["X"],
        "symbol": ["s1"],
        "ann_date": pd.to_datetime(["2024-04-30"]),
        "end_date": pd.to_datetime(["2023-12-31"]),
        "mkv": [100.0],
        "stk_mkv_ratio": [100.0],   # nav 口径（默认）依赖：该股占净值 100%
    })
    # s1 在 2023-12-31 属电子，2024-03-01 被重分类到通信
    membership = pd.DataFrame({
        "code": ["s1", "s1"],
        "trade_date": pd.to_datetime(["2023-12-31", "2024-03-01"]),
        "industry_code": ["801080.SI", "801770.SI"],
    })
    by_end = aggregate_industry_weights(portfolio, membership, anchor="end_date")
    by_ann = aggregate_industry_weights(portfolio, membership, anchor="ann_date")

    assert by_end.set_index("sw_code")["weight"].index.tolist() == ["801080.SI"]  # 电子
    assert by_ann.set_index("sw_code")["weight"].index.tolist() == ["801770.SI"]  # 通信


def _nav_portfolio() -> pd.DataFrame:
    """X 只披露 2 只持仓，各占净值 20%——即披露覆盖率 40%。

    旧口径分母 = 已披露 mkv 之和 → 权重和 = 1.0（看起来 100% 押注该行业）；
    新口径分母 = 基金净值 → 权重和 = 0.40（真实暴露）。
    """
    return pd.DataFrame({
        "code": ["X", "X"],
        "symbol": ["s1", "s2"],
        "ann_date": pd.to_datetime(["2024-04-30"] * 2),
        "end_date": pd.to_datetime(["2023-12-31"] * 2),
        "mkv": [50.0, 50.0],
        "stk_mkv_ratio": [20.0, 20.0],          # 百分数
    })


def _nav_membership() -> pd.DataFrame:
    return pd.DataFrame({
        "code": ["s1", "s2"],
        "trade_date": pd.to_datetime(["2023-12-31"] * 2),
        "industry_code": ["801080.SI", "801080.SI"],
    })


def test_aggregate_nav_denominator_reflects_shallow_disclosure():
    """浅披露 ETF 在 nav 口径下权重 = 覆盖率，不再虚高到 1.0。"""
    out = aggregate_industry_weights(_nav_portfolio(), _nav_membership())
    w = out.set_index("sw_code")["weight"]
    assert w["801080.SI"] == pytest.approx(0.40)   # (20+20)/100
    assert w.sum() == pytest.approx(0.40)


def test_aggregate_denominators_are_distinguishable():
    """同一份持仓两种口径必须给出不同结果——防 denominator 参数被接错而静默同值。

    这条断言的判别力在于：若实现忽略了 denominator、两条分支跑同一段代码，
    两个值会相等，测试立刻红。
    """
    port, memb = _nav_portfolio(), _nav_membership()
    w_nav = aggregate_industry_weights(port, memb, denominator="nav")
    w_dis = aggregate_industry_weights(port, memb, denominator="disclosed")
    assert w_nav.set_index("sw_code")["weight"]["801080.SI"] == pytest.approx(0.40)
    assert w_dis.set_index("sw_code")["weight"]["801080.SI"] == pytest.approx(1.00)


def test_aggregate_nav_uncovered_holdings_do_not_dilute_others():
    """未分类持仓（债/现金）在 nav 口径下不进任何行业，也**不**稀释其他行业。

    与旧口径的关键差别：旧口径把未分类持仓算进分母，会压低所有行业的权重；
    新口径分母是净值，未分类持仓只是「不贡献」，不改变已分类行业的数值。
    """
    port = pd.DataFrame({
        "code": ["X", "X"],
        "symbol": ["s1", "s_bond"],
        "ann_date": pd.to_datetime(["2024-04-30"] * 2),
        "end_date": pd.to_datetime(["2023-12-31"] * 2),
        "mkv": [90.0, 10.0],
        "stk_mkv_ratio": [45.0, 5.0],
    })
    memb = pd.DataFrame({          # 只有 s1 有行业
        "code": ["s1"],
        "trade_date": pd.to_datetime(["2023-12-31"]),
        "industry_code": ["801080.SI"],
    })
    out = aggregate_industry_weights(port, memb, denominator="nav")
    w = out.set_index("sw_code")["weight"]
    assert w["801080.SI"] == pytest.approx(0.45)   # 45/100，与 s_bond 无关


def test_aggregate_nav_drops_period_with_all_null_ratio():
    """stk_mkv_ratio 全空的报告期在 nav 口径下被丢弃，且**不**回退到 disclosed。

    回退会让不同口径的行混进同一张表，ETF 之间不可比——比少几行危险得多。
    该守卫是防御性的：当前缓存（12962 期）实测全空期数为 0。
    """
    port = pd.DataFrame({
        "code": ["X", "X", "Y"],
        "symbol": ["s1", "s2", "s1"],
        "ann_date": pd.to_datetime(["2024-04-30"] * 3),
        "end_date": pd.to_datetime(["2023-12-31"] * 3),
        "mkv": [50.0, 50.0, 80.0],
        "stk_mkv_ratio": [None, None, 60.0],       # X 全空，Y 正常
    })
    memb = pd.DataFrame({
        "code": ["s1", "s2"],
        "trade_date": pd.to_datetime(["2023-12-31"] * 2),
        "industry_code": ["801080.SI", "801080.SI"],
    })
    with pytest.warns(UserWarning, match="stk_mkv_ratio"):
        out = aggregate_industry_weights(port, memb, denominator="nav")
    assert "X" not in set(out["code"])             # 丢弃，不回退
    assert out.set_index("code")["weight"]["Y"] == pytest.approx(0.60)


def test_aggregate_nav_keeps_period_with_mixed_null_ratio():
    """同 (code, end_date) 部分行 stk_mkv_ratio 为 null、部分非 null → 整期保留。

    真实数据最常见的缺失形态：pandas sum 跳过 NaN，weight = 非 null 行之和。
    与全空期整期丢弃（上一条）形成对照——只有整期完全不可比才丢。
    """
    port = pd.DataFrame({
        "code": ["X", "X", "X"],
        "symbol": ["s1", "s2", "s3"],
        "ann_date": pd.to_datetime(["2024-04-30"] * 3),
        "end_date": pd.to_datetime(["2023-12-31"] * 3),
        "mkv": [50.0, 30.0, 20.0],
        "stk_mkv_ratio": [30.0, None, None],       # 仅 s1 披露 ratio
    })
    memb = pd.DataFrame({
        "code": ["s1", "s2", "s3"],
        "trade_date": pd.to_datetime(["2023-12-31"] * 3),
        "industry_code": ["801080.SI", "801150.SI", "801780.SI"],
    })
    out = aggregate_industry_weights(port, memb, denominator="nav")
    assert set(out["code"]) == {"X"}               # 整期保留，未触发全空丢弃
    assert out["weight"].sum() == pytest.approx(0.30)   # 非 null 之和 30/100


def test_aggregate_rejects_unknown_denominator():
    port, memb = _nav_portfolio(), _nav_membership()
    with pytest.raises(ValueError, match="denominator"):
        aggregate_industry_weights(port, memb, denominator="nav_ratio")


# ── ETFExposureBuilder（缓存 + 编排）────────────────────────────────────────


def _fake_portfolio_atom(code, start, end) -> pd.DataFrame:
    """RRGStore range 原子取数的桩——code==X 返回其 (2023-12-31) 两条持仓（忽略窗口，
    复刻 ``fetch_etf_portfolio_range`` 一次返回该 code 全部报告期；store 自分发到缺口季度）。"""
    if code == "X":
        return pd.DataFrame({
            "code": ["X", "X"],
            "symbol": ["s1", "s2"],
            "ann_date": pd.to_datetime(["2024-04-30", "2024-04-30"]),
            "end_date": pd.to_datetime(["2023-12-31", "2023-12-31"]),
            "mkv": [70.0, 30.0],
            "amount": [0.0, 0.0],
            "stk_mkv_ratio": [70.0, 30.0],   # 占净值百分比（nav 口径默认），与 mkv 占比一致
            "stk_float_ratio": [0.0, 0.0],
        }, columns=_PORTFOLIO_COLS)
    return pd.DataFrame(columns=_PORTFOLIO_COLS)


def _fake_membership_raw(start, end) -> pd.DataFrame:
    """store_mod.fetch_sw_membership 桩（两个位置参数 start, end）。"""
    return pd.DataFrame({
        "code": ["s1", "s2"],
        "trade_date": pd.to_datetime(["2023-12-31", "2023-12-31"]),
        "industry_code": ["801080.SI", "801150.SI"],
    })


@pytest.fixture
def patched_fetch(monkeypatch, tmp_path):
    """打桩持仓原子取数（store 层）和 membership 取数（store 层）；
    返回计数器 dict，含 'store' 键供测试复用同一 store 实例。"""
    import src.store as store_mod

    calls = {"pf": 0, "mb": 0}

    def pf_atom(code, start, end):
        calls["pf"] += 1
        return _fake_portfolio_atom(code, start, end)

    def mb(start, end):
        calls["mb"] += 1
        return _fake_membership_raw(start, end)

    monkeypatch.setattr(store_mod, "fetch_etf_portfolio_range", pf_atom)
    monkeypatch.setattr(store_mod, "fetch_sw_membership", mb)

    # 创建共享 store（用 tmp_path 下的 duckdb，避免文件冲突）
    store = store_mod.RRGStore(db_path=tmp_path / "shared.duckdb", read_only=False)
    calls["store"] = store
    yield calls
    store.close()


def test_builder_build_industry_weights_orchestrates(tmp_path, patched_fetch):
    """首次构建：store 每只 ETF 调**一次** range 原子取数，membership 调 1 次。"""
    b = ETFExposureBuilder(["X"], "2024-01-01", "2024-12-31", cache_dir=tmp_path,
                           store=patched_fetch["store"])
    w = b.build_industry_weights()
    assert patched_fetch["pf"] == 1   # 单只 ETF → 一次区间取数（替代逐季度 N 次）
    assert patched_fetch["mb"] == 1
    assert w.set_index("sw_code")["weight"].sum() == pytest.approx(1.0)


def test_builder_second_instance_hits_cache(tmp_path, patched_fetch):
    """首次构建后，第二个实例：store coverage 避免重取持仓；membership DuckDB 水位避免重取。"""
    store = patched_fetch["store"]
    ETFExposureBuilder(["X"], "2024-01-01", "2024-12-31",
                       cache_dir=tmp_path, store=store).build_industry_weights()
    assert patched_fetch["pf"] == 1
    assert patched_fetch["mb"] == 1

    # 第二个实例复用同一 store（已有 portfolio + membership DuckDB coverage）
    w2 = ETFExposureBuilder(["X"], "2024-01-01", "2024-12-31",
                            cache_dir=tmp_path, store=store).build_industry_weights()
    assert patched_fetch["pf"] == 1   # 未增加：store coverage 命中
    assert patched_fetch["mb"] == 1   # 未增加：membership DuckDB 水位命中
    assert w2.set_index("sw_code")["weight"].sum() == pytest.approx(1.0)


def test_builder_refresh_forces_refetch(tmp_path, patched_fetch):
    """refresh=True 强制重新取持仓和 membership（均走 RRGStore/DuckDB bypass）。"""
    b = ETFExposureBuilder(["X"], "2024-01-01", "2024-12-31", cache_dir=tmp_path,
                           store=patched_fetch["store"])
    b.build_industry_weights()
    b.build_industry_weights(refresh=True)
    assert patched_fetch["pf"] == 2   # 两轮各一次区间取数
    assert patched_fetch["mb"] == 2


def test_builder_reset_clears_in_memory_state(tmp_path, patched_fetch):
    """reset() 清空内存缓存 → _membership/_portfolio/_industry_weights 均为 None。
    持仓 + membership 由 store DuckDB 管（coverage/水位仍在），不重抓远程。"""
    b = ETFExposureBuilder(["X"], "2024-01-01", "2024-12-31", cache_dir=tmp_path,
                           store=patched_fetch["store"])
    b.build_industry_weights()
    assert b._membership is not None
    assert b._portfolio is not None

    b.reset()
    # 内存缓存被清空
    assert b._membership is None
    assert b._portfolio is None
    assert b._industry_weights is None

    # 再次构建：store DuckDB coverage 命中，不重抓远程
    b.build_industry_weights()
    assert patched_fetch["pf"] == 1   # store coverage 命中，未增加
    assert patched_fetch["mb"] == 1   # membership DuckDB 水位命中，未增加


def test_builder_uses_store_for_portfolio(tmp_path, monkeypatch):
    """build_industry_weights 从 RRGStore 取持仓（不再 portfolio parquet）。"""
    import src.store as store_mod
    from src.exposure import ETFExposureBuilder
    from src.data_provider import _PORTFOLIO_COLS

    def fake_atom(code, start, end):
        if code == "A":
            return pd.DataFrame(
                [{"code": "A", "symbol": "s1", "ann_date": pd.Timestamp("2024-04-30"),
                  "end_date": pd.Timestamp("2024-03-31"), "mkv": 100.0, "amount": 0.0,
                  "stk_mkv_ratio": 100.0, "stk_float_ratio": 0.0}],
                columns=_PORTFOLIO_COLS)
        return pd.DataFrame(columns=_PORTFOLIO_COLS)

    monkeypatch.setattr(store_mod, "fetch_etf_portfolio_range", fake_atom)
    monkeypatch.setattr(store_mod, "fetch_sw_membership",
                        lambda start, end: pd.DataFrame({"code": ["s1"],
                            "trade_date": [pd.Timestamp("2024-03-31")],
                            "industry_code": ["801080.SI"]}))

    store = store_mod.RRGStore(db_path=tmp_path / "t.duckdb", read_only=False)
    b = ETFExposureBuilder(codes=["A"], start_date="2024-04-01", end_date="2024-06-30",
                           cache_dir=tmp_path / "exp", store=store)
    iw = b.build_industry_weights()
    assert set(iw["code"]) == {"A"} and (iw["sw_code"] == "801080.SI").all()
    store.close()


def test_builder_denominator_passthrough(tmp_path, monkeypatch):
    """ETFExposureBuilder(..., denominator=...) 必须透传到 aggregate_industry_weights。

    浅披露 fixture（各持仓占净值 20%）：disclosed 口径权重和=1.00（已披露 mkv
    占比），nav 口径应为 0.40。若 build_industry_weights 弄丢透传，会静默回退
    默认 nav 口径，本测试立即红。
    """
    import src.store as store_mod

    def fake_atom(code, start, end):
        if code == "X":
            return pd.DataFrame(
                [{"code": "X", "symbol": s, "ann_date": pd.Timestamp("2024-04-30"),
                  "end_date": pd.Timestamp("2023-12-31"), "mkv": 50.0, "amount": 0.0,
                  "stk_mkv_ratio": 20.0, "stk_float_ratio": 0.0}
                 for s in ("s1", "s2")],
                columns=_PORTFOLIO_COLS)
        return pd.DataFrame(columns=_PORTFOLIO_COLS)

    monkeypatch.setattr(store_mod, "fetch_etf_portfolio_range", fake_atom)
    monkeypatch.setattr(store_mod, "fetch_sw_membership",
                        lambda start, end: pd.DataFrame(
                            {"code": ["s1", "s2"],
                             "trade_date": [pd.Timestamp("2023-12-31")] * 2,
                             "industry_code": ["801080.SI", "801150.SI"]}))

    store = store_mod.RRGStore(db_path=tmp_path / "t.duckdb", read_only=False)
    b = ETFExposureBuilder(codes=["X"], start_date="2024-01-01", end_date="2024-12-31",
                           store=store, denominator="disclosed")
    assert b.build_industry_weights()["weight"].sum() == pytest.approx(1.00)  # nav 给 0.40
    store.close()


def test_builder_membership_uses_store(tmp_path, monkeypatch):
    """load_membership 经 RRGStore.get_membership（不再走旧成分 parquet）。"""
    import src.store as store_mod
    from src.exposure import ETFExposureBuilder
    from src.data_provider import _PORTFOLIO_COLS

    def fake_portfolio_atom(code, start, end):
        if code == "A":
            return pd.DataFrame(
                [{"code": "A", "symbol": "s1", "ann_date": pd.Timestamp("2024-04-30"),
                  "end_date": pd.Timestamp("2024-03-31"), "mkv": 100.0, "amount": 0.0,
                  "stk_mkv_ratio": 100.0, "stk_float_ratio": 0.0}], columns=_PORTFOLIO_COLS)
        return pd.DataFrame(columns=_PORTFOLIO_COLS)

    mem_calls = []

    def fake_membership(start, end):
        mem_calls.append((str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())))
        return pd.DataFrame({"code": ["s1"], "trade_date": [pd.Timestamp("2024-03-31")],
                             "industry_code": ["801080.SI"]})

    monkeypatch.setattr(store_mod, "fetch_etf_portfolio_range", fake_portfolio_atom)
    monkeypatch.setattr(store_mod, "fetch_sw_membership", fake_membership)

    store = store_mod.RRGStore(db_path=tmp_path / "t.duckdb", read_only=False)
    b = ETFExposureBuilder(codes=["A"], start_date="2024-04-01", end_date="2024-06-30",
                           store=store)
    iw = b.build_industry_weights()
    assert (iw["sw_code"] == "801080.SI").all()
    assert len(mem_calls) == 1
    store.close()

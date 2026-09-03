"""Backtest 层单测——4 个纯函数 + vbt 主入口冒烟。

全部使用合成数据，不依赖 DB/qlib/.env/vectorbt 之外的任何外部资源。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    build_exposure_panel,
    build_overseas_share,
    build_report_universe_panel,
    build_representative_panel,
    filter_top_per_industry,
    mask_listed,
    plot_cumulative_vs_benchmark,
    portfolio_stats,
    run_signal_backtest,
    run_vbt_backtest,
)


@pytest.fixture
def mapping() -> pd.DataFrame:
    """4 只 ETF, 2 个 track, 2 个 SW：A1/A2 → trackA → 电子；B1/B2 → trackB → 医药。"""
    return pd.DataFrame({
        "etf_code": ["A1", "A2", "B1", "B2"],
        "track_code": ["trackA", "trackA", "trackB", "trackB"],
        "sw_code": ["801080.SI", "801080.SI", "801150.SI", "801150.SI"],
    })


@pytest.fixture
def amount() -> pd.DataFrame:
    """30 日 amount——A1 永远 top；A2 D5 之后才上市；B2 D15 之后量大。"""
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame(np.random.rand(30, 4) * 1e7 + 1e6, index=dates,
                      columns=["A1", "A2", "B1", "B2"])
    df.loc[: dates[4], "A2"] = np.nan
    df.loc[:, "A1"] += 5e7
    df.loc[dates[15:], "B2"] += 3e7
    return df


@pytest.fixture
def meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "list_date": pd.to_datetime(
                ["2023-01-01", "2024-01-08", "2023-06-01", "2023-06-01"]
            ),
        },
        index=["A1", "A2", "B1", "B2"],
    )


def test_representative_panel_picks_top_amount(amount, mapping):
    """长串 A1 amount 永远 top → trackA 始终 A1；其它 idx 受 lookback 影响。"""
    rebal = amount.index[[5, 15, 25]]
    reps = build_representative_panel(amount, mapping, rebal, lookback=5)
    assert (reps["trackA"] == "A1").all(), "A1 amount 永远最高，应全程代表 trackA"
    assert reps.shape == (3, 2)


def test_representative_panel_skips_unlisted_etf(amount, mapping):
    """A2 在 D5 之前未上市 (amount=NaN)，那时不能成为 trackA 代表。"""
    rebal = amount.index[[2]]
    reps = build_representative_panel(amount, mapping, rebal, lookback=5)
    assert reps.loc[rebal[0], "trackA"] == "A1"


def test_filter_top_per_industry_keeps_only_top_n(amount, mapping):
    """每个 SW 行业仅保留过去 N 日均成交额 top_n 只代表；落选 track 置 NaN。"""
    rebal = amount.index[[25]]  # 用尾日，B2 已有量加成
    reps = build_representative_panel(amount, mapping, rebal, lookback=5)
    # 此时 reps 含两条 track（trackA→A1、trackB→B2 或 B1），SW 是 电子+医药
    # top_n=1 时每 SW 各留 1 只 → 不应有 track 落选（每 SW 本就 1 个 track）
    out_n1 = filter_top_per_industry(reps, amount, mapping, top_n=1, lookback=5)
    assert out_n1.notna().sum(axis=1).iloc[0] == 2  # 两 SW × 1 代表 = 2 条 track 保留


def test_filter_top_per_industry_drops_when_two_tracks_share_sw():
    """两个 track 同属一个 SW 行业 + top_n=1 → 仅保留 amount 高的那条 track。"""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    codes = ["X1", "X2"]  # X1 trackA, X2 trackB, 但同 SW
    mapping = pd.DataFrame({
        "etf_code": codes,
        "track_code": ["trackA", "trackB"],  # 两 track
        "sw_code": ["801080.SI", "801080.SI"],  # 同 SW
    })
    # X1 量始终 > X2 → trackA 应胜出
    amount = pd.DataFrame({"X1": [1e8] * 10, "X2": [1e7] * 10}, index=dates)
    reps = build_representative_panel(amount, mapping, dates[[5]], lookback=3)
    out = filter_top_per_industry(reps, amount, mapping, top_n=1, lookback=3)
    assert out.loc[dates[5], "trackA"] == "X1"  # 留下
    assert pd.isna(out.loc[dates[5], "trackB"])  # 落选


@pytest.fixture
def industry_weights() -> pd.DataFrame:
    """A1 两个报告期持仓：23Q4(电子0.7/医药0.3, 公告4/30)、24H1(电子1.0, 公告8/30)。"""
    return pd.DataFrame({
        "code": ["A1", "A1", "A1"],
        "end_date": pd.to_datetime(["2023-12-31", "2023-12-31", "2024-06-30"]),
        "ann_date": pd.to_datetime(["2024-04-30", "2024-04-30", "2024-08-30"]),
        "sw_code": ["801080.SI", "801150.SI", "801080.SI"],
        "weight": [0.7, 0.3, 1.0],
    })


def test_exposure_panel_uses_continuous_weights(industry_weights):
    """E_t 取连续行业权重（非 0/1）；某换仓日取 ann_date≤t 的最近报告期。"""
    reps_panel = pd.DataFrame(
        {"trackA": ["A1"]}, index=pd.DatetimeIndex(["2024-05-15"])
    )
    exp = build_exposure_panel(reps_panel, industry_weights)
    e = exp[pd.Timestamp("2024-05-15")]
    # 2024-05-15 只能看到 23Q4（公告 4/30≤5/15），24H1 公告 8/30 未到
    assert e.loc["A1", "801080.SI"] == pytest.approx(0.7)
    assert e.loc["A1", "801150.SI"] == pytest.approx(0.3)


def test_exposure_panel_pit_picks_latest_disclosed_period(industry_weights):
    """换仓日推进后，PIT 切到更近的报告期（24H1）。"""
    reps_panel = pd.DataFrame(
        {"trackA": ["A1"]}, index=pd.DatetimeIndex(["2024-09-15"])
    )
    exp = build_exposure_panel(reps_panel, industry_weights)
    e = exp[pd.Timestamp("2024-09-15")]
    # 24H1 公告 8/30≤9/15 → 切到电子=1.0，且不含上一期的医药
    assert e.loc["A1", "801080.SI"] == pytest.approx(1.0)
    assert e.loc["A1"].get("801150.SI", 0.0) == pytest.approx(0.0)


def test_exposure_panel_excludes_unannounced(industry_weights):
    """换仓日早于任何公告日 → 该 ETF 无可见持仓，E_t 不含它（本例整日被跳过）。"""
    reps_panel = pd.DataFrame(
        {"trackA": ["A1"]}, index=pd.DatetimeIndex(["2024-01-15"])
    )
    exp = build_exposure_panel(reps_panel, industry_weights)
    # 2024-01-15 早于 4/30 → A1 无 PIT 持仓 → 该换仓日被跳过
    assert pd.Timestamp("2024-01-15") not in exp


def test_pit_includes_holdings_announced_exactly_on_rebalance_day():
    """公告日**恰好等于**换仓日的持仓必须可见——PIT 边界的「不要太晚」这一侧。

    与 test_exposure_panel_excludes_unannounced 成对：那条守住「别用未来
    数据」（ann_date > d 不可见），这条守住「当天公告当天就该看得见」。
    只守单侧的边界断言看起来像在守边界，实际漏掉一天。
    """
    d = pd.Timestamp("2024-03-29")
    iw = pd.DataFrame({
        "code": ["A"],
        "end_date": pd.to_datetime(["2023-12-31"]),
        "ann_date": [d],                       # 公告日 == 换仓日
        "sw_code": ["801080.SI"],
        "weight": [0.9],
    })
    panel = pd.DataFrame({"t1": ["A"]}, index=pd.DatetimeIndex([d]))
    exposures = build_exposure_panel(panel, iw)
    assert d in exposures, "公告日等于换仓日的持仓应当可见（边界为闭区间）"
    assert exposures[d].loc["A", "801080.SI"] == pytest.approx(0.9)


def test_pit_excludes_holdings_announced_one_day_after():
    """公告日晚于换仓日 1 天即不可见——PIT 边界的「不要太早」这一侧。

    与 test_pit_includes_holdings_announced_exactly_on_rebalance_day 成对
    ——两侧合起来才是完整的 PIT 边界。
    """
    d = pd.Timestamp("2024-03-29")
    iw = pd.DataFrame({
        "code": ["A"],
        "end_date": pd.to_datetime(["2023-12-31"]),
        "ann_date": [d + pd.Timedelta(days=1)],
        "sw_code": ["801080.SI"],
        "weight": [0.9],
    })
    panel = pd.DataFrame({"t1": ["A"]}, index=pd.DatetimeIndex([d]))
    assert build_exposure_panel(panel, iw) == {}, "尚未公告的持仓不得进入暴露矩阵"


def test_mask_listed_zeros_unlisted_and_renormalizes(meta):
    """A2 未上市日权重→0；其它 ETF 权重重归一化到 sum=1。"""
    dates = pd.DatetimeIndex(["2024-01-03", "2024-01-15", "2024-01-29"])
    w = pd.DataFrame(
        [[0.5, 0.5, 0.0, 0.0],   # D[0]: A2 未上市
         [0.3, 0.3, 0.2, 0.2],   # D[1]: 全已上市
         [0.0, 0.5, 0.5, 0.0]],
        index=dates, columns=["A1", "A2", "B1", "B2"],
    )
    out = mask_listed(w, meta)
    # D[0]: A2 (list_date=2024-01-08) > D[0]=2024-01-03 → A2=0, A1 从 0.5 归一到 1.0
    assert out.loc[dates[0], "A2"] == 0.0
    assert out.loc[dates[0], "A1"] == pytest.approx(1.0)
    # D[1] D[2]: 全已上市，原权重不变
    assert out.loc[dates[1]].sum() == pytest.approx(1.0)
    np.testing.assert_array_almost_equal(out.loc[dates[1]].values, [0.3, 0.3, 0.2, 0.2])


def test_mask_listed_all_zero_row_stays_zero(meta):
    """全空行（warm-up 月）应保持全 0，不触发除零。"""
    dates = pd.DatetimeIndex(["2024-01-03"])
    w = pd.DataFrame([[0.0, 0.0, 0.0, 0.0]], index=dates, columns=meta.index)
    out = mask_listed(w, meta)
    assert (out.iloc[0] == 0.0).all()


def test_mask_listed_keeps_weight_on_listing_day_itself():
    """上市**当日**的权重必须保留——`>= list_date` 的「不要太晚」一侧。

    与 test_mask_listed_zeros_unlisted_and_renormalizes 成对：那条守住上市前
    清零，这条守住上市首日不被误杀。变异 `>=` → `>` 曾能存活正是缺这一条。
    """
    d = pd.Timestamp("2024-03-29")
    meta = pd.DataFrame({"list_date": [d]}, index=["A"])
    w = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([d]))
    out = mask_listed(w, meta)
    assert out.loc[d, "A"] == pytest.approx(1.0), "上市首日应可交易，权重不得被清零"


def test_mask_listed_zeros_weight_one_day_before_listing():
    """上市前一日权重清零——「不要太早」一侧。

    与 test_mask_listed_keeps_weight_on_listing_day_itself 成对。
    """
    d = pd.Timestamp("2024-03-29")
    meta = pd.DataFrame({"list_date": [d]}, index=["A"])
    w = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([d - pd.Timedelta(days=1)]))
    out = mask_listed(w, meta)
    assert out.iloc[0]["A"] == pytest.approx(0.0), "上市前不得持有"


def test_run_vbt_backtest_returns_single_group_portfolio():
    """vbt 多标的回测：cash_sharing+group_by → pf.value() 是单条曲线。"""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    codes = ["A", "B"]
    np.random.seed(42)
    close = pd.DataFrame(
        100 + np.cumsum(np.random.randn(30, 2) * 0.5, axis=0),
        index=dates, columns=codes,
    )
    adj = close * 1.01
    target = pd.DataFrame(index=dates, columns=codes, dtype=float)
    target.iloc[5] = [0.5, 0.5]
    target.iloc[15] = [1.0, 0.0]

    pf = run_vbt_backtest(target, adj, fees=5e-4, slippage=5e-4, init_cash=1e6)
    val = pf.value()
    # 单组合 → pf.value() 是 Series, 不是 DataFrame
    assert isinstance(val, pd.Series), "cash_sharing+group_by 应返回单条净值曲线（Series）"
    assert len(val) == 30
    assert val.iloc[0] == pytest.approx(1e6, rel=1e-2)  # 初始资金附近


def test_run_vbt_no_phantom_jump_on_heterogeneous_adjustment():
    """换仓日不得因复权因子异质注入幻收益（回归 close/price 价基错配 bug）。

    构造：``adjclose`` 全程 FLAT（零真实收益）但两资产复权水平不同（A=12、B=15）。
    第 0 日全仓 A、第 3 日换仓全仓 B。价格不动 + 零费 → 净值应恒定、换仓日收益≈0。
    旧实现（``close=adjclose, price=原价``）会在换仓日跳变 ~25%（= 15/12 复权因子比）。
    """
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    adjclose = pd.DataFrame({"A": [12.0] * 6, "B": [15.0] * 6}, index=dates)
    target = pd.DataFrame(index=dates, columns=["A", "B"], dtype=float)
    target.iloc[0] = [1.0, 0.0]
    target.iloc[3] = [0.0, 1.0]

    pf = run_vbt_backtest(target, adjclose, fees=0.0, slippage=0.0, init_cash=1e6)
    val = pf.value()
    # 零收益 + 零费 → 净值恒定，换仓日无任何跳变
    assert val.pct_change().abs().max() < 1e-9, "换仓日出现幻跳变（价基错配回归）"
    assert val.iloc[-1] == pytest.approx(val.iloc[0]), "末净值应=初净值（零收益零费）"


def test_run_vbt_fills_at_adjopen_not_adjclose():
    """成交价=adjopen：换仓按开盘价成交，执行日 open→close 涨幅计入新持仓（贴实盘）。

    信号在第 0 日给出（内部 ``shift(1)`` → 第 1 日执行）；执行日 ``adjopen[A]=10``、
    ``adjclose[A]=11``（日内 +10%，同后复权基）。满仓 A → 按开盘成交：执行日组合按
    收盘估值 +10%；退化为按收盘成交则执行日 ≈0%。
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    adjclose = pd.DataFrame({"A": [10.0, 11.0, 11.0, 11.0]}, index=dates)
    adjopen = pd.DataFrame({"A": [10.0, 10.0, 11.0, 11.0]}, index=dates)
    target = pd.DataFrame(index=dates, columns=["A"], dtype=float)
    target.iloc[0] = [1.0]  # 第 0 日信号 → shift(1) 第 1 日开盘执行

    pf_open = run_vbt_backtest(target, adjclose, adjopen, fees=0.0, slippage=0.0, init_cash=1e6)
    pf_close = run_vbt_backtest(target, adjclose, fees=0.0, slippage=0.0, init_cash=1e6)
    # 开盘成交：执行日买在 open=10、收在 close=11 → 当日 +10%
    assert pf_open.value().iloc[1] == pytest.approx(1.1e6, rel=1e-3), "应按开盘价 10 成交"
    # 收盘成交（adjopen=None）：买在 close=11、收在 close=11 → 当日 0%
    assert pf_close.value().iloc[1] == pytest.approx(1e6, rel=1e-3), "退化应按收盘价 11 成交"


def test_run_vbt_executes_next_day_no_lookahead():
    """无日内前视：换仓日 d 的目标在 d+1 开盘成交，d→d+1 隔夜跳空不被信号方吃到。

    A 在第 2 日跳空翻倍（open=close=20）；信号在第 1 日给出。正确口径（``shift(1)``，
    第 2 日执行）→ 第 2 日按 open=20 建仓、收在 20，组合不吃 10→20 跳空，value 全程
    ≈ 初始资金。若同日前视（第 1 日按 open=10 建仓），value 第 2 日会翻倍。
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    px = pd.DataFrame({"A": [10.0, 10.0, 20.0, 20.0]}, index=dates)
    target = pd.DataFrame(index=dates, columns=["A"], dtype=float)
    target.iloc[1] = [1.0]  # 第 1 日信号

    pf = run_vbt_backtest(target, px, px, fees=0.0, slippage=0.0, init_cash=1e6)
    val = pf.value()
    assert val.iloc[1] == pytest.approx(1e6, rel=1e-3), "第 1 日仍持现金（订单顺延到第 2 日）"
    assert val.iloc[2] == pytest.approx(1e6, rel=1e-3), "第 2 日按 open=20 建仓，未吃隔夜跳空（无前视）"


def test_pit_latest_holdings_picks_latest_announced():
    """ann_date <= t 中取 end_date 最大的那期；未来公告期不可见。

    含两只 ETF 验证 per-code 独立：同一截止日 X 切到 24H1、Y 仍停在 23Q4
    （其 24H1 公告在未来），证明一只 ETF 的最近报告期不会渗到另一只。
    """
    from src.backtest import _pit_latest_holdings
    iw = pd.DataFrame({
        "code":     ["X", "X", "X", "Y", "Y"],
        "end_date": pd.to_datetime([
            "2023-12-31", "2024-06-30", "2024-12-31",  # X 三期
            "2023-12-31", "2024-06-30",                 # Y 两期
        ]),
        "ann_date": pd.to_datetime([
            "2024-01-20", "2024-08-20", "2025-01-20",  # X：24H1 已公告(8/20)
            "2024-01-20", "2025-08-20",                 # Y：24H1 公告在未来(25/8/20)
        ]),
        "sw_code":  ["801080.SI", "801080.SI", "801150.SI",
                     "801080.SI", "801150.SI"],
        "weight":   [0.9, 0.8, 0.7, 0.6, 0.5],
    })
    snap = _pit_latest_holdings(iw, pd.Timestamp("2024-09-01"))
    latest = snap.set_index("code")["end_date"]
    # per-code 独立：X 切到 24H1，Y 的 24H1 公告未到 → 仍停在 23Q4
    assert latest.loc["X"] == pd.Timestamp("2024-06-30")
    assert latest.loc["Y"] == pd.Timestamp("2023-12-31")
    assert _pit_latest_holdings(iw, pd.Timestamp("2024-01-01")).empty  # 无可见期


@pytest.fixture
def dyn_amount() -> pd.DataFrame:
    """动态代表测试用 amount：覆盖 E/F/G/H/I/J 六只 ETF，40 个交易日。"""
    dates = pd.date_range("2024-05-01", periods=40, freq="B")
    cols = ["E", "F", "G", "H", "I", "J"]
    df = pd.DataFrame(5e6, index=dates, columns=cols, dtype=float)
    df["H"] += 9e7   # H 成交额远高于 I（同行业 top_n 测试）
    df["I"] += 1e7
    return df


def test_dynamic_panel_pit_no_future_leak(dyn_amount):
    """ETF 主导行业跨期漂移：2024-06 换仓只能看到过去期(电子)，看不到未来期(医药)。"""
    from src.backtest import build_dynamic_representative_panel
    iw = pd.DataFrame({
        "code":     ["E", "E", "E", "E"],
        "end_date": pd.to_datetime(["2023-12-31", "2023-12-31",
                                    "2024-12-31", "2024-12-31"]),
        "ann_date": pd.to_datetime(["2024-01-20", "2024-01-20",
                                    "2025-01-20", "2025-01-20"]),
        "sw_code":  ["801080.SI", "801150.SI", "801150.SI", "801080.SI"],
        "weight":   [0.9, 0.1, 0.9, 0.1],   # 旧期主导电子；新期主导医药
    })
    reps = build_dynamic_representative_panel(
        iw, dyn_amount, pd.DatetimeIndex([pd.Timestamp("2024-06-17")]),
        lookback=20, min_dominant_weight=0.5, top_n=1)
    # 2025-01-20 公告期不可见 → 用 2023 期主导电子 801080，绝不是医药 801150
    assert "801080.SI" in reps.columns
    assert reps.iloc[0]["801080.SI"] == "E"
    assert "801150.SI" not in reps.columns


def test_dynamic_panel_purity_gate(dyn_amount):
    """主导占比 < min_dominant_weight 的 ETF（宽基/分散）出局；高占比进池。"""
    from src.backtest import build_dynamic_representative_panel
    iw = pd.DataFrame({
        "code":     ["F", "F", "F", "G"],
        "end_date": pd.to_datetime(["2024-03-31"] * 4),
        "ann_date": pd.to_datetime(["2024-04-20"] * 4),
        "sw_code":  ["801080.SI", "801150.SI", "801770.SI", "801080.SI"],
        "weight":   [0.3, 0.3, 0.3, 0.9],   # F 分散(max0.3)；G 集中(0.9)
    })
    reps = build_dynamic_representative_panel(
        iw, dyn_amount, pd.DatetimeIndex([pd.Timestamp("2024-06-17")]),
        min_dominant_weight=0.5, top_n=1)
    used = set(reps.iloc[0].dropna())
    assert "G" in used and "F" not in used


def test_dynamic_panel_top_n_by_liquidity(dyn_amount):
    """同一动态行业两只 ETF，top_n=1 留过去 lookback 日均成交额高的那只(H>I)。"""
    from src.backtest import build_dynamic_representative_panel
    iw = pd.DataFrame({
        "code":     ["H", "I"],
        "end_date": pd.to_datetime(["2024-03-31", "2024-03-31"]),
        "ann_date": pd.to_datetime(["2024-04-20", "2024-04-20"]),
        "sw_code":  ["801080.SI", "801080.SI"],
        "weight":   [0.9, 0.9],
    })
    reps = build_dynamic_representative_panel(
        iw, dyn_amount, pd.DatetimeIndex([pd.Timestamp("2024-06-17")]),
        min_dominant_weight=0.5, top_n=1)
    assert reps.iloc[0]["801080.SI"] == "H"
    # top_n=2：同行业保留两只，rank0 列纯 sw_code，rank1 列加 #1 后缀
    reps2 = build_dynamic_representative_panel(
        iw, dyn_amount, pd.DatetimeIndex([pd.Timestamp("2024-06-17")]),
        min_dominant_weight=0.5, top_n=2)
    assert reps2.iloc[0]["801080.SI"] == "H"        # rank0 = 高成交额
    assert reps2.iloc[0]["801080.SI#1"] == "I"      # rank1 = 次高，#1 列


def test_dynamic_panel_absent_when_no_pit_holdings(dyn_amount):
    """ann_date 全在换仓日之后的 ETF（未披露）当日缺席。"""
    from src.backtest import build_dynamic_representative_panel
    iw = pd.DataFrame({
        "code":     ["J"],
        "end_date": pd.to_datetime(["2024-12-31"]),
        "ann_date": pd.to_datetime(["2025-01-20"]),   # 晚于换仓日
        "sw_code":  ["801080.SI"],
        "weight":   [0.95],
    })
    reps = build_dynamic_representative_panel(
        iw, dyn_amount, pd.DatetimeIndex([pd.Timestamp("2024-06-17")]),
        min_dominant_weight=0.5, top_n=1)
    assert reps.empty or "J" not in set(reps.iloc[0].dropna())


def test_portfolio_stats_returns_7_fields():
    """portfolio_stats 输出 7 个字段（对齐研报表 13）。"""
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    codes = ["A"]
    close = pd.DataFrame(100 + np.arange(20).reshape(-1, 1) * 0.5, index=dates, columns=codes)
    target = pd.DataFrame(index=dates, columns=codes, dtype=float)
    target.iloc[5] = [1.0]

    pf = run_vbt_backtest(target, close, fees=0.0, slippage=0.0, init_cash=1e6)
    stats = portfolio_stats(pf)
    expected = {"累计收益率", "年化收益率", "最大回撤", "年化波动率",
                "夏普比率", "Calmar 比率", "总成交笔数"}
    assert set(stats.index) == expected


# ---------------------------------------------------------------------------
# run_signal_backtest —— §2 行业指数 boolean 信号 → 等权组合回测
# ---------------------------------------------------------------------------


@pytest.fixture
def sw_price() -> pd.DataFrame:
    """3 个行业 × 40 个交易日的合成行业指数长表（含 close/open）。"""
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    inds = ["801010.SI", "801030.SI", "801080.SI"]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.normal(0, 0.01, (40, 3)), axis=0)),
        index=dates, columns=inds,
    )
    open_ = close.shift(1).bfill() * (1 + np.random.normal(0, 0.002, close.shape))
    return (
        close.stack().rename("close").to_frame()
        .join(open_.stack().rename("open"))
        .reset_index()
        .rename(columns={"level_0": "trade_date", "level_1": "code"})
    )


@pytest.fixture
def signal(sw_price) -> pd.DataFrame:
    """每 10 日翻一次持仓集合，含一段全 False（空仓）。"""
    dates = sw_price["trade_date"].drop_duplicates().sort_values()
    inds = ["801010.SI", "801030.SI", "801080.SI"]
    sig = pd.DataFrame(False, index=pd.DatetimeIndex(dates), columns=inds)
    sig.iloc[0:10, 0] = True            # 持 1 个
    sig.iloc[10:20, [0, 1]] = True      # 持 2 个
    sig.iloc[20:30] = False             # 空仓
    sig.iloc[30:40] = True              # 持 3 个
    return sig


def test_run_signal_backtest_equal_weight_and_trades(signal, sw_price):
    """boolean 信号 → 等权组合 → vbt 跑通；on_change 有成交且持仓集合切换都触发换仓。"""
    pf = run_signal_backtest(signal, sw_price)
    stats = portfolio_stats(pf)
    assert set(stats.index) == {
        "累计收益率", "年化收益率", "最大回撤", "年化波动率",
        "夏普比率", "Calmar 比率", "总成交笔数",
    }
    # 4 段持仓（1→2→空→3）→ 必有多笔成交
    assert pf.orders.count() > 0


def test_run_signal_backtest_on_change_fewer_orders_than_daily(signal, sw_price):
    """on_change（仅变化日换仓）成交笔数应远少于 daily（每日再平衡）。"""
    n_on_change = run_signal_backtest(signal, sw_price, rebalance="on_change").orders.count()
    n_daily = run_signal_backtest(signal, sw_price, rebalance="daily").orders.count()
    assert n_on_change < n_daily


def test_run_signal_backtest_rejects_bad_rebalance(signal, sw_price):
    with pytest.raises(ValueError, match="rebalance"):
        run_signal_backtest(signal, sw_price, rebalance="monthly")


def test_run_signal_backtest_rejects_disjoint_columns(signal, sw_price):
    """信号行业列与价格 code 列无交集 → ValueError（列对齐硬契约）。"""
    bad = signal.rename(columns={c: c.replace(".SI", ".XX") for c in signal.columns})
    with pytest.raises(ValueError, match="无交集"):
        run_signal_backtest(bad, sw_price)


def test_plot_cumulative_vs_benchmark_uses_passed_price(signal, sw_price):
    """传入的基准价格序列 → Benchmark 曲线末值 = 价格首末比（自定义基准生效）。"""
    pf = run_signal_backtest(signal, sw_price)
    dates = pf.returns().index
    bench = pd.Series(np.linspace(1000.0, 1200.0, len(dates)), index=dates)  # +20%
    fig = plot_cumulative_vs_benchmark(pf, bench)
    traces = {t.name: np.asarray(t.y) for t in fig.data if t.name}
    assert {"Benchmark", "Value"} <= set(traces)
    assert float(traces["Benchmark"][-1]) == pytest.approx(1.2, abs=1e-6)


def test_plot_cumulative_vs_benchmark_rejects_disjoint_index(signal, sw_price):
    """基准价格与策略收益率 index 无重合 → ValueError。"""
    pf = run_signal_backtest(signal, sw_price)
    bench = pd.Series([1.0, 2.0], index=pd.to_datetime(["1999-01-01", "1999-01-02"]))
    with pytest.raises(ValueError, match="无重合日"):
        plot_cumulative_vs_benchmark(pf, bench)


# ============ build_dynamic_etf_industry（动态 etf→行业 long 表，单一事实源）============


def _iw_simple() -> pd.DataFrame:
    """两只 ETF、单一已披露报告期的 industry_weights。"""
    return pd.DataFrame({
        "code":     ["E", "E", "F", "F"],
        "end_date": pd.to_datetime(["2024-03-31"] * 4),
        "ann_date": pd.to_datetime(["2024-04-20"] * 4),
        "sw_code":  ["801080.SI", "801150.SI", "801770.SI", "801010.SI"],
        "weight":   [0.9, 0.1, 0.6, 0.4],
    })


def test_build_dynamic_etf_industry_argmax_and_columns():
    """每 (换仓日, ETF) 取 PIT 持仓 argmax 主导行业；列含 dominant_weight。"""
    from src.backtest import build_dynamic_etf_industry
    out = build_dynamic_etf_industry(
        _iw_simple(), pd.DatetimeIndex([pd.Timestamp("2024-06-17")]),
        min_dominant_weight=0.5)
    assert list(out.columns) == ["rebal_date", "etf_code", "sw_code", "dominant_weight"]
    row = out.set_index("etf_code")
    assert row.loc["E", "sw_code"] == "801080.SI"   # argmax 0.9
    assert row.loc["F", "sw_code"] == "801770.SI"   # argmax 0.6


def test_build_dynamic_etf_industry_tie_break_smallest_sw():
    """主导权重平局 → 取 sw_code 字典序最小（确定性）。"""
    from src.backtest import build_dynamic_etf_industry
    iw = pd.DataFrame({
        "code":     ["E", "E"],
        "end_date": pd.to_datetime(["2024-03-31"] * 2),
        "ann_date": pd.to_datetime(["2024-04-20"] * 2),
        "sw_code":  ["801150.SI", "801080.SI"],
        "weight":   [0.5, 0.5],
    })
    out = build_dynamic_etf_industry(
        iw, pd.DatetimeIndex([pd.Timestamp("2024-06-17")]))
    assert out.set_index("etf_code").loc["E", "sw_code"] == "801080.SI"


def test_build_dynamic_etf_industry_purity_gate():
    """主导占比 < min_dominant_weight 的 ETF 出局，高占比保留。"""
    from src.backtest import build_dynamic_etf_industry
    iw = pd.DataFrame({
        "code":     ["F", "G"],
        "end_date": pd.to_datetime(["2024-03-31"] * 2),
        "ann_date": pd.to_datetime(["2024-04-20"] * 2),
        "sw_code":  ["801080.SI", "801010.SI"],
        "weight":   [0.3, 0.9],
    })
    out = build_dynamic_etf_industry(
        iw, pd.DatetimeIndex([pd.Timestamp("2024-06-17")]), min_dominant_weight=0.5)
    assert set(out["etf_code"]) == {"G"}


def test_build_dynamic_etf_industry_empty_when_no_visible_period():
    """ann_date > 换仓日（未披露）→ 空表，列齐全（供下游安全消费）。"""
    from src.backtest import build_dynamic_etf_industry
    out = build_dynamic_etf_industry(
        _iw_simple(), pd.DatetimeIndex([pd.Timestamp("2024-01-01")]))
    assert out.empty
    assert list(out.columns) == ["rebal_date", "etf_code", "sw_code", "dominant_weight"]


# ---------------------------------------------------------------------------
# build_report_universe_panel —— 研报 §3.1 全池候选（同指数 ADV top1 去重）
# ---------------------------------------------------------------------------


@pytest.fixture
def report_meta() -> pd.DataFrame:
    """5 只 ETF：A1/A2 同跟 idxA；B1/B2 同跟 idxB；M 是货基（index_code 缺失）。"""
    return pd.DataFrame(
        {
            "index_code": ["idxA", "idxA", "idxB", "idxB", None],
            "list_date": pd.to_datetime(["2023-01-01"] * 5),
        },
        index=["A1", "A2", "B1", "B2", "M"],
    )


@pytest.fixture
def report_amount() -> pd.DataFrame:
    """20 日成交额：A2 恒高于 A1；B1/B2 前 10 日相等（测平局），后 10 日 B2 更高。"""
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    df = pd.DataFrame(1.0e7, index=dates, columns=["A1", "A2", "B1", "B2", "M"])
    df["A2"] = 5.0e7
    df.loc[dates[10:], "B2"] = 9.0e7
    return df


def test_report_universe_picks_top_adv_per_index(report_meta, report_amount):
    """同一跟踪指数下只保留 ADV 最高的那只（研报 §3.1 去重规则）。"""
    rebal = pd.DatetimeIndex([report_amount.index[15]])
    panel = build_report_universe_panel(report_meta, report_amount, rebal, lookback=5)
    assert panel.loc[rebal[0], "idxA"] == "A2"
    assert panel.loc[rebal[0], "idxB"] == "B2"


def test_report_universe_tie_breaks_by_ts_code(report_meta, report_amount):
    """ADV 完全相等时取 ts_code 字典序最小者——保证结果不依赖 groupby 内部顺序。"""
    rebal = pd.DatetimeIndex([report_amount.index[5]])   # 前 10 日 B1==B2
    panel = build_report_universe_panel(report_meta, report_amount, rebal, lookback=5)
    assert panel.loc[rebal[0], "idxB"] == "B1"


def test_report_universe_has_no_lookahead(report_meta, report_amount):
    """篡改换仓日**之后**的成交额，该日选择结果必须不变。"""
    rebal = pd.DatetimeIndex([report_amount.index[5]])
    before = build_report_universe_panel(report_meta, report_amount, rebal, lookback=5)

    tampered = report_amount.copy()
    tampered.loc[report_amount.index[6]:, "B2"] = 1.0e12   # 未来暴涨
    after = build_report_universe_panel(report_meta, tampered, rebal, lookback=5)

    pd.testing.assert_frame_equal(before, after)


def test_report_universe_drops_missing_index_code(report_meta, report_amount):
    """index_code 缺失的 ETF（货基）不进池子。"""
    rebal = pd.DatetimeIndex([report_amount.index[15]])
    panel = build_report_universe_panel(report_meta, report_amount, rebal, lookback=5)
    assert "M" not in panel.to_numpy().ravel().tolist()


def test_report_universe_skips_index_with_no_turnover(report_meta, report_amount):
    """某指数在该期全无成交 → 该格 NaN（列仍存在），且不影响其他指数列。"""
    amt = report_amount.copy()
    d1, d2 = report_amount.index[5], report_amount.index[15]
    amt.loc[:d1, ["B1", "B2"]] = 0.0   # d1 之前（含）零成交，之后恢复原值
    panel = build_report_universe_panel(report_meta, amt,
                                        pd.DatetimeIndex([d1, d2]), lookback=5)
    assert pd.isna(panel.loc[d1, "idxB"])          # 列存在（d2 有选择），格为 NaN
    assert panel.loc[d2, "idxB"] == "B2"           # 恢复成交后 B2 以更高 ADV 入选
    assert panel.loc[d1, "idxA"] == "A2"


def test_report_universe_feeds_exposure_panel(report_meta, report_amount):
    """形状契约：输出能直接喂 build_exposure_panel 且产出非空 E_t。"""
    rebal = pd.DatetimeIndex([report_amount.index[15]])
    panel = build_report_universe_panel(report_meta, report_amount, rebal, lookback=5)

    iw = pd.DataFrame({
        "code": ["A2", "B2"],
        "end_date": pd.to_datetime(["2023-12-31", "2023-12-31"]),
        "ann_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "sw_code": ["CI005025.CI", "CI005018.CI"],
        "weight": [0.8, 0.7],
    })
    exposures = build_exposure_panel(panel, iw)
    assert rebal[0] in exposures
    assert set(exposures[rebal[0]].index) == {"A2", "B2"}


def test_report_universe_requires_index_code_column(report_amount):
    """meta 缺 index_code 必须显式报错，不得静默退化成「每只 ETF 自成一组」。"""
    bad_meta = pd.DataFrame({"list_date": pd.to_datetime(["2023-01-01"])}, index=["A1"])
    with pytest.raises(ValueError, match="index_code"):
        build_report_universe_panel(bad_meta, report_amount,
                                    pd.DatetimeIndex([report_amount.index[5]]))


# ── 境外（港股）ETF 剔除 ───────────────────────────────────────────────────


def _hk_portfolio() -> pd.DataFrame:
    """3 只 ETF：HKPURE 全港股、MIXED 一半港股、APURE 全 A 股。"""
    return pd.DataFrame({
        "code":    ["HKPURE", "HKPURE", "MIXED", "MIXED", "APURE", "APURE"],
        "symbol":  ["0700.HK", "0939.HK", "0700.HK", "600000.SH",
                    "600000.SH", "000001.SZ"],
        "ann_date": pd.to_datetime(["2024-01-20"] * 6),
        "end_date": pd.to_datetime(["2023-12-31"] * 6),
        "mkv":     [60.0, 40.0, 50.0, 50.0, 70.0, 30.0],
    })


def test_overseas_share_computes_hk_fraction():
    """境外占比 = 港股持仓 mkv / 该报告期已披露持仓 mkv 之和。"""
    out = build_overseas_share(_hk_portfolio())
    s = out.set_index("code")["overseas_share"]
    assert s["HKPURE"] == pytest.approx(1.0)
    assert s["MIXED"] == pytest.approx(0.5)
    assert s["APURE"] == pytest.approx(0.0)
    assert set(out.columns) == {"code", "end_date", "ann_date", "overseas_share"}


def test_report_universe_excludes_overseas_dominant_etf():
    """境外占比 > 阈值的 ETF 不进候选池；恰好 == 阈值的保留（严格大于）。"""
    meta = pd.DataFrame(
        {"index_code": ["idxHK", "idxMIX", "idxA"],
         "list_date": pd.to_datetime(["2023-01-01"] * 3)},
        index=["HKPURE", "MIXED", "APURE"],
    )
    dates = pd.date_range("2024-02-01", periods=5, freq="B")
    amount = pd.DataFrame(1.0e7, index=dates, columns=["HKPURE", "MIXED", "APURE"])
    rebal = pd.DatetimeIndex([dates[-1]])

    panel = build_report_universe_panel(
        meta, amount, rebal, lookback=3,
        overseas_share=build_overseas_share(_hk_portfolio()),
        overseas_threshold=0.5,
    )
    held = set(panel.loc[rebal[0]].dropna())
    assert "HKPURE" not in held          # 1.0 > 0.5 → 剔除
    assert "MIXED" in held               # 0.5 不大于 0.5 → 保留
    assert "APURE" in held


def test_report_universe_overseas_exclusion_happens_before_dedup():
    """剔除必须发生在同指数去重**之前**——否则港股冠军会白占掉一个指数槽位。

    构造：HKPURE 与 ARUNNER 跟踪同一指数，HKPURE 的 ADV 更高。若先去重后剔除，
    该指数当期就没有代表；正确顺序下 ARUNNER 应递补上位。
    """
    meta = pd.DataFrame(
        {"index_code": ["idxSame", "idxSame"],
         "list_date": pd.to_datetime(["2023-01-01"] * 2)},
        index=["HKPURE", "ARUNNER"],
    )
    dates = pd.date_range("2024-02-01", periods=5, freq="B")
    amount = pd.DataFrame({"HKPURE": 9.0e7, "ARUNNER": 1.0e7}, index=dates)
    rebal = pd.DatetimeIndex([dates[-1]])

    port = pd.concat([_hk_portfolio(), pd.DataFrame({
        "code": ["ARUNNER"], "symbol": ["600519.SH"],
        "ann_date": pd.to_datetime(["2024-01-20"]),
        "end_date": pd.to_datetime(["2023-12-31"]), "mkv": [100.0],
    })], ignore_index=True)

    panel = build_report_universe_panel(
        meta, amount, rebal, lookback=3,
        overseas_share=build_overseas_share(port), overseas_threshold=0.5,
    )
    assert panel.loc[rebal[0], "idxSame"] == "ARUNNER"


def test_report_universe_overseas_exclusion_is_pit():
    """公告日晚于换仓日的报告期不参与判定——无前视。

    HKPURE 的港股占比要到 2024-01-20 公告后才可见；在 2024-01-10 换仓时
    尚无任何可见报告期，此时不得剔除（信息还不存在）。
    """
    meta = pd.DataFrame(
        {"index_code": ["idxHK"], "list_date": pd.to_datetime(["2023-01-01"])},
        index=["HKPURE"],
    )
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    amount = pd.DataFrame(1.0e7, index=dates, columns=["HKPURE"])
    ovs = build_overseas_share(_hk_portfolio())      # ann_date = 2024-01-20

    early = pd.DatetimeIndex([dates[5]])             # 2024-01-08，公告前
    p_early = build_report_universe_panel(meta, amount, early, lookback=3,
                                          overseas_share=ovs, overseas_threshold=0.5)
    assert p_early.loc[early[0], "idxHK"] == "HKPURE"   # 尚不可见 → 不剔除


def test_report_universe_without_overseas_share_is_unchanged():
    """不传 overseas_share 时行为与改动前完全一致（向后兼容）。"""
    meta = pd.DataFrame(
        {"index_code": ["idxHK", "idxA"],
         "list_date": pd.to_datetime(["2023-01-01"] * 2)},
        index=["HKPURE", "APURE"],
    )
    dates = pd.date_range("2024-02-01", periods=5, freq="B")
    amount = pd.DataFrame(1.0e7, index=dates, columns=["HKPURE", "APURE"])
    rebal = pd.DatetimeIndex([dates[-1]])
    panel = build_report_universe_panel(meta, amount, rebal, lookback=3)
    assert set(panel.loc[rebal[0]].dropna()) == {"HKPURE", "APURE"}


# ---------------------------------------------------------------------------
# §3.11 ①：mask_listed 不得把半仓行强推到满仓（保持原行和）
# ---------------------------------------------------------------------------

def test_mask_listed_preserves_partial_position_when_nothing_masked():
    """etf_capped 在 k=3 时行和 = 0.6（3 × 0.2）；无未上市 ETF 时 mask_listed 必须原样返回。

    让它变红：无条件把行归一到 1（旧实现 → 0.333 各）。
    """
    d = pd.Timestamp("2024-06-28")
    meta = pd.DataFrame({"list_date": [pd.Timestamp("2020-01-01")] * 3}, index=["A", "B", "C"])
    w = pd.DataFrame({"A": [0.2], "B": [0.2], "C": [0.2]}, index=pd.DatetimeIndex([d]))
    out = mask_listed(w, meta)
    assert out.loc[d].sum() == pytest.approx(0.6)
    np.testing.assert_allclose(out.loc[d].to_numpy(), [0.2, 0.2, 0.2])


def test_mask_listed_redistributes_masked_weight_but_keeps_row_sum():
    """双侧：真有 ETF 被上市日清零时，剩余权重按**原行和**重摊（0.6 → 两只各 0.3），
    而不是归一到 1（各 0.5）。

    让它变红：改回归一到 1；或清零后不重摊（各 0.2、行和掉到 0.4）。
    """
    d = pd.Timestamp("2024-06-28")
    meta = pd.DataFrame({"list_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"),
                                       pd.Timestamp("2025-01-01")]}, index=["A", "B", "C"])  # C 未上市
    w = pd.DataFrame({"A": [0.2], "B": [0.2], "C": [0.2]}, index=pd.DatetimeIndex([d]))
    out = mask_listed(w, meta)
    assert out.loc[d, "C"] == 0.0
    assert out.loc[d].sum() == pytest.approx(0.6)
    np.testing.assert_allclose(out.loc[d, ["A", "B"]].to_numpy(), [0.3, 0.3])

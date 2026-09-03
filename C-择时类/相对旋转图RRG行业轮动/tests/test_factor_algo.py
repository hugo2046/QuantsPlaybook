"""RRG + 扩散指标单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factor_algo import classify_quadrant, compute_rrg, compute_rs, compute_rs_momentum, compute_rs_ratio, diffusion_count_ratio, diffusion_value_weighted, equal_weight_benchmark
from src.factor_algo import _resolve_industries


def test_equal_weight_benchmark_known_returns():
    """等权组合净值 = 各列日收益等权平均后累乘；首日 NAV=1.0。"""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    # A: 日收益 +10%, 0%；B: 日收益 0%, +10% → 截面等权日收益 0, +5%, +5%
    price = pd.DataFrame({"A": [10.0, 11.0, 11.0], "B": [20.0, 20.0, 22.0]}, index=dates)
    nav = equal_weight_benchmark(price)
    expected = pd.Series([1.0, 1.05, 1.1025], index=dates)
    pd.testing.assert_series_equal(nav, expected)


def test_equal_weight_benchmark_skips_nan_column():
    """某列某日 NaN（未上市/停牌）→ 该日从截面等权均值中剔除（skipna）。"""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    # B 前两日未上市(NaN)；day1 截面只剩 A 的 +20%
    price = pd.DataFrame({"A": [10.0, 12.0, 12.0], "B": [np.nan, np.nan, 50.0]}, index=dates)
    nav = equal_weight_benchmark(price)
    expected = pd.Series([1.0, 1.2, 1.2], index=dates)
    pd.testing.assert_series_equal(nav, expected)


def test_rs_constant_ratio_equals_100x():
    """price / benchmark = c → RS = 100c"""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    price = pd.DataFrame({"A": [10.0] * 10, "B": [20.0] * 10}, index=dates)
    benchmark = pd.Series([5.0] * 10, index=dates)
    rs = compute_rs(price, benchmark)
    expected = pd.DataFrame({"A": [200.0] * 10, "B": [400.0] * 10}, index=dates)
    pd.testing.assert_frame_equal(rs, expected)


def test_rs_accepts_dataframe_benchmark():
    """benchmark 是单列 DataFrame 也应能接受"""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    price = pd.DataFrame({"A": [10.0] * 5}, index=dates)
    benchmark_df = pd.DataFrame({"BM": [5.0] * 5}, index=dates)
    rs = compute_rs(price, benchmark_df)
    assert rs.iloc[0, 0] == 200.0


def test_rsratio_flat_converges_to_100():
    """price ≡ benchmark → 任何 lookback / smooth 下 RS_Ratio 都应是 100"""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    benchmark = pd.Series(np.linspace(100.0, 200.0, 500), index=dates)
    price = pd.DataFrame({"A": benchmark.values, "B": benchmark.values}, index=dates)
    rs_ratio = compute_rs_ratio(price, benchmark, lookback=220, smooth_window=20)
    # warm-up: 前 lookback + smooth - 1 = 239 行 NaN，之后应等于 100
    valid = rs_ratio.iloc[239:]
    assert np.allclose(valid.values, 100.0, atol=1e-9)


def test_rsratio_outperform_above_100():
    """price 持续跑赢 benchmark → RS_Ratio 持续 > 100"""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    benchmark = pd.Series(np.linspace(100.0, 110.0, 500), index=dates)
    # price 每日跑赢 benchmark 0.05%
    out_factor = np.cumprod(np.full(500, 1.0005))
    price = pd.DataFrame({"A": benchmark.values * out_factor}, index=dates)
    rs_ratio = compute_rs_ratio(price, benchmark, lookback=220, smooth_window=20)
    valid = rs_ratio.iloc[239:]
    assert (valid["A"] > 100.0).all()


def test_rsmom_underperform_below_100():
    """price 加速跑输 → RS_Ratio 单调下降 → RS_Mom 持续 < 100

    注：等速跑输时 RS_Ratio 为常数，RS_Mom = 100 恰好；需要加速跑输才能使
    RS_Ratio 持续下降，从而 RS_Mom < 100。
    """
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    benchmark = pd.Series(np.ones(500) * 100.0, index=dates)
    # price 以加速下跌速率跑输 benchmark（daily rate 从 0.999 逐步下降至 ~0.989）
    t = np.arange(500)
    daily_rate = 0.999 - t * 0.00002
    under_factor = np.cumprod(daily_rate)
    price = pd.DataFrame({"A": 100.0 * under_factor}, index=dates)
    rs_mom = compute_rs_momentum(
        price, benchmark, lookback_ratio=220, lookback_mom=60, smooth_window=20
    )
    # warm-up: lookback_ratio + (smooth-1) + lookback_mom + (smooth-1) = 318 行 NaN
    valid = rs_mom.iloc[318:]
    assert (valid["A"] < 100.0).all()


def test_compute_rrg_returns_two_frames_matching_individual_calls(
    synthetic_industry_prices: pd.DataFrame, synthetic_benchmark: pd.Series
):
    """聚合函数结果应与分别调用 compute_rs_ratio + compute_rs_momentum 一致"""
    rs_ratio_agg, rs_mom_agg = compute_rrg(
        synthetic_industry_prices, synthetic_benchmark,
        lookback_ratio=220, lookback_mom=60, smooth_window=20,
    )
    rs_ratio_single = compute_rs_ratio(
        synthetic_industry_prices, synthetic_benchmark, lookback=220, smooth_window=20
    )
    rs_mom_single = compute_rs_momentum(
        synthetic_industry_prices, synthetic_benchmark,
        lookback_ratio=220, lookback_mom=60, smooth_window=20,
    )
    pd.testing.assert_frame_equal(rs_ratio_agg, rs_ratio_single)
    pd.testing.assert_frame_equal(rs_mom_agg, rs_mom_single)


def test_warmup_nan_count(
    synthetic_industry_prices: pd.DataFrame, synthetic_benchmark: pd.Series
):
    """起始 lookback_ratio + lookback_mom + 2*(smooth-1) = 318 行 RS_Mom 必为 NaN，
    第 318 行起首次出现非 NaN。"""
    _, rs_mom = compute_rrg(
        synthetic_industry_prices, synthetic_benchmark,
        lookback_ratio=220, lookback_mom=60, smooth_window=20,
    )
    expected_nan_rows = 220 + (20 - 1) + 60 + (20 - 1)  # = 318
    assert rs_mom.iloc[:expected_nan_rows].isna().all().all()
    assert not rs_mom.iloc[expected_nan_rows].isna().any()


def test_insufficient_length_raises():
    """数据长度刚好不足（恰好 lookback_ratio + lookback_mom + 2*(smooth-1) = 318 行）→ 抛 ValueError

    边界值 319 = lookback_ratio + lookback_mom + 2*(smooth-1) + 1
    """
    dates = pd.date_range("2020-01-01", periods=318, freq="B")
    price = pd.DataFrame(
        np.random.default_rng(0).random((318, 2)) * 100,
        index=dates, columns=["A", "B"],
    )
    benchmark = pd.Series(
        np.random.default_rng(1).random(318) * 100, index=dates
    )
    with pytest.raises(ValueError, match="数据长度不足"):
        compute_rrg(
            price, benchmark, lookback_ratio=220, lookback_mom=60, smooth_window=20
        )


def test_minimum_length_just_passes():
    """数据长度刚好达标（319 行）→ 不抛异常，且 rs_mom 最后一行非 NaN"""
    dates = pd.date_range("2020-01-01", periods=319, freq="B")
    rng = np.random.default_rng(2)
    price = pd.DataFrame(
        np.cumprod(1.0 + rng.normal(0.0003, 0.012, (319, 2)), axis=0) * 100,
        index=dates, columns=["A", "B"],
    )
    benchmark = price.mean(axis=1)
    _, rs_mom = compute_rrg(
        price, benchmark, lookback_ratio=220, lookback_mom=60, smooth_window=20
    )
    assert not rs_mom.iloc[-1].isna().any()


def test_classify_quadrant_four_corners():
    """四象限四个典型点分别归类 1/2/3/4"""
    rs_ratio = pd.DataFrame({"A": [110.0, 90.0, 90.0, 110.0]})
    rs_mom = pd.DataFrame({"A": [110.0, 110.0, 90.0, 90.0]})
    quad = classify_quadrant(rs_ratio, rs_mom)
    assert list(quad["A"]) == [1.0, 2.0, 3.0, 4.0]


def test_classify_quadrant_nan_on_boundary_or_missing():
    """RS_Ratio 或 RS_Mom 等于 100 / 含 NaN → 返回 NaN"""
    rs_ratio = pd.DataFrame({"A": [100.0, np.nan, 110.0]})
    rs_mom = pd.DataFrame({"A": [110.0, 110.0, np.nan]})
    quad = classify_quadrant(rs_ratio, rs_mom)
    assert quad["A"].isna().all()


def _make_monotone_close(n_days: int, n_stocks: int, direction: str) -> pd.DataFrame:
    """生成纯单调价序列：direction='up' 单调上行；'down' 单调下行。"""
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    codes = [f"S{i}" for i in range(n_stocks)]
    if direction == "up":
        arr = np.tile(np.arange(1, n_days + 1, dtype=float).reshape(-1, 1), (1, n_stocks))
    else:
        arr = np.tile(np.arange(n_days, 0, -1, dtype=float).reshape(-1, 1), (1, n_stocks))
    return pd.DataFrame(arr, index=dates, columns=codes)


def test_diffusion_count_all_up_equals_1():
    """全行业全成分股单调上涨 → 平滑后扩散 = 1.0"""
    close = _make_monotone_close(300, 3, "up")
    membership = pd.DataFrame(
        [["IND1"] * 3] * 300, index=close.index, columns=close.columns
    )
    diff = diffusion_count_ratio(close, membership, lookback=220, smooth_window=20)
    # warm-up: lookback + smooth - 1 = 239
    valid = diff.iloc[239:]["IND1"]
    assert (valid == 1.0).all()


def test_diffusion_count_all_down_equals_0():
    """全行业全成分股单调下跌 → 扩散 = 0.0"""
    close = _make_monotone_close(300, 3, "down")
    membership = pd.DataFrame(
        [["IND1"] * 3] * 300, index=close.index, columns=close.columns
    )
    diff = diffusion_count_ratio(close, membership, lookback=220, smooth_window=20)
    valid = diff.iloc[239:]["IND1"]
    assert (valid == 0.0).all()


def test_diffusion_count_half_half():
    """一半涨一半跌 → 扩散 = 0.5"""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    up = np.arange(1, 301, dtype=float)
    down = np.arange(300, 0, -1, dtype=float)
    close = pd.DataFrame(
        {"S1": up, "S2": up, "S3": down, "S4": down}, index=dates
    )
    membership = pd.DataFrame(
        [["IND1"] * 4] * 300, index=dates, columns=close.columns
    )
    diff = diffusion_count_ratio(close, membership, lookback=220, smooth_window=20)
    assert np.allclose(diff.iloc[239:]["IND1"], 0.5)


def test_diffusion_membership_changes():
    """成分股调出后，分母仅基于剩余成分股"""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    up = np.arange(1, 51, dtype=float)
    down = np.arange(50, 0, -1, dtype=float)
    close = pd.DataFrame({"S1": up, "S2": up, "S3": down}, index=dates)
    # 前 30 行：3 股票均在 IND1；后 20 行：S3 调出
    membership = pd.DataFrame(
        {
            "S1": ["IND1"] * 50,
            "S2": ["IND1"] * 50,
            "S3": ["IND1"] * 30 + [np.nan] * 20,
        },
        index=dates,
    )
    diff = diffusion_count_ratio(close, membership, lookback=10, smooth_window=5)
    # warm-up: lookback + smooth - 1 = 14。15-29 是 3 股票期，期望 2/3
    assert np.allclose(diff.iloc[15:30]["IND1"], 2.0 / 3.0)
    # 35-49 是 S3 调出后，期望 2/2 = 1.0
    assert np.allclose(diff.iloc[35:50]["IND1"], 1.0)


def test_diffusion_warmup_is_nan():
    """起始 lookback + smooth - 1 行整体 NaN"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    close = pd.DataFrame(
        rng.uniform(50, 150, size=(300, 3)),
        index=dates, columns=["S1", "S2", "S3"]
    )
    membership = pd.DataFrame(
        [["IND1"] * 3] * 300, index=dates, columns=close.columns
    )
    diff = diffusion_count_ratio(close, membership, lookback=220, smooth_window=20)
    assert diff.iloc[:239]["IND1"].isna().all()
    assert not np.isnan(diff.iloc[239]["IND1"])


def test_diffusion_value_weighted_concentration():
    """大市值股票上涨、小市值下跌 → 扩散显著 > 0.5"""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    up = np.arange(1, 301, dtype=float)
    down = np.arange(300, 0, -1, dtype=float)
    close = pd.DataFrame({"S1": up, "S2": down}, index=dates)
    membership = pd.DataFrame(
        {"S1": ["IND1"] * 300, "S2": ["IND1"] * 300}, index=dates
    )
    # S1（上涨）市值 900，S2（下跌）市值 100 → 加权扩散应 = 900/1000 = 0.9
    free_float_mv = pd.DataFrame(
        {"S1": [900.0] * 300, "S2": [100.0] * 300}, index=dates
    )
    diff = diffusion_value_weighted(
        close, membership, free_float_mv, lookback=220, smooth_window=20
    )
    assert np.allclose(diff.iloc[239:]["IND1"], 0.9)


def test_diffusion_value_weighted_equal_mv_matches_count_version():
    """所有股票市值相等 → 加权扩散应与数量占比扩散一致"""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    up = np.arange(1, 301, dtype=float)
    down = np.arange(300, 0, -1, dtype=float)
    close = pd.DataFrame(
        {"S1": up, "S2": up, "S3": down, "S4": down}, index=dates
    )
    membership = pd.DataFrame(
        [["IND1"] * 4] * 300, index=dates, columns=close.columns
    )
    free_float_mv = pd.DataFrame(
        np.ones((300, 4)) * 100.0, index=dates, columns=close.columns
    )
    diff_value = diffusion_value_weighted(
        close, membership, free_float_mv, lookback=220, smooth_window=20
    )
    diff_count = diffusion_count_ratio(close, membership, lookback=220, smooth_window=20)
    pd.testing.assert_series_equal(
        diff_value.iloc[239:]["IND1"], diff_count.iloc[239:]["IND1"]
    )


def test_classify_quadrant_raises_on_column_mismatch():
    """rs_ratio 与 rs_mom 列不同 → ValueError（避免静默生成幻列）"""
    rs_ratio = pd.DataFrame({"A": [110.0], "B": [110.0]})
    rs_mom = pd.DataFrame({"A": [110.0], "C": [110.0]})
    with pytest.raises(ValueError, match="列不一致"):
        classify_quadrant(rs_ratio, rs_mom)


def test_compute_rs_ratio_insufficient_length_raises():
    """compute_rs_ratio 长度不足 → ValueError（避免静默全 NaN）"""
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    price = pd.DataFrame({"A": np.random.default_rng(0).random(200) * 100}, index=dates)
    benchmark = pd.Series(np.random.default_rng(1).random(200) * 100, index=dates)
    with pytest.raises(ValueError, match="数据长度不足"):
        compute_rs_ratio(price, benchmark, lookback=220, smooth_window=20)


def test_compute_rs_momentum_insufficient_length_raises():
    """compute_rs_momentum 长度不足 → ValueError"""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    price = pd.DataFrame({"A": np.random.default_rng(0).random(300) * 100}, index=dates)
    benchmark = pd.Series(np.random.default_rng(1).random(300) * 100, index=dates)
    with pytest.raises(ValueError, match="数据长度不足"):
        compute_rs_momentum(
            price, benchmark, lookback_ratio=220, lookback_mom=60, smooth_window=20
        )


def test_compute_rrg_disjoint_calendars_raises():
    """price 与 benchmark 等长但日历错位 → ValueError"""
    rng = np.random.default_rng(0)
    price_dates = pd.date_range("2018-01-01", periods=500, freq="B")
    bench_dates = pd.date_range("2021-01-01", periods=500, freq="B")
    price = pd.DataFrame(
        np.cumprod(1.0 + rng.normal(0.0003, 0.01, (500, 2)), axis=0) * 100,
        index=price_dates, columns=["A", "B"],
    )
    benchmark = pd.Series(
        np.cumprod(1.0 + rng.normal(0.0003, 0.01, 500)) * 100,
        index=bench_dates,
    )
    with pytest.raises(ValueError, match="日历重合不足"):
        compute_rrg(price, benchmark, lookback_ratio=220, lookback_mom=60, smooth_window=20)


def test_compute_rrg_empty_benchmark_dataframe_raises():
    """benchmark 是 0 列 DataFrame → 友好 ValueError，而非 IndexError"""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    price = pd.DataFrame(
        np.random.default_rng(0).random((500, 2)) * 100,
        index=dates, columns=["A", "B"],
    )
    benchmark_empty = pd.DataFrame(index=dates)
    with pytest.raises(ValueError, match="benchmark DataFrame 必须至少有 1 列"):
        compute_rrg(price, benchmark_empty)


def test_resolve_industries_handles_mixed_types():
    """_resolve_industries 在 membership 含混合 str/int 时不抛 TypeError"""
    membership = pd.DataFrame(
        {"S1": ["IND1", "IND1"], "S2": [11, 11], "S3": ["IND1", np.nan]}
    )
    result = _resolve_industries(membership)
    # 排序后包含 11 与 'IND1' 两个值，不报错即通过
    assert len(result) == 2
    assert 11 in result and "IND1" in result


def test_diffusion_value_weighted_nan_mv_excludes_stock():
    """成分股 free_float_mv 为 NaN → 从分子分母双方剔除（仅基于剩余成分股）"""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    up = np.arange(1, 301, dtype=float)
    down = np.arange(300, 0, -1, dtype=float)
    close = pd.DataFrame({"S1": up, "S2": up, "S3": down}, index=dates)
    membership = pd.DataFrame(
        {"S1": ["IND1"] * 300, "S2": ["IND1"] * 300, "S3": ["IND1"] * 300},
        index=dates,
    )
    # S1 mv=NaN（缺失），S2 mv=100（上涨），S3 mv=100（下跌）
    free_float_mv = pd.DataFrame(
        {"S1": [np.nan] * 300, "S2": [100.0] * 300, "S3": [100.0] * 300},
        index=dates,
    )
    diff = diffusion_value_weighted(
        close, membership, free_float_mv, lookback=220, smooth_window=20
    )
    # 仅 S2/S3 参与；S2 上涨/S3 下跌 → 100/(100+100) = 0.5
    assert np.allclose(diff.iloc[239:]["IND1"], 0.5)


def test_compute_rs_no_warn_when_indices_match():
    """price.index 与 benchmark.index 完全一致 → 不触发 warn，不 ffill"""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    price = pd.DataFrame(
        np.random.default_rng(0).random((300, 2)) * 100 + 50,
        index=dates, columns=["A", "B"],
    )
    benchmark = pd.Series(
        np.random.default_rng(1).random(300) * 100 + 50, index=dates
    )
    # pytest.warns(None) 在 pytest 8+ 已弃用；改用 warnings.catch_warnings 显式断言
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        rs = compute_rs(price, benchmark)
    assert len([w for w in caught if issubclass(w.category, UserWarning)]) == 0
    # 形状锚定到 benchmark.index，且无 NaN（输入完全干净）
    assert rs.index.equals(benchmark.index)
    assert not rs.isna().any().any()


def test_compute_rs_warns_and_ffills_on_missing_dates():
    """price 在 benchmark.index 上缺日 → 触发 UserWarning + ffill 兜底"""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    # benchmark 是完整 300 天；price 缺中段 3 天
    benchmark = pd.Series(rng.random(300) * 100 + 50, index=dates)
    missing_idx = [dates[100], dates[150], dates[200]]
    price = pd.DataFrame(
        rng.random((300, 2)) * 100 + 50,
        index=dates, columns=["A", "B"],
    ).drop(missing_idx)

    with pytest.warns(UserWarning, match="缺 3 个交易日"):
        rs = compute_rs(price, benchmark)

    # ffill 后整张表零 NaN（price 在 dates[0] 就有值，前向填充覆盖所有缺日）
    assert rs.index.equals(benchmark.index)
    assert not rs.isna().any().any()
    # 缺日的 rs 应该等于前一日的 ffill 值除以当日 benchmark
    # 即 rs[dates[100]] = price[dates[99]] / benchmark[dates[100]] * 100
    expected_at_missing = price.loc[dates[99]] / benchmark.loc[dates[100]] * 100.0
    pd.testing.assert_series_equal(
        rs.loc[dates[100]], expected_at_missing,
        check_names=False,
    )


def test_compute_rs_ffill_leaves_head_nan_when_no_source():
    """price 头部缺日且无前值 → ffill 救不了，头部保留 NaN（不假装填上）"""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    benchmark = pd.Series(np.arange(1, 11, dtype=float), index=dates)
    # price 仅含后 7 天，前 3 天缺失且无可 ffill 的前值
    price = pd.DataFrame(
        {"A": np.arange(4, 11, dtype=float)},
        index=dates[3:],
    )
    with pytest.warns(UserWarning, match="缺 3 个交易日"):
        rs = compute_rs(price, benchmark)
    # 前 3 行 NaN（ffill 无源），后 7 行有值
    assert rs.iloc[:3].isna().all().all()
    assert not rs.iloc[3:].isna().any().any()

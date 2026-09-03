"""测试共享 fixture：合成行情数据，覆盖 RRG warm-up 长度。

路径配置：
    将 etf_rrg/ 加入 sys.path，使 ``from src.X import Y`` 在测试中可用。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使 ``from src.X import Y`` 在测试中可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_industry_prices() -> pd.DataFrame:
    """5 个行业 × 800 个交易日的几何布朗运动价（足够覆盖 220+60+20 warm-up）。"""
    rng = np.random.default_rng(seed=20260525)
    n_days, n_industries = 800, 5
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    ret = rng.normal(loc=0.0003, scale=0.012, size=(n_days, n_industries))
    prices = np.cumprod(1.0 + ret, axis=0) * 100.0
    return pd.DataFrame(
        prices,
        index=dates,
        columns=[f"IND{i + 1}" for i in range(n_industries)],
    )


@pytest.fixture
def synthetic_benchmark(synthetic_industry_prices: pd.DataFrame) -> pd.Series:
    """行业价的截面均值，作为行业等权基准。"""
    return synthetic_industry_prices.mean(axis=1)


@pytest.fixture
def synthetic_stock_data() -> dict:
    """
    30 个股票 × 300 个交易日，10/10/10 分属 3 个行业。

    返回 dict，含 close / membership / free_float_mv 三张宽表。
    """
    rng = np.random.default_rng(seed=20260525)
    n_days, n_stocks_per_ind = 300, 10
    industries = ["IND1", "IND2", "IND3"]
    total = len(industries) * n_stocks_per_ind
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    codes = [f"S{i:03d}" for i in range(total)]

    ret = rng.normal(loc=0.0003, scale=0.018, size=(n_days, total))
    close = pd.DataFrame(np.cumprod(1.0 + ret, axis=0) * 50.0, index=dates, columns=codes)

    membership_row = []
    for ind in industries:
        membership_row.extend([ind] * n_stocks_per_ind)
    membership = pd.DataFrame([membership_row] * n_days, index=dates, columns=codes)

    free_float_mv = pd.DataFrame(
        rng.uniform(1e8, 1e10, size=(n_days, total)), index=dates, columns=codes
    )
    return {"close": close, "membership": membership, "free_float_mv": free_float_mv}

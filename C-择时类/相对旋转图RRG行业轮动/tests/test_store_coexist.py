"""回归：store（读写）打开时，data_provider 仍能在同一 duckdb 上读。

背景：notebook 全程持有一个 RRGStore（读写模式），同时 generator→fetch_rrg_dataset
会经 _connect() 打开同一 duckdb。若 _connect 用 read_only=True，DuckDB 会抛
ConnectionException（禁止同进程混用读写/只读配置）。此 bug 仅在真实数据端到端
（notebook）暴露，合成数据单测（各用独立临时库、不并发持有 store）抓不到。
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.store import RRGStore
from src import data_provider as dp


@pytest.fixture
def snap_with_industry(tmp_path, monkeypatch):
    db = tmp_path / "etf_rrg.duckdb"
    # 先用 store 建 schema（读写），再补一张行业表，供 data_provider 读
    RRGStore(db_path=str(db), read_only=False).close()
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE sw_industry_daily(trade_date DATE, code TEXT, industry_name TEXT, close DOUBLE, open DOUBLE)")
    dates = pd.bdate_range("2021-01-04", periods=10)
    con.executemany("INSERT INTO sw_industry_daily VALUES (?,?,?,?,?)",
                    [(d.date(), "801010.SI", "农林牧渔", 100.0 + i, 99.0 + i) for i, d in enumerate(dates)])
    con.close()
    monkeypatch.setattr(dp, "DEFAULT_DB_PATH", str(db))
    return db


def test_read_while_store_open(snap_with_industry):
    # 关键：store 保持打开（读写），此时 data_provider 读同一文件不得冲突
    with RRGStore(db_path=str(snap_with_industry)):
        df = dp.get_sw_industry_data("2021-01-04", "2021-01-15", fields="close")
    assert not df.empty
    assert df["code"].iloc[0] == "801010.SI"

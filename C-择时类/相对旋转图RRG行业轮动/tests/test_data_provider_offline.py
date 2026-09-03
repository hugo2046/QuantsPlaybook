"""data_provider 离线只读版测试：用合成 duckdb，零外部依赖。"""
from __future__ import annotations
import duckdb
import pandas as pd
import pytest
from src import data_provider as dp


@pytest.fixture
def snap_db(tmp_path, monkeypatch):
    """建一个含各表若干行的合成快照，并把 DEFAULT_DB_PATH 指过去。"""
    db = tmp_path / "etf_rrg.duckdb"
    con = duckdb.connect(str(db))
    dates = pd.bdate_range("2021-01-01", periods=60)
    con.execute("CREATE TABLE trade_days(trade_date DATE)")
    con.executemany("INSERT INTO trade_days VALUES (?)", [(d.date(),) for d in dates])
    con.execute("CREATE TABLE sw_industry_daily(trade_date DATE, code TEXT, industry_name TEXT, close DOUBLE, open DOUBLE)")
    rows = [(d.date(), c, n, 100.0 + i, 99.0 + i)
            for i, d in enumerate(dates)
            for c, n in [("801010.SI", "农林牧渔"), ("801030.SI", "基础化工")]]
    con.executemany("INSERT INTO sw_industry_daily VALUES (?,?,?,?,?)", rows)
    con.execute("CREATE TABLE etf_meta(ts_code TEXT, index_code TEXT, index_name TEXT, list_date TEXT, list_status TEXT, etf_type TEXT, setup_date TEXT, mgt_fee DOUBLE)")
    con.executemany("INSERT INTO etf_meta VALUES (?,?,?,?,?,?,?,?)", [
        ("510300.SH", "000300.SH", "沪深300", "20120528", "L", "纯境内", "20120505", 0.5),
        ("159999.OF", "000905.SH", "中证500", "20200101", "L", "纯境内", "20191201", 0.5),
        ("510500.SH", None, "货基", "20130101", "L", "纯境内", "20121201", 0.3),
    ])
    con.execute("CREATE TABLE etf_daily(date DATE, code TEXT, adjclose DOUBLE, adjopen DOUBLE, amount DOUBLE)")
    con.executemany("INSERT INTO etf_daily VALUES (?,?,?,?,?)",
                    [(d.date(), "510300.SH", 4.0 + i * 0.01, 3.99 + i * 0.01, 1e8) for i, d in enumerate(dates)])
    con.close()
    monkeypatch.setattr(dp, "DEFAULT_DB_PATH", str(db))
    return db


def test_get_sw_industry_data_reads_snapshot(snap_db):
    df = dp.get_sw_industry_data(start_dt="2021-01-01", end_dt="2021-02-01", fields="close")
    assert set(df.columns) == {"trade_date", "code", "industry_name", "close"}
    assert df["code"].nunique() == 2
    assert str(df["trade_date"].dtype).startswith("datetime64")


def test_fetch_rrg_dataset_ind_avg(snap_db):
    bench, ind_close, code2name = dp.fetch_rrg_dataset(
        start_date="2021-01-01", end_date="2021-02-01", classification="sw"
    )
    assert isinstance(bench, pd.Series)
    assert list(ind_close.columns) == ["801010.SI", "801030.SI"]
    assert code2name["801010.SI"] == "农林牧渔"


def test_fetch_rrg_dataset_external_benchmark_raises(snap_db):
    with pytest.raises(dp.OfflineDataError):
        dp.fetch_rrg_dataset(benchmark_code="000852.SH", start_date="2021-01-01", end_date="2021-02-01")


def test_fetch_etf_meta_filters(snap_db):
    meta = dp.fetch_etf_meta(only_listed=True, only_etf=True, drop_money=True)
    assert "510300.SH" in meta.index          # 交易所+有指数
    assert "159999.OF" not in meta.index       # only_etf 剔 .OF
    assert "510500.SH" not in meta.index        # drop_money 剔 index_code 缺失
    assert str(meta["list_date"].dtype).startswith("datetime64")


def test_fetch_etf_daily_pivots(snap_db):
    adjclose, adjopen, amount = dp.fetch_etf_daily(["510300.SH"], "2021-01-11", "2021-02-01", lookback=5)
    assert "510300.SH" in adjclose.columns
    assert (amount.values == 1e8).all()


def test_store_guard_atoms_raise(snap_db):
    with pytest.raises(dp.OfflineDataError):
        dp.fetch_sw_membership("2021-01-01", "2021-02-01")
    with pytest.raises(dp.OfflineDataError):
        dp.fetch_etf_portfolio_range("510300.SH", "2021-01-01", "2021-02-01")

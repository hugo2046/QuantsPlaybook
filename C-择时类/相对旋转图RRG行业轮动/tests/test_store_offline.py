"""store 离线行为：缓存命中直接读，miss 触发 data_provider guard → OfflineDataError。"""
import duckdb
import pandas as pd
import pytest
from src.store import RRGStore
from src import data_provider as dp


@pytest.fixture
def db_with_membership(tmp_path):
    db = tmp_path / "etf_rrg.duckdb"
    store = RRGStore(db_path=str(db), read_only=False)  # 建 schema 需写权限
    con = store._con
    dates = pd.bdate_range("2021-01-04", periods=20)
    con.executemany("INSERT INTO sw_membership VALUES (?,?,?)",
                    [(d.date(), "000001.SZ", "801780.SI") for d in dates])
    con.execute("INSERT INTO sw_membership_coverage VALUES (1, ?, ?, now())",
                [dates[0].date(), dates[-1].date()])
    store.close()
    return db


def test_get_membership_cache_hit(db_with_membership):
    with RRGStore(db_path=str(db_with_membership)) as s:
        out = s.get_membership("2021-01-05", "2021-01-15", classification="sw")
    assert not out.empty
    assert (out["code"] == "000001.SZ").all()


def test_get_membership_miss_raises(db_with_membership):
    # 请求窗口超出 coverage → 产生 gap → 调 fetch_sw_membership guard → OfflineDataError
    with RRGStore(db_path=str(db_with_membership)) as s:
        with pytest.raises(dp.OfflineDataError):
            s.get_membership("2019-01-01", "2021-01-15", classification="sw")

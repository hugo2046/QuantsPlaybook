"""离线只读部署：快照放只读介质 / 被其他只读进程打开时，仍能正常读。

背景：离线版只读快照、从不写库。若以读写模式打开 duckdb，会有两个部署问题：
1. 快照放只读盘或 chmod 444（分发物的自然部署）→ 读写打开直接 IOException，整个项目开不了库；
2. 读写连接是跨进程独占的，任何别的进程打开该库都会互相锁死。
只读打开可同时解决两者（多个只读连接可并存）。
"""
from __future__ import annotations

import os
import stat

import duckdb
import pandas as pd
import pytest

from src import data_provider as dp
from src.store import RRGStore


@pytest.fixture
def snapshot(tmp_path):
    """建一个含 schema 与一行成分的快照，然后置为只读文件。"""
    db = tmp_path / "etf_rrg.duckdb"
    with RRGStore(db_path=str(db), read_only=False) as s:   # 建库需写权限
        dates = pd.bdate_range("2021-01-04", periods=5)
        s._con.executemany(
            "INSERT INTO sw_membership VALUES (?,?,?)",
            [(d.date(), "000001.SZ", "801780.SI") for d in dates],
        )
        s._con.execute(
            "INSERT INTO sw_membership_coverage VALUES (1, ?, ?, now())",
            [dates[0].date(), dates[-1].date()],
        )
    os.chmod(db, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 444：模拟只读介质
    yield db
    os.chmod(db, stat.S_IRUSR | stat.S_IWUSR)  # 复原以便 tmp_path 清理


def test_store_defaults_to_read_only(snapshot):
    """默认即只读——只读文件上也能开、能查。"""
    with RRGStore(db_path=str(snapshot)) as s:
        assert s.read_only is True
        out = s.get_membership("2021-01-04", "2021-01-08", classification="sw")
    assert not out.empty


def test_store_read_write_fails_on_read_only_file(snapshot):
    """反证：显式读写打开在只读文件上必然失败（这正是改默认值的原因）。"""
    with pytest.raises(duckdb.IOException):
        RRGStore(db_path=str(snapshot), read_only=False)


def test_multiple_read_only_connections_coexist(snapshot, monkeypatch):
    """store 与 data_provider 同时只读打开同一库不冲突（旧读写模式下会 ConnectionException）。"""
    monkeypatch.setattr(dp, "DEFAULT_DB_PATH", str(snapshot))
    with RRGStore(db_path=str(snapshot)):
        con = dp._connect()          # 第二个只读连接
        n = con.execute("SELECT count(*) FROM sw_membership").fetchone()[0]
        con.close()
    assert n == 5

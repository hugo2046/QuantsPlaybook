"""DEFAULT_DB_PATH 必须锚定到项目根、与 cwd 无关。

回归背景：notebook 以 example/ 为 cwd 运行时，相对路径 "data/etf_rrg.duckdb" 会错解析到
example/data/（迁移项目无此目录），导致有真实快照也抛 OfflineDataError。
本文件**不使用**会 monkeypatch DEFAULT_DB_PATH 的 fixture，专门检查默认值本身。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import data_provider as dp

_PROJECT_ROOT = Path(dp.__file__).resolve().parent.parent


def test_default_db_path_is_absolute_and_under_project_root():
    p = Path(dp.DEFAULT_DB_PATH)
    assert p.is_absolute()
    assert p == _PROJECT_ROOT / "data" / "etf_rrg.duckdb"


def test_connect_resolves_independent_of_cwd(tmp_path, monkeypatch):
    # 指向一个绝对、不存在的路径，并切到无关 cwd。若 _connect 相对 cwd 解析，报错会
    # 引用 cwd 下的相对片段而非这个绝对路径。用「不存在的路径」而非真实快照，避免测试
    # 依赖真实 1.1G 库的锁状态（如 VS Code/内核持写锁时读写连接会 IOException）——
    # 锚定绝对路径的不变量另由 test_default_db_path_is_absolute... 守。
    ghost = tmp_path / "nope" / "etf_rrg.duckdb"
    monkeypatch.setattr(dp, "DEFAULT_DB_PATH", str(ghost))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(dp.OfflineDataError) as ei:
        dp._connect()
    assert str(ghost) in str(ei.value)  # 报错引用锚定后的绝对路径，与 cwd 无关

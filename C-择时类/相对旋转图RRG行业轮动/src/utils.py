"""
Author: Hugo
Date: 2024-07-29 09:19:38
LastEditors: Hugo
Description: 通用工具函数。RRG legacy 工具（calc_zscore / standardize /
    mark_trend_tag / add_uuid_from_data）已在 plotting v2 中移除
    （spec docs/superpowers/specs/2026-05-28-rrg-plotting-v2-design.md）。
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def datetime2str(date: Any, fmt: str = "%Y%m%d") -> str:
    return pd.to_datetime(date).strftime(fmt)


def get_race_cons_dict(race_frame: pd.DataFrame) -> Dict:
    return (
        race_frame.groupby("race_name")
        .apply(lambda x: x["code"].unique().tolist())
        .to_dict()
    )


def fillnan2None(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({np.nan: None})


def to_equal_weight(mask: pd.DataFrame) -> pd.DataFrame:
    """
    boolean 选择矩阵 → 等权权重矩阵。

    :param mask: index=date, columns=标的，bool（True=选中）。
    :returns: 同形状 float；每行权重和=1（全 False 行则全 0）。
    """
    counts = mask.sum(axis=1)
    return mask.astype(float).div(counts.where(counts > 0), axis=0).fillna(0.0)


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """日频 index → 每自然月最后一个交易日。"""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)

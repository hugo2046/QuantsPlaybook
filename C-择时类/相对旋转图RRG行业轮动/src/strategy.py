"""ETF-RRG 策略信号层：日频 boolean mask 纯函数（DataFrame in / DataFrame out）。

研报覆盖 3 类信号（西部证券 §2.1–2.3）：
- A 纯 RRG：select_by_quadrant（象限内按距中心距离取 top_n）
- B 纯扩散：select_by_diffusion（扩散 top_n）
- C 扩散+RRG：select_diffusion_with_rrg（扩散 top_n 先选 → 剔除非保留象限，不补足）

所有函数输出完整日频结果；月末/周末采样交给调用者（禁内置采样逻辑）。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.factor_algo import classify_quadrant

__all__ = [
    "distance_from_center",
    "select_by_quadrant",
    "select_by_diffusion",
    "select_diffusion_with_rrg",
]


def distance_from_center(
    rs_ratio: pd.DataFrame, rs_mom: pd.DataFrame, center: float = 100.0
) -> pd.DataFrame:
    """
    RRG 平面上各点到中心 (center, center) 的欧氏距离，日频。

    :param rs_ratio: RS_Ratio 宽表，index=date, columns=industry_code。
    :param rs_mom: RS_Momentum 宽表，形状与 rs_ratio 一致。
    :param center: 中心值（研报比率法中心=100）。
    :returns: 同形状距离 DataFrame；任一输入 NaN → 该格 NaN。
    """
    return np.sqrt((rs_ratio - center) ** 2 + (rs_mom - center) ** 2)


def select_by_diffusion(diffusion: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    信号 B（研报 §2.2）：每日扩散指标 top_n。

    :param diffusion: 扩散指标宽表，index=date, columns=industry_code。
    :param top_n: 每日选取的行业数（研报默认 6）。
    :returns: 同形状 boolean DataFrame；NaN（warm-up）当日不选。
    """
    ranks = diffusion.rank(axis=1, ascending=False, method="first")
    return ranks.le(top_n)  # NaN rank（warm-up）.le → False，天然不选


def select_by_quadrant(
    rs_ratio: pd.DataFrame,
    rs_mom: pd.DataFrame,
    quadrants: Iterable[int] = (1,),
    top_n: int = 6,
) -> pd.DataFrame:
    """
    信号 A（研报 §2.1）：保留 quadrants 内行业；若超 top_n，按距中心距离取最远 top_n。

    :param rs_ratio: RS_Ratio 宽表，index=date, columns=industry_code。
    :param rs_mom: RS_Momentum 宽表，形状与 rs_ratio 一致。
    :param quadrants: 保留的象限集合（如 (1,) 单象限、(1, 2) 相邻双象限）。
    :param top_n: 每日上限（研报默认 6）。
    :returns: 同形状 boolean DataFrame；warm-up NaN 当日不选。
    """
    quad = classify_quadrant(rs_ratio, rs_mom)
    in_quad = quad.isin(list(quadrants))
    dist = distance_from_center(rs_ratio, rs_mom).where(in_quad)
    ranks = dist.rank(axis=1, ascending=False, method="first")
    return in_quad & ranks.le(top_n)


def select_diffusion_with_rrg(
    diffusion: pd.DataFrame,
    rs_ratio: pd.DataFrame,
    rs_mom: pd.DataFrame,
    top_n: int = 6,
    keep_quadrants: Iterable[int] = (1, 2),
) -> pd.DataFrame:
    """
    信号 C（研报 §2.3.1）：扩散 top_n 先选，再剔除非 keep_quadrants 象限的行业，**不补足**。

    :param diffusion: 扩散指标宽表，index=date, columns=industry_code。
    :param rs_ratio: RS_Ratio 宽表，columns 须与 diffusion 一致。
    :param rs_mom: RS_Momentum 宽表，形状与 rs_ratio 一致。
    :param top_n: 扩散先选数（研报默认 6）。
    :param keep_quadrants: 保留象限（研报默认 (1, 2)，即剔除三、四象限）。
    :returns: 同形状 boolean DataFrame，剩余 ≤ top_n 个（研报原样不补足）。
    """
    top = select_by_diffusion(diffusion, top_n=top_n)
    quad = classify_quadrant(rs_ratio, rs_mom)
    keep = quad.isin(list(keep_quadrants))
    return (top & keep).reindex(columns=diffusion.columns).fillna(False).astype(bool)

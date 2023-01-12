'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-12-02 19:17:45
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-12-02 20:09:05
Description: 
'''
from typing import Tuple

import numpy as np


def renormalize(
    a: np.ndarray, from_range: Tuple[float, float], to_range: Tuple[float, float]
) -> np.ndarray:
    """Renormalize `a` from one range to another."""
    from_delta = from_range[1] - from_range[0]
    to_delta = to_range[1] - to_range[0]
    return (to_delta * (a - from_range[0]) / from_delta) + to_range[0]


def min_rel_rescale(a: np.ndarray, to_range: Tuple[float, float]) -> np.ndarray:
    """Rescale elements in `a` relatively to minimum."""
    a_min = np.min(a)
    a_max = np.max(a)
    if a_max - a_min == 0:
        return np.full(a.shape, to_range[0])
    from_range = (a_min, a_max)

    from_range_ratio = np.inf
    if a_min != 0:
        from_range_ratio = a_max / a_min

    to_range_ratio = to_range[1] / to_range[0]
    if from_range_ratio < to_range_ratio:
        to_range = (to_range[0], to_range[0] * from_range_ratio)
    return renormalize(a, from_range, to_range)


def max_rel_rescale(a: np.ndarray, to_range: Tuple[float, float]) -> np.ndarray:
    """Rescale elements in `a` relatively to maximum."""
    a_min = np.min(a)
    a_max = np.max(a)
    if a_max - a_min == 0:
        return np.full(a.shape, to_range[1])
    from_range = (a_min, a_max)

    from_range_ratio = np.inf
    if a_min != 0:
        from_range_ratio = a_max / a_min

    to_range_ratio = to_range[1] / to_range[0]
    if from_range_ratio < to_range_ratio:
        to_range = (to_range[1] / from_range_ratio, to_range[1])
    return renormalize(a, from_range, to_range)
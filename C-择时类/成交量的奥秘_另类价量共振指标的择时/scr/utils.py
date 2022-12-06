'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-11-11 16:53:12
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-12-06 15:47:42
Description: 
'''
from typing import Dict, Tuple

import emoji
import numpy as np
import pandas as pd


def trans2strftime(ser: pd.Series, fmt: str = '%Y-%m-%d') -> pd.Series:

    return pd.to_datetime(ser).dt.strftime(fmt)


def get_value_from_traderanalyzerdict(dic: Dict, *args) -> float:
    """获取嵌套字典中的指定key"""
    if len(args) == 1:
        return dic.get(args[0], 0)
    for k in args:

        if res := dic.get(k, None):
            return get_value_from_traderanalyzerdict(res, *args[1:])

        return 0


def transform_status_table(status: pd.Series) -> pd.DataFrame:

    status: pd.DataFrame = status.to_frame('Status')
    status.index.names = ['Sec_name']
    status['Flag'] = status['Status'].apply(
        lambda x: emoji.emojize(x[2], language='alias'))
    status['Status'] = status['Status'].apply(lambda x: x[0])

    return status.reset_index()


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


# 回测参数
BACKTEST_CONFIG: Dict = dict(n=3, threshold=(1.125, 1.275),
                             bma_window=50, ama_window=100, fast_window=5, slow_window=90)

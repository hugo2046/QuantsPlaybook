"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-27 20:50:18
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-27 20:50:31
FilePath: 
Description: 
"""
from functools import partial
from typing import List, Tuple, Union

import pandas as pd
from FactorZoo import SportBettingsFactor
from tqdm import tqdm

from joblib import Parallel, delayed


def get_factor(
    sportbetting: SportBettingsFactor, factor_name: str, window: int
) -> pd.Series:
    return getattr(sportbetting, factor_name)(window=window, usedf=False)


def get_factors_frame(
    data: pd.DataFrame,
    window: int,
    factor_names: Union[str, List, Tuple] = None,
    general_names: Union[str, List, Tuple] = None,
) -> pd.DataFrame:
    if (factor_names is None) and (general_names is None):
        raise ValueError(
            "factor_names and general_names can't be None at the same time"
        )

    if (factor_names is not None) and isinstance(factor_names, str):
        factor_names: List = [factor_names]

    if general_names is not None:
        if isinstance(general_names, str):
            general_names: List = [general_names]
        factor_names: List = get_factor_name(general_names)

    sportbetting: SportBettingsFactor = SportBettingsFactor(data)

    parallel = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")
    result = parallel(
        delayed(get_factor)(sportbetting, x, window) for x in factor_names
    )
    return pd.concat(result, axis=1, sort=True)


def get_factor_name(general_names: Union[str, List, Tuple] = None) -> List[str]:
    """构建factor名称

    Parameters
    ----------
    general_names : Union[str,List,Tuple], optional
        None表示构建全部因子, by default None
        1. interday: 日间因子
        2. intraday: 日内因子
        3. overnight: 隔夜因子

    Returns
    -------
    List[str]
        因子名称
    """
    if general_names is None:
        general_names: set = {"interday", "intraday", "overnight"}

    if isinstance(general_names, str):
        general_names: set = {general_names}

    factors: List = []

    for factor in general_names:
        factors.extend(
            [
                f"{factor}_volatility_reverse",
                f"{factor}_turnover_reverse",
                f"{factor}_turnover_f_reverse",
                f"revise_{factor}_reverse",
            ]
        )

    factors += ['coin_team','coin_team_f']
    return factors

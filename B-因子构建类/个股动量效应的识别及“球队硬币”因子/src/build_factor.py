"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-27 20:50:18
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-27 20:50:31
FilePath: 
Description: 
"""
from typing import List, Tuple, Union

import pandas as pd
from FactorZoo import SportBettingFactor


def get_factor(
    data: pd.DataFrame, window: int, factor_name: str, method: int = None
) -> pd.Series:
    """根据因子名获取因子"""
    sportbetting: SportBettingFactor = SportBettingFactor(data)
    return getattr(sportbetting, factor_name)(window=window, method=method, usedf=False)


def get_factors_frame(
    data: pd.DataFrame,
    window: int,
    method: int = None,
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

    dfs: List = []
    for factor_name in factor_names:
        if factor_name.find("turnover"):
            if method is None:
                for i in (1, 2):
                    dfs.append(
                        get_factor(data, window, method=i, factor_name=factor_name)
                    )
            else:
                dfs.append(
                    get_factor(data, window, method=method, factor_name=factor_name)
                )

        else:
            dfs.append(get_factor(data, window, method=method, factor_name=factor_name))

    return pd.concat(dfs, axis=1)


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
                f"revise_{factor}_reverse",
            ]
        )

    return factors

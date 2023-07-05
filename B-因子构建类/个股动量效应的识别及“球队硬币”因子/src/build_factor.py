"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-27 20:50:18
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-27 20:50:31
FilePath: 
Description: 
"""

from typing import Dict, List, Tuple, Union

import pandas as pd
from FactorZoo import SportBettingsFactor
from joblib import Parallel, delayed
from qlib.data import D
from loguru import logger


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
        factor_names: List = _generate_factor(general_names)

    sportbetting: SportBettingsFactor = SportBettingsFactor(data)

    parallel = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")

    result = parallel(
        delayed(get_factor)(sportbetting, x, window) for x in factor_names
    )
    return pd.concat(result, axis=1, sort=True)


def _generate_factor(general_names: Union[str, List, Tuple] = None) -> List[str]:
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

    factors += ["coin_team", "coin_team_f"]
    return factors


# 进获取A股所有股票 qlib过滤器速度很慢 不如手动过滤
# 40s
# POOLS: List = D.list_instruments(
#     D.instruments("CN", filter_pipe=[NameDFilter("(3|6|0)\d{5}\.(SH|SZ)")]),
#     as_list=True,
# )


def _fliter_name(code: str) -> bool:
    """过滤北交所"""
    import re

    res = re.search("(3|6|0)\d{5}\.(SH|SZ)", code)
    return res is not None


def get_A_stock() -> List:
    """获取A股票池"""
    POOLS: List = D.list_instruments(
        D.instruments("CN"),
        as_list=True,
    )

    return list(filter(lambda x: _fliter_name(x), POOLS))


def get_base_data(start_dt: str = None, end_dt: str = None) -> pd.DataFrame:
    """获取基础数据

    return:
        data: pd.DataFrame
            index: pd.MultiIndex level0-instrument level1-datetime
            columns: close open turnover_rate turnover_rate_f
    """
    POOLS: List = get_A_stock()
    data: pd.DataFrame = D.features(
        POOLS,
        fields=["$close", "$open", "$turnover_rate", "$turnover_rate_f"],
        start_time=start_dt,
        end_time=end_dt,
    )
    data.columns = data.columns.str.replace("$", "", regex=True)
    return data


def get_oto_data(
    start_dt: str = None, end_dt: str = None, periods: int = 1
) -> pd.DataFrame:
    """获取T+1日开盘收益率数据

    return:
        data: pd.DataFrame
            index: pd.MultiIndex level0-datetime level1-instrument columns-next_ret
    """
    POOLS: List = get_A_stock()

    next_ret_expr: str = f"Ref($open,-{1+periods})/Ref($open,-1)-1"
    cols_dict: Dict = {next_ret_expr: "next_ret"}
    next_ret: pd.DataFrame = D.features(
        POOLS,
        fields=[next_ret_expr],
        start_time=start_dt,
        end_time=end_dt,
    )
    next_ret.rename(columns=cols_dict, inplace=True)
    return next_ret.swaplevel().sort_index()


def get_factor_data_and_forward_return(
    start_dt: str = None, end_dt: str = None, window: int = 20, periods: int = 1
) -> pd.DataFrame:
    """获取因子数据及下期收益数据 下期收益为OTO收益
    总体模拟T日因子 使用T+1日开盘收益率作为下期收益
    T+1日开盘收益率 = T+2日开盘价/T+1日开盘价-1
    Parameters
    ----------
    start_dt : str, optional
        起始日, by default None
    end_dt : str, optional
        结束日, by default None
    window : int, optional
        因子计算期窗口, by default 20
    periods : int, optional
        未来收益率,为1时表示未来一期, by default 1

    Returns
    -------
    pd.DataFrame
        MultiIndex level0:datetime level1:instrument columns:factor next_ret
    """
    logger.info("start get base data...")
    # 获取计算因子的基础数据
    data: pd.DataFrame = get_base_data(start_dt, end_dt)
    logger.info("base data success!")

    logger.info("start get next return data...")
    # 获取下期收益数据
    next_ret: pd.DataFrame = get_oto_data(start_dt, end_dt, periods=periods)

    logger.info("next return data success!")
    logger.info("start get factor data...")
    # 计算因子
    factor_data: pd.DataFrame = get_factors_frame(
        data, window, general_names=["interday", "intraday", "overnight"]
    )
    logger.info("factor data success!")
    logger.info("start merge data...")
    # 8min7.3s
    factor_data: pd.DataFrame = factor_data.sort_index()
    # 合并数据
    all_data: pd.DataFrame = pd.concat((factor_data, next_ret), axis=1)
    all_data: pd.DataFrame = all_data.sort_index()
    logger.info("merge data success!")

    slice_idx: pd.Index = all_data.index.levels[0][window:-(1+periods)]

    return all_data.loc[slice_idx]

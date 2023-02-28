"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-12 16:58:39
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-12 17:36:08
Description: 
"""

from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import gradient_free_optimizers as gfo
import numpy as np
import pandas as pd
from alphalens.utils import MaxLossExceededError, get_clean_factor_and_forward_returns

from .core import Factor_Calculator

CPU_WORKER_NUM: int = int(cpu_count() * 0.7)


def calculate_best_chunk_size(data_length: int, n_workers: int) -> int:

    chunk_size, extra = divmod(data_length, n_workers * 5)
    if extra:
        chunk_size += 1
    return chunk_size


def search_factor_params(
    factor_calc: Factor_Calculator,
    price: pd.DataFrame,
    name: str,
    search_space: Dict,
    iterations: int,
    score_func: callable,
    abs_score: bool = True,
    method: str = "EvolutionStrategyOptimizer",
):
    def opt_strat(window: int, window1: int, window2: int):

        if (window1 <= window2) and (
            name
            in {
                "diff_period_mom",
                "turnover_pct",
                "long_short_pct",
            }
        ):
            return np.nan

        origin_factor: pd.DataFrame = factor_calc.transform(
            name, window=window, window1=window1, window2=window2
        )
        try:
            factor_data: pd.DataFrame = get_clean_factor_and_forward_returns(
                factor=origin_factor.stack(),
                prices=price,
                quantiles=5,
                periods=(1,),
            )
        except MaxLossExceededError as e:
            # 如果max_loss超过设置 这里自动跳过
            err_str: str = str(e)
            print(err_str)
            err_value: np.float32 = _get_err_msg_value(err_str)
            factor_data: pd.DataFrame = get_clean_factor_and_forward_returns(
                factor=origin_factor.stack(),
                prices=price,
                quantiles=5,
                periods=(1,),
                max_loss=err_value,
            )

        if abs_score:
            return np.abs(score_func(factor_data))
        else:
            return score_func(factor_data)

    opt = getattr(gfo, method)(search_space)
    opt.search(lambda x: opt_strat(**x), n_iter=iterations, verbosity=["progress_bar"])
    return opt.best_para


def _get_err_msg_value(err_msg: str) -> np.float32:
    """获取alphalens报错中的max_loss值

    Args:
        err_msg (str): 报错文本

    Returns:
        np.float32: max_loss值
    """
    import re

    if res := re.search(r"exceeded (\d{1,2}\.\d{1,2})", err_msg):
        return np.float32(res.group(1)) / 100
    else:
        print("未识别到错误值")


def _func2search_para(
    name: str,
    factor_calc: Factor_Calculator,
    price: pd.DataFrame,
    search_space: Dict,
    iterations: int,
    score_func: callable,
    abs_score: bool = True,
    method: str = "EvolutionStrategyOptimizer",
) -> Dict:
    print(f"执行{name}因子参数优化")
    para: Dict = search_factor_params(
        factor_calc,
        price,
        name,
        search_space,
        iterations,
        score_func,
        abs_score,
        method,
    )

    return {name: para}


def mult_opt_strat(
    factor_calc: Factor_Calculator,
    price: pd.DataFrame,
    names: List,
    search_space: Dict,
    iterations: int,
    score_func: callable,
    abs_score: bool = True,
    method: str = "EvolutionStrategyOptimizer",
):

    names_num: int = len(names)
    chunk_size: int = calculate_best_chunk_size(names_num, CPU_WORKER_NUM)
    func = partial(
        _func2search_para,
        factor_calc=factor_calc,
        price=price,
        search_space=search_space,
        iterations=iterations,
        score_func=score_func,
        abs_score=abs_score,
        method=method,
    )
    pool = Pool(processes=CPU_WORKER_NUM)
    opt_best_para = pool.imap(func, names, chunksize=chunk_size)
    pool.close()
    pool.join()

    opt_best_para: Tuple = tuple(opt_best_para)

    return {k: v for i in opt_best_para for k, v in i.items()}

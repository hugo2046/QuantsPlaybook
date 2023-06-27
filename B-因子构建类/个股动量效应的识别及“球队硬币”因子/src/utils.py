'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-26 10:37:39
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-27 16:23:53
Description: 
'''
from collections import namedtuple
from typing import  Union

import numpy as np
import pandas as pd


def check_sign(left:Union[float,np.ndarray],right:Union[float,np.ndarray])->Union[float,np.ndarray]:
    """比较两个数的符号是否相同

    Parameters
    ----------
    left : Union[float,np.ndarray]
        数字或数组
    right : Union[float,np.ndarray]
        数字或数组

    Returns
    -------
    Union[float,np.ndarray]
        结果为布尔值或布尔数组
    """
    return (left>=0)==(right>=0)




def mom_effect_stats(
    factor_df: pd.Series, next_ret_df: pd.Series, window: int = 20
) -> namedtuple:
    """个股的动量效应统计

    Parameters
    ----------
    factor_df : pd.Series
        index-date columns-code value-factor
    next_ret_df : pd.Series
        index-date columns-code value-next_ret
    window : int, optional
        滚动窗口期, by default None

    Returns
    -------
    namedtuple
        net - 净动量比例
        avg - 均净动量比例
        stable - 稳净动量比例
        index-date column columns-code value-stable_excess_mom_rate
    """

    res: namedtuple = namedtuple("Res", ["net", "avg", "stable"])

    # 计算因子与截面均值的差值
    excess_factor: pd.DataFrame = factor_df.sub(factor_df.mean(axis=1), axis=0)
    # 计算未来收益与截面均值的差值
    excess_nextret: pd.DataFrame = next_ret_df.sub(next_ret_df.std(axis=1), axis=0)
    # 对齐
    excess_factor, excess_nextret = excess_factor.align(excess_nextret, join="outer")
    # 比较超额因子值与超额收益率的符号
    # true表示出现动量效应 false表示出现反转效应
    sign_df: pd.DataFrame = excess_factor.apply(
        lambda ser: check_sign(ser, excess_nextret[ser.name])
    )
    # 净动量比例 = 出现动量效应的股票数量减去出现反转效应的股票数量,再除以横截面的总股票数量
    excess_mom_rate: pd.DataFrame = (
        sign_df.sum(axis=1) - (~sign_df).sum(axis=1)
    ) / factor_df.count(axis=1)
    # 均净动量比例 = "净动量比例”取时序均值
    avg_excess_mom_rate: pd.DataFrame = excess_mom_rate.rolling(window=window).mean()
    # 标准差净动量比例 = "净动量比例”取时序标准差
    std_excess_mom_rate: pd.DataFrame = excess_mom_rate.rolling(window=window).std()
    # 稳净动量比例 = 均净动量比例/标准差净动量比例
    stable_excess_mom_rate: pd.DataFrame = avg_excess_mom_rate.div(std_excess_mom_rate)

    return res(excess_mom_rate, avg_excess_mom_rate, stable_excess_mom_rate)
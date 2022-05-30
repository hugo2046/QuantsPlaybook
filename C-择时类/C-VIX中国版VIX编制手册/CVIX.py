'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-27 14:16:36
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-05-30 16:03:51
FilePath: 
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# import datetime as dt
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def get_cboe_vix(opt_data: pd.DataFrame, rate_df: pd.DataFrame) -> pd.Series:
    """计算CIV和SKEW

    Args:
        opt_data (pd.DataFrame): _description_
        rate_df (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    date_index = []  # 储存index
    vix_value = []  # 储存vix
    skew_value = []  # 储存skew

    for trade_date, slice_df in opt_data.groupby('date'):

        # maturity距离到期日的时间字典
        # df_jy近一月
        # df_cjy次近月
        maturity, df_jy, df_cjy = filter_contract(slice_df)

        # 获取无风险收益
        rf_rate_jy = rate_df.loc[trade_date, int(maturity['jy'] * 365)]
        rf_rate_cjy = rate_df.loc[trade_date, int(maturity['cjy'] * 365)]

        # 计算远期价格
        fp_jy = cal_forward_price(maturity['jy'], rf_rate=rf_rate_jy, df=df_jy)
        fp_cjy = cal_forward_price(maturity['cjy'],
                                   rf_rate=rf_rate_cjy,
                                   df=df_cjy)

        # 计算中间价格
        df_mp_jy = cal_mid_price(maturity['jy'], df_jy, fp_jy)
        df_mp_cjy = cal_mid_price(maturity['cjy'], df_cjy, fp_cjy)

        # 计算行权价差
        df_diff_k_jy = cal_k_diff(df_jy)
        df_diff_k_cjy = cal_k_diff(df_cjy)

        # 计算VIX
        df_tovix_jy = pd.concat([df_mp_jy, df_diff_k_jy], axis=1).reset_index()
        df_tovix_cjy = pd.concat([df_mp_cjy, df_diff_k_cjy],
                                 axis=1).reset_index()

        nearest_k_jy = _nearest_k(df_jy, fp_jy)
        nearest_k_cjy = _nearest_k(df_cjy, fp_cjy)

        vix = cal_vix(df_tovix_jy, fp_jy, rf_rate_jy, maturity['jy'],
                      nearest_k_jy, df_tovix_cjy, fp_cjy, rf_rate_cjy,
                      maturity['cjy'], nearest_k_cjy)

        skew = cal_skew(df_tovix_jy, fp_jy, rf_rate_jy, maturity['jy'],
                        nearest_k_jy, df_tovix_cjy, fp_cjy, rf_rate_cjy,
                        maturity['cjy'], nearest_k_cjy)

        date_index.append(trade_date)
        vix_value.append(vix)
        skew_value.append(skew)

    data = pd.DataFrame({
        "CH_VIX": vix_value,
        "CH_SKEW": skew_value
    },
        index=date_index)

    data.fillna(method='pad', inplace=True)

    data.index = pd.DatetimeIndex(data.index)

    return data


# 计算近、次近月VIX
def cal_vix(df_jy: pd.DataFrame, forward_price_jy: float, rf_rate_jy: float,
            maturity_jy: float, nearest_k_jy: float, df_cjy: pd.DataFrame,
            forward_price_cjy: float, rf_rate_cjy: float, maturity_cjy: float,
            nearest_k_cjy: float):

    sigma_jy = cal_vix_sub(df_jy, forward_price_jy, rf_rate_jy, maturity_jy,
                           nearest_k_jy)

    sigma_cjy = cal_vix_sub(df_cjy, forward_price_cjy, rf_rate_cjy,
                            maturity_cjy, nearest_k_cjy)

    w = (maturity_cjy - 30.0 / 365) / (maturity_cjy - maturity_jy)

    to_sqrt = maturity_jy * sigma_jy * w + maturity_cjy * sigma_cjy * (1 - w)

    if to_sqrt >= 0:

        vix = 100 * np.sqrt(to_sqrt * 365.0 / 30)

    else:

        vix = np.nan

    return vix


# 计算SKEW
def cal_skew(df_jy: pd.DataFrame, forward_price_jy: float,
             rf_rate_jy: float, maturity_jy: float, nearest_k_jy: float,
             df_cjy: pd.DataFrame, forward_price_cjy: float,
             rf_rate_cjy: float, maturity_cjy: float,
             nearest_k_cjy: float) -> float:

    s_jy = cal_moments_sub(df_jy, maturity_jy, rf_rate_jy,
                           forward_price_jy, nearest_k_jy)

    s_cjy = cal_moments_sub(df_cjy, maturity_cjy, rf_rate_cjy,
                            forward_price_cjy, nearest_k_cjy)

    w = (maturity_cjy - 30.0 / 365) / (maturity_cjy - maturity_jy)

    skew = 100 - 10 * (w * s_jy + (1 - w) * s_cjy)

    return skew


def cal_vix_sub(df: pd.DataFrame, forward_price: float, rf_rate: float,
                maturity: float, nearest_k: float):
    def _vix_sub_fun(x):
        ret = x['diff_k'] * np.exp(
            rf_rate * maturity) * x['mid_p'] / np.square(x['exercise_price'])
        return ret

    temp_var = df.apply(lambda x: _vix_sub_fun(x), axis=1)

    sigma = 2 * temp_var.sum() / maturity - np.square(forward_price /
                                                      nearest_k - 1) / maturity

    return sigma


def cal_moments_sub(df: pd.DataFrame, maturity: float, rf_rate: float,
                    forward_price: float, nearest_k: float) -> float:

    e1, e2, e3 = cal_epsilon(forward_price, nearest_k)

    temp_p1 = -np.sum(
        df['mid_p'] * df['diff_k'] / np.square(df['exercise_price']))

    p1 = np.exp(maturity * rf_rate) * (temp_p1) + e1

    temp_p2 = np.sum(df['mid_p'] * df['diff_k'] * 2 *
                     (1 - np.log(df['exercise_price'] / forward_price)) /
                     np.square(df['exercise_price']))
    p2 = np.exp(maturity * rf_rate) * (temp_p2) + e2

    temp_p3 = temp_p3 = np.sum(
        df['mid_p'] * df['diff_k'] * 3 *
        (2 * np.log(df['exercise_price'] / forward_price) -
         np.square(np.log(df['exercise_price'] / forward_price))) /
        np.square(df['exercise_price']))

    p3 = np.exp(maturity * rf_rate) * (temp_p3) + e3

    s = (p3 - 3 * p1 * p2 + 2 * p1**3) / (p2 - p1**2)**(3 / 2)

    return s


def cal_epsilon(forward_price: float, nearest_k: float) -> tuple:

    e1 = -(1 + np.log(forward_price / nearest_k) - forward_price / nearest_k)

    e2 = 2 * np.log(nearest_k / forward_price) * (
        nearest_k / forward_price - 1) + np.square(
            np.log(nearest_k / forward_price)) * 0.5

    e3 = 3 * np.square(np.log(nearest_k / forward_price)) * (
        np.log(nearest_k / forward_price) / 3 - 1 + forward_price / nearest_k)

    return e1, e2, e3


def cal_forward_price(maturity: dict, rf_rate: float,
                      df: pd.DataFrame) -> float:

    # 获取认购与认沽的绝对值差异最小值的信息
    min_con = df.sort_values('diff').iloc[0]

    # 获取的最小exercise_price
    k_min = min_con.name

    # F = Strike Price + e^RT x (Call Price - Put Price)
    f_price = k_min + np.exp(
        maturity * rf_rate) * (min_con['call'] - min_con['put'])

    return f_price


# 计算中间价格
def cal_mid_price(maturity: dict, df: pd.DataFrame,
                  forward_price: float) -> pd.DataFrame:
    def _cal_mid_fun(x, val: float):
        res = None
        if x['exercise_price'] < val:
            res = x['put']
        elif x['exercise_price'] > val:
            res = x['call']
        else:
            res = (x['put'] + x['call']) / 2
        return res

    # 小于远期价格且最靠近的合约的行权价
    m_k = _nearest_k(df, forward_price)

    ret = pd.DataFrame(index=df.index)

    # 计算中间件
    m_p_lst = df.reset_index().apply(lambda x: _cal_mid_fun(x, val=m_k),
                                     axis=1)

    ret['mid_p'] = m_p_lst.values

    return ret


def cal_k_diff(df: pd.DataFrame) -> pd.DataFrame:

    arr_k = df.index.values
    ret = pd.DataFrame(index=df.index)

    res = []
    res.append(arr_k[1] - arr_k[0])
    res.extend(0.5 * (arr_k[2:] - arr_k[0:-2]))
    res.append(arr_k[-1] - arr_k[-2])
    ret['diff_k'] = res
    return ret


def _nearest_k(df: pd.DataFrame, forward_price: float) -> float:

    # 行权价等于或小于远期价格的合约
    temp_df = df[df.index <= forward_price]
    if temp_df.empty:

        temp_df = df

    m_k = temp_df.sort_values('diff').index[0]

    return m_k


# 选出当日的近远月合约(且到期日大于1周)
def filter_contract(cur_df: pd.DataFrame) -> Tuple[Dict, pd.Series, pd.Series]:
    """选出当日的近远月合约(且到期日大于1周)

    Args:
        cur_df (pd.DataFrame): 用于筛选的数据
        | index | date      | close  | contract_type | exercise_price | maturity |
        | :---- | :-------- | :----- | :------------ | :------------- | :------- |
        | 0     | 2021/7/29 | 0.5275 | call          | 4.332          | 0.649315 |

    Returns:
        Tuple[Dict,pd.Series,pd.Series]: 
            1. 合约字典 {'jy':xx.xx,'cjy':xx.xx}        
            2. 近月 pd.DataFrame 
            3. 次近月
    """
    def _check_fields(x_df: pd.DataFrame) -> pd.DataFrame:

        # 目标字段
        target_fields: List = ['call', 'put']

        for col in target_fields:

            if col not in x_df.columns:
                print("%s字段为空" % col)
                x_df[col] = 0

        return x_df

    # 今天在交易的合约的到期日
    ex_t: np.ndarray = cur_df['maturity'].unique()
    # 选择到期日大于等于5天的数据
    ex_t: np.ndarray = ex_t[ex_t >= 5. / 365]

    # 到期日排序，最小两个为近月、次近月
    try:
        jy_dt, cjy_dt = np.sort(ex_t)[:2]

    except ValueError:

        print(ex_t, np.sort(ex_t)[:2])

    maturity_dict: Dict = dict(zip(['jy', 'cjy'], [jy_dt, cjy_dt]))

    # 选取近月及次近月合约
    # 到期时间为近月及次近月的合约
    cur_df: pd.DataFrame = cur_df[cur_df['maturity'].isin([jy_dt, cjy_dt])]

    keep_cols: List = ['close', 'contract_type', 'exercise_price']

    cur_df_jy: pd.DataFrame = cur_df.query('maturity==@jy_dt')[keep_cols]
    cur_df_cjy: pd.DataFrame = cur_df.query('maturity==@cjy_dt')[keep_cols]

    cur_df_jy: pd.DataFrame = cur_df_jy.pivot_table(index='exercise_price',
                                                    columns='contract_type',
                                                    values='close')

    cur_df_cjy: pd.DataFrame = cur_df_cjy.pivot_table(index='exercise_price',
                                                      columns='contract_type',
                                                      values='close')

    # 检查字段
    cur_df_jy: pd.DataFrame = _check_fields(cur_df_jy)
    cur_df_cjy: pd.DataFrame = _check_fields(cur_df_cjy)

    # 绝对值差异
    cur_df_jy['diff'] = np.abs(cur_df_jy['call'] - cur_df_jy['put'])
    cur_df_cjy['diff'] = np.abs(cur_df_cjy['call'] - cur_df_cjy['put'])

    return maturity_dict, cur_df_jy, cur_df_cjy

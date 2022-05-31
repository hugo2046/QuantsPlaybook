'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-27 17:54:06
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-05-31 16:03:08
FilePath: 
Description: 
'''
from collections import namedtuple
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# 设置全年天数
YEARS = 365


def _get_near_or_next_options(df: pd.Series, n: int = 1) -> pd.DataFrame:
    """获取opt_data中近月及次近月

    Args:
        df (pd.Series)
        | idnex | list_date | exercise_date | exercise_price | contract_type | code          |
        | :---- | :-------- | :------------ | :------------- | :------------ | :------------ |
        | 0     | 2021/7/29 | 2022/3/23     | 4.332          | CO            | 10003549.XSHG |

    Returns:
        pd.DataFrame
    """
    # 需要到期日至现在大于等于一周
    df = df[df['maturity'] >= 5 / YEARS]
    cond = (df['maturity'] <= np.sort(df['maturity'].unique())[n])

    return df[cond]


def _build_strike_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """构建期权价差表

    Parameters
    ----------
    df : pd.DataFrame
        | index | date      | exercise_date | close  | contract_type | exercise_price | maturity |
        | :---- | :-------- | :------------ | :----- | :------------ | :------------- | :------- |
        | 0     | 2021/7/29 | 2022/3/23     | 0.5275 | call          | 4.332          | 0.649315 |

    Returns
    -------
    pd.DataFrame
        | contract_type  | call   | put    | diff    |
        | :------------- | :----- | :----- | :------ |
        | exercise_price |        |        |         |
        | 2.2            | 0.1826 | 0.0617 | 0.1209  |
        | 2.25           | 0.146  | 0.0777 | 0.0683  |
        | 2.3            | 0.1225 | 0.0969 | 0.0256  |
        | 2.35           | 0.0942 | 0.1268 | 0.0326 |
        | 2.4            | 0.0735 | 0.1542 | 0.0807 |
    """
    matrix: pd.DataFrame = pd.pivot_table(df,
                                          index='exercise_price',
                                          columns='contract_type',
                                          values='close')
    matrix['diff'] = (matrix['call'] - matrix['put'])
    return matrix


def _get_min_strike_diff(strike_matrix: pd.DataFrame) -> pd.Series:
    """获取strike_matrix认购期权和认沽期权价差绝对值最小数据信息

    Parameters
    ----------
    df : pd.DataFrame
        | contract_type  | call   | put    | diff    |
        | :------------- | :----- | :----- | :------ |
        | exercise_price |        |        |         |
        | 2.2            | 0.1826 | 0.0617 | 0.1209  |
        | 2.25           | 0.146  | 0.0777 | 0.0683  |
        | 2.3            | 0.1225 | 0.0969 | 0.0256  |
        | 2.35           | 0.0942 | 0.1268 | 0.0326 |
        | 2.4            | 0.0735 | 0.1542 | 0.0807 |

    Returns
    -------
    pd.Series
        index - exercise_price|call|put|diff| values
    """
    df_ = strike_matrix.reset_index()
    min_idx = df_['diff'].abs().idxmin()

    return df_.loc[min_idx]


def calc_delta_k_table(strike_matrix: pd.DataFrame) -> pd.Series:
    """期权合约行权价价值表

    Parameters
    ----------
    strike_matrix : 
        pd.DataFrame
        | contract_type  | call   | put    | diff    |
        | :------------- | :----- | :----- | :------ |
        | exercise_price |        |        |         |
        | 2.2            | 0.1826 | 0.0617 | 0.1209  |
        | 2.25           | 0.146  | 0.0777 | 0.0683  |
        | 2.3            | 0.1225 | 0.0969 | 0.0256  |
        | 2.35           | 0.0942 | 0.1268 | 0.0326  |
        | 2.4            | 0.0735 | 0.1542 | 0.0807  |

    Returns
    -------
    pd.Series
        index-exercies_price values-中间价
    """
    exercise_price: np.ndarray = strike_matrix.index._values
    # 获取长度
    size = len(exercise_price)
    diff_ser = pd.Series(index=strike_matrix.index, data=np.empty(size))
    # 构建diff
    diff_ser.iloc[0] = exercise_price[1] - exercise_price[0]
    diff_ser.iloc[1:-1] = 0.5 * (exercise_price[2:] - exercise_price[0:-2])
    diff_ser.iloc[-1] = exercise_price[-1] - exercise_price[-2]
    # diff: List = []
    # diff.append(exercise_price[1] - exercise_price[0])
    # diff.extend(0.5 * (exercise_price[2:] - exercise_price[0:-2]))
    # diff.append(exercise_price[-1] - exercise_price[-2])
    # pd.Series(index=exercise_price, data=diff)
    return diff_ser


def _get_median_price_table(strike_matrix: pd.DataFrame,
                            F: float) -> Tuple[pd.Series, float]:
    """根据执行价矩阵获取中间报价表

    Parameters
    ----------
    strike_matrix : pd.DataFrame
        | contract_type  | call   | put    | diff    |
        | :------------- | :----- | :----- | :------ |
        | exercise_price |        |        |         |
        | 2.2            | 0.1826 | 0.0617 | 0.1209  |
        | 2.25           | 0.146  | 0.0777 | 0.0683  |
        | 2.3            | 0.1225 | 0.0969 | 0.0256  |
        | 2.35           | 0.0942 | 0.1268 | 0.0326  |
        | 2.4            | 0.0735 | 0.1542 | 0.0807  |
    F : float
        F

    Returns
    -------
    pd.Seroes
        index-exercies_price values-中间价
    """
    # 获取K
    exercise_price: np.ndarray = strike_matrix.index._values
    # 选取比F小，但差值又最小的合约
    K_cond1: np.ndarray = (exercise_price < F)

    try:
        # K值
        K_0: float = strike_matrix.loc[K_cond1, 'diff'].idxmin()
    except ValueError:
        print('F:%.4f' % F)
        print(K_cond1)
        print(strike_matrix)
        raise ValueError('无对应的K_0数据!')

    # 根据K构建中间价
    call_ser: pd.Series = strike_matrix.loc[exercise_price > K_0, 'call']
    put_ser: pd.Series = strike_matrix.loc[exercise_price < K_0, 'put']
    median: float = (strike_matrix.loc[K_0, 'call'] +
                     strike_matrix.loc[K_0, 'put']) * 0.5

    # 合并call,put
    all_ser: pd.Series = pd.concat((put_ser, call_ser))
    # 添加中间价
    all_ser[K_0] = median

    return all_ser.sort_index(), K_0


def _get_free_rate(shibor_ser: pd.Series, near_term: float,
                   next_tern: float) -> Tuple:
    """根据near_term,next_tern获取对应的shibor值

    Parameters
    ----------
    shibor_ser : pd.Series
        shibor数据
    near_term : float
        近月
    next_tern : float
        次近月

    Returns
    -------
    Tuple
        近月无风险收益,次近月无风险收益
    """
    near_term_ = max(round(near_term * YEARS), 1)  # 最小为1
    next_tern_ = round(next_tern * YEARS)

    try:
        shibor_ser.loc[near_term_]
    except KeyError:
        print(near_term, next_tern)
        print(near_term_, next_tern_)
        print(shibor_ser)
        raise KeyError('无对应的shibor!')

    return shibor_ser.loc[near_term_], shibor_ser.loc[next_tern_]


def calc_F(K: float, R: float, T: float, C: float, P: float) -> float:
    """计算远期价格水平

    Parameters
    ----------
    K : float
        K为认购期权和认沽期权间价差最小的期权合约对应的执行价
    R : float
        无风险收益
    T : float
        期限
    C : float
        C 为对应的认购期权价格
    P : float
        P 为认沽期权价格

    Returns
    -------
    float
        远期价格水平
    """
    return K + np.exp(R * T) * (C - P)


def calc_sigma(K_0: float, K: np.ndarray, delta_K: np.ndarray, Q_K: np.ndarray,
               F: float, R: float, T: float) -> float:
    """计算sigma

    Args:
        K_0 (float): K_0
        K (np.ndarray): 执行价
        delta_K (np.ndarray): $\delta_K$
        Q_K (np.ndarray): 中间报价
        F (float): 远期价
        R (float): 无风险收益
        T (float): 期限

    Returns:
        float: sigma
    """
    return (2 / T) * np.sum(delta_K / np.power(K, 2) * np.exp(R * T) *
                            Q_K) - (1 / T) * np.power(F / K_0 - 1, 2)


def calc_vix(near_sigma: float, next_sigma: float, near_term: float,
             next_term: float) -> float:
    """计算VIX

    Args:
        near_sigma (float): 近月sigma
        next_sigma (float): 次近月sigma
        near_term (float): 近月期限
        next_term (float): 次近月期限

    Returns:
        float: VIX
    """
    weight = calc_weight(near_term, next_term)

    return np.sqrt(
        (near_term * np.power(near_sigma, 2) * weight +
         next_term * np.power(next_sigma, 2) * (1 - weight)) * (YEARS / 30))


def calc_weight(t1: float, t2: float) -> float:
    """计算权重

    Args:
        t1 (float): near_term
        t2 (float): next_term

    Returns:
        float: 权重
    """
    t30 = 30 / YEARS

    return (t2 - t30) / (t2 - t1)


def get_daily_vix(df: pd.DataFrame) -> namedtuple:
    """计算vix

    Args:
        df (pd.DataFrame): 
        | index | date      | exercise_date | close  | contract_type | exercise_price | maturity | near_maturity | next_maturity | near_rate | next_rate |
        | :---- | :-------- | :------------ | :----- | :------------ | :------------- | :------- | :------------ | :------------ | :-------- | :-------- |
        | 1     | 2015/3/11 | 2015/3/25     | 0.0552 | call          | 2.35           | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |
        | 2     | 2015/3/11 | 2015/3/25     | 0.1348 | put           | 2.5            | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |
        | 3     | 2015/3/11 | 2015/3/25     | 0.0063 | call          | 2.5            | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |

    Returns:
        namedtuple: strike_matrix,F,K,median_table,delta_k,sigma,term,vix
                    strike_matrix (Dict):k-near|next 近月,次近月
                                        v-pd.DataFrame
                                        | contract_type  | call   | put    | diff    |
                                        | :------------- | :----- | :----- | :------ |
                                        | exercise_price |        |        |         |
                                        | 2.2            | 0.1826 | 0.0617 | 0.1209  |
                                        | 2.25           | 0.146  | 0.0777 | 0.0683  |
                                        | 2.3            | 0.1225 | 0.0969 | 0.0256  |
                                        | 2.35           | 0.0942 | 0.1268 | 0.0326  |
                                        | 2.4            | 0.0735 | 0.1542 | 0.0807  |
                                        
                    F (Dict):k-near|next 近月,次近月 v-float
                    k (Dict):k-near|next 近月,次近月 v-float
                    median_table (Dict):k-near|next 近月,次近月
                                 v-pd.DataFrame
                                | contract_type  | call   | put    | diff    |
                                | :------------- | :----- | :----- | :------ |
                                | exercise_price |        |        |         |
                                | 2.2            | 0.1826 | 0.0617 | 0.1209  |
                                | 2.25           | 0.146  | 0.0777 | 0.0683  |
                                | 2.3            | 0.1225 | 0.0969 | 0.0256  |
                                | 2.35           | 0.0942 | 0.1268 | 0.0326  |
                                | 2.4            | 0.0735 | 0.1542 | 0.0807  |
                    delta_k (Dict):k-near|next 近月,次近月 
                                   v-pd.Series index-执行价 values
                    sigma (Dict):k-near|next 近月,次近月 v-float
                    term  (Dict):k-near|next 近月,次近月 v-float
                    vix-float
                   
    """
    # 储存中间变量
    variable = namedtuple(
        'Variable', 'strike_matrix,F,K,median_table,delta_k,sigma,term,vix')
    # 获取对应的期权信息
    ## 近月
    near_df: pd.DataFrame = df[df['maturity'] == df['near_maturity']]
    near_strike_matrix: pd.DataFrame = _build_strike_matrix(near_df)

    near_strike_ser: pd.Series = _get_min_strike_diff(near_strike_matrix)

    ## 近月无风险收益等
    near_term: float = near_df['near_maturity'].iloc[0]
    near_term_rate: float = near_df['near_rate'].iloc[0]

    ## 次近月
    next_df: pd.DataFrame = df[df['maturity'] == df['next_maturity']]
    next_strike_matrix: pd.DataFrame = _build_strike_matrix(
        df[df['maturity'] == df['next_maturity']])

    next_strike_ser: pd.Series = _get_min_strike_diff(next_strike_matrix)
    ## 次近月无风险收益等
    next_term: float = next_df['next_maturity'].iloc[0]
    next_term_rate: float = next_df['next_rate'].iloc[0]

    # 计算远期价格
    near_F: float = calc_F(near_strike_ser['exercise_price'], near_term_rate,
                           near_term, near_strike_ser['call'],
                           near_strike_ser['put'])

    next_F: float = calc_F(next_strike_ser['exercise_price'], next_term_rate,
                           next_term, next_strike_ser['call'],
                           next_strike_ser['put'])

    # 计算中间价格表
    near_median_table, near_K_0 = _get_median_price_table(
        near_strike_matrix, near_F)
    next_median_table, next_K_0 = _get_median_price_table(
        next_strike_matrix, next_F)

    # 计算delta_k
    near_delta_k: pd.Series = calc_delta_k_table(near_median_table)
    next_delta_k: pd.Series = calc_delta_k_table(next_median_table)

    # 计算simma
    near_sigma: float = calc_sigma(near_K_0, near_median_table.index._values,
                                   near_delta_k.values,
                                   near_median_table.values, near_F,
                                   near_term_rate, near_term)

    next_sigma: float = calc_sigma(next_K_0, next_median_table.index._values,
                                   next_delta_k.values,
                                   next_median_table.values, next_F,
                                   next_term_rate, next_term)
    # 计算vix
    vix = calc_vix(near_sigma, next_sigma, near_term, next_term)

    return variable({
        'near': near_strike_matrix,
        'next': near_strike_matrix
    }, {
        'near': near_F,
        'next': next_F
    }, {
        'near': near_K_0,
        'next': next_K_0
    }, {
        'near': near_median_table,
        'next': next_median_table
    }, {
        'near': near_delta_k,
        'next': next_delta_k
    }, {
        'near': near_sigma,
        'next': next_sigma
    }, {
        'near': near_term,
        'next': next_term
    }, vix)

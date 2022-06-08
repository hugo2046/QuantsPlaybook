'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-27 17:54:06
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-07 14:28:09
Description: VIX,STEW计算相关的核心计算
'''
import warnings
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

# 设置全年天数
YEARS = 365


################################################################################################
#                             计算VIX,STEW的组件
################################################################################################
def _get_near_or_next_options(df: pd.Series,
                              n: int = 1,
                              filter_num: int = 7) -> pd.DataFrame:
    """获取opt_data中近月及次近月

    Args:
        df (pd.Series)
        | idnex | list_date | exercise_date | exercise_price | contract_type | code          |
        | :---- | :-------- | :------------ | :------------- | :------------ | :------------ |
        | 0     | 2021/7/29 | 2022/3/23     | 4.332          | CO            | 10003549.XSHG |
        n (int): 下标起始为0 为1时表示获取maturity最小的两个值
        filter_num (int):表示过滤小于filter_num的合约
    Returns:
        pd.DataFrame
    """
    # # 需要到期日至现在大于等于一周
    df = df[df['maturity'] >= (filter_num / YEARS)]
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

    empty_put = False  # 判断是否empty
    try:
        # K值
        K_0: float = strike_matrix.loc[K_cond1, 'diff'].idxmin()
    except ValueError:
        # TODO:执行价全部小于远期价格时的处理是否正确？？？
        K_0: float = F
        empty_put: bool = True
        warnings.warn('F:%.4f,strike_marix中最小执行价为:%.4f,故开跌部分无数据,无中间价K0.' %
                      (F, strike_matrix.index[0]))

    # 根据K构建中间价
    call_ser: pd.Series = strike_matrix.loc[exercise_price > K_0, 'call']

    if not empty_put:
        put_ser: pd.Series = strike_matrix.loc[exercise_price < K_0, 'put']

        median: float = (strike_matrix.loc[K_0, 'call'] +
                         strike_matrix.loc[K_0, 'put']) * 0.5
        # 添加中间价
        put_ser[K_0] = median

        # 合并call,put
        all_ser: pd.Series = pd.concat((put_ser, call_ser))

    else:
        all_ser: pd.Series = call_ser

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

    return np.sqrt((near_term * near_sigma * weight + next_term * next_sigma *
                    (1 - weight)) * (YEARS / 30))


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


def _get_sigma(df: pd.DataFrame, method: str = None) -> Dict:
    """

    Parameters
    ----------
    df : pd.DataFrame
        | index | date      | exercise_date | close  | contract_type | exercise_price | maturity | near_maturity | next_maturity | near_rate | next_rate |
        | :---- | :-------- | :------------ | :----- | :------------ | :------------- | :------- | :------------ | :------------ | :-------- | :-------- |
        | 1     | 2015/3/11 | 2015/3/25     | 0.0552 | call          | 2.35           | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |
        | 2     | 2015/3/11 | 2015/3/25     | 0.1348 | put           | 2.5            | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |
        | 3     | 2015/3/11 | 2015/3/25     | 0.0063 | call          | 2.5            | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |

    method:near 或者 next
    Returns
    -------
    Dict

    """
    if method is None:
        raise ValueError('method参数不能为空')

    method = method.lower()
    if method not in ['near', 'next']:

        raise ValueError('method参数必须为near或者next')

    # TODO:这里直接丢弃了缺失值
    strike_matrix: pd.DataFrame = _build_strike_matrix(df).dropna()
    # 计算F值所需信息
    strike_row: pd.Series = _get_min_strike_diff(strike_matrix)

    # 近月无风险收益等
    term: float = df[f'{method}_maturity'].iloc[0]
    term_rate: float = df[f'{method}_rate'].iloc[0]

    # 计算远期价格
    F: float = calc_F(strike_row['exercise_price'], term_rate, term,
                      strike_row['call'], strike_row['put'])

    # 计算中间价格表
    median_table, K_0 = _get_median_price_table(strike_matrix, F)

    # 计算delta_k
    delta_k: pd.Series = calc_delta_k_table(median_table)

    # 计算simma
    sigma: float = calc_sigma(K_0, median_table.index._values, delta_k.values,
                              median_table.values, F, term_rate, term)

    return {
        'strike_matrix': strike_matrix,
        'median_table': median_table,
        'delta_k': delta_k,
        'F': F,
        'K0': K_0,
        'term': term,
        'term_rate': term_rate,
        'sigma': sigma
    }


def calc_epsilons(F0: float, K0: float) -> Tuple:
    """根据F0和K0计算epsilon

    Args:
        F0 (float): 
        K0 (float): 

    Returns:
        Tuple: 
    """
    epsilon1 = -1 - np.log(F0 / K0) + (F0 / K0)
    epsilon2 = 2 * np.log(K0 / F0) * (F0 / K0 - 1) + 0.5 * np.square(
        np.log(K0 / F0))
    epsilon3 = 3 * np.square(np.log(
        K0 / F0)) * (np.log(K0 / F0) / 3 - 1 + F0 / K0)

    return epsilon1, epsilon2, epsilon3


def calc_p_values(K: np.ndarray, Q_K: np.ndarray, delta_K: np.ndarray,
                  epsilons: Tuple, rate: float, term: float,
                  F: float) -> Tuple:

    e1, e2, e3 = epsilons
    e_rt: float = np.exp(rate * term)
    p1: float = e_rt * (-1 * np.sum((Q_K * delta_K) / np.square(K))) + e1

    p2: float = e_rt * np.sum(
        (2 * Q_K * delta_K) / np.square(K) * (1 - np.log(K / F))) + e2

    p3: float = e_rt * np.sum(
        (3 * Q_K * delta_K) / np.square(K) *
        (2 * np.log(K / F) - np.square(np.log(K / F)))) + e3

    return p1, p2, p3


def calc_s(P: Union[np.ndarray, Tuple, List]) -> float:
    p1, p2, p3 = P
    return (p3 - 3 * p1 * p2 + 2 * np.power(p1, 3)) / np.power(
        p2 - np.square(p1), 3 / 2)


def _get_s(df: pd.DataFrame) -> pd.Series:

    df.set_index('trade_date', inplace=True)

    df['espilons'] = df.apply(lambda x: calc_epsilons(x['F'], x['K0']), axis=1)

    df['P'] = df.apply(lambda x: calc_p_values(x['K'], x['Q_K'], x[
        'delta_k'], x['espilons'], x['term_rate'], x['term'], x['F']),
        axis=1)

    return df['P'].apply(calc_s)


def calc_skew(w: float, s_near: float, s_next: float) -> float:

    return 100 - 10 * (w * s_near + (1 - w) * s_next)


################################################################################################
#                             结果
################################################################################################


class CVIX():
    def __init__(self, data: pd.DataFrame) -> None:
        """VIX

        Args:
            data (pd.DataFrame): 
                | index | date      | exercise_date | close  | contract_type | exercise_price | maturity | near_maturity | next_maturity | near_rate | next_rate |
                | :---- | :-------- | :------------ | :----- | :------------ | :------------- | :------- | :------------ | :------------ | :-------- | :-------- |
                | 1     | 2015/3/11 | 2015/3/25     | 0.0552 | call          | 2.35           | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |
                | 2     | 2015/3/11 | 2015/3/25     | 0.1348 | put           | 2.5            | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |
                | 3     | 2015/3/11 | 2015/3/25     | 0.0063 | call          | 2.5            | 0.038356 | 0.038356      | 0.115068      | 0.04814   | 0.052589  |
            """
        self.data: pd.DataFrame = data
        self.variable_dict: defaultdict = defaultdict(list)

    def vix(self) -> pd.Series:

        return self.data.groupby('date').apply(lambda x: self._calc_vix(x))

    def _calc_vix(self, df: pd.DataFrame) -> pd.Series:

        trade_date = df.name
        # 获取对应的期权信息
        # 近月
        near_df: pd.DataFrame = df.query('maturity == near_maturity')
        near_sigma_variable: Dict = _get_sigma(near_df, 'near')
        near_sigma_variable['trade_date'] = trade_date

        self._get_variable_dict(near_sigma_variable, 'near')

        # 次近月
        next_df: pd.DataFrame = df.query('maturity == next_maturity')
        next_sigma_variable: Dict = _get_sigma(next_df, 'next')
        next_sigma_variable['trade_date'] = trade_date

        self._get_variable_dict(next_sigma_variable, 'next')
        # 计算vix

        vix = calc_vix(near_sigma_variable['sigma'],
                       next_sigma_variable['sigma'],
                       near_sigma_variable['term'],
                       next_sigma_variable['term'])

        return vix

    def skew(self) -> pd.Series:

        if ('next' in self.variable_dict) and ('near' in self.variable_dict):
            next_variable = pd.DataFrame(self.variable_dict['next'])
            near_variable = pd.DataFrame(self.variable_dict['near'])

        else:

            self.vix()

        next_s = _get_s(next_variable)
        near_s = _get_s(near_variable)

        df = pd.concat(
            (near_variable['term'], next_variable['term'], near_s, next_s),
            axis=1)

        df.columns = ['t1', 't2', 'p1', 'p2']

        df['w'] = df.apply(lambda x: calc_weight(x['t1'], x['t2']), axis=1)

        return df.apply(lambda x: calc_skew(x['w'], x['p1'], x['p2']), axis=1)

    def _get_variable_dict(self, sigma_variable: Dict, name: str) -> None:

        tmp: Dict = {}

        for k, v in sigma_variable.items():

            if k == 'median_table':

                tmp['Q_K'] = v.values
                tmp['K'] = np.array(v.index)

            elif k == 'delta_k':

                tmp['delta_k'] = v.values

            elif k not in ['strike_matrix']:

                tmp[k] = v

        self.variable_dict[name].append(tmp)


def prepare_data2calc(opt_data: pd.DataFrame,
                      shibor_data: pd.DataFrame) -> pd.DataFrame:
    """前期数据均值

    Parameters
    ----------
    opt_data : pd.DataFrame
            期权合约数据
            | index | date      | exercise_date | close  | contract_type | exercise_price | maturity |
            | :---- | :-------- | :------------ | :----- | :------------ | :------------- | :------- |
            | 0     | 2021/7/29 | 2022/3/23     | 0.5275 | call          | 4.332          | 0.649315 |
    shibor_data : pd.DataFrame
            无风险收益数据
            | index    | 1      | 2        | 3        | ...      | 356      | 357   | 358     | 360      |
            | :------- | :----- | :------- | :------- | :------- | :------- | :---- | :------ | :------- |
            | 2015/1/4 | 0.0364 | 0.038687 | 0.040898 | 0.043026 | 0.045063 | 0.047 | 0.04883 | 0.050544 |

    Returns
    -------
    pd.DataFrame
        _description_
    """

    # step 1.获取每日的次月及次近月合约
    # 获取每日的近月和次近月期权合约
    filter_opt_data: pd.DataFrame = opt_data.groupby(
        'date', group_keys=False).apply(_get_near_or_next_options,
                                        filter_num=7)
    filter_opt_data['date'] = pd.to_datetime(filter_opt_data['date'])

    # 获取每日
    maturity_ser: pd.Series = filter_opt_data.groupby('date').apply(
        lambda x: x['maturity'].unique())

    # step 2.根据近月次近月合约获取对应的无风险收益
    # 近月、次近月
    shibor_data = shibor_data.loc[maturity_ser.index].copy()
    sel_rate: pd.Series = shibor_data.apply(lambda x: _get_free_rate(
        x, np.min(maturity_ser.loc[x.name]), np.max(maturity_ser.loc[x.name])),
        axis=1)

    # 根据maturity对齐shibor
    maturity_align, shibor_algin = maturity_ser.align(sel_rate,
                                                      axis=0,
                                                      join='left')

    # step 3.拼接数据
    # 储存中间变量
    df = pd.DataFrame(
        index=maturity_align.index,
        columns='near_maturity,next_maturity,near_rate,next_rate'.split(','))

    df['near_maturity'] = maturity_align.apply(lambda x: np.min(x))
    df['next_maturity'] = maturity_align.apply(lambda x: np.max(x))

    df['near_rate'] = shibor_algin.apply(lambda x: np.min(x))
    df['next_rate'] = shibor_algin.apply(lambda x: np.max(x))
    df.index.names = ['date']

    #opt_data['date'] = pd.to_datetime(opt_data['date'])
    data_all = pd.merge(filter_opt_data,
                        df.reset_index(),
                        on='date',
                        how='outer')

    return data_all


################################################################################################
#                                其他
################################################################################################
def get_quantreg_res(endog: pd.Series, exog: pd.Series) -> pd.DataFrame:
    """_summary_

    Args:
        endog (pd.Series): _description_
        exog (pd.Series): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # 分位数回归
    mod = sm.QuantReg(endog, sm.add_constant(exog))
    quantiles = np.arange(0.05, 1, 0.2)

    def fit_model(q) -> List:

        res = mod.fit(q=q)

        return [q, res.params.iloc[0], res.params.iloc[1]
                ] + res.conf_int().iloc[1].tolist()

    models = [fit_model(x) for x in quantiles]
    models = pd.DataFrame(models,
                          columns=["q", "Intercept", "vix", "lb", "ub"])

    return models


def get_n_next_ret(endog: pd.Series,
                   exog: pd.Series,
                   periods: List = [5, 20, 60]) -> Tuple:
    """构建未来N期收益并对齐信号与收益

    Args:
        endog (pd.Series): _description_
        exog (pd.Series): _description_

    Returns:
        Tuple: _description_
    """

    # 未来N日收益
    next_ret_name: List = ['未来%s日收益' % i for i in periods]

    # 生成未来N期的收益序列
    next_chg: pd.DataFrame = pd.concat(
        (endog.pct_change(i).shift(-i) for i in periods), axis=1)
    next_chg.columns = next_ret_name
    next_chg = next_chg.dropna()

    # 日期对齐
    algin_next_chg, algin_vix = next_chg.align(exog, join='left', axis=0)
    return algin_next_chg, algin_vix


def create_quantile_bound(signal: pd.Series, window: int,
                          bound: Tuple) -> pd.DataFrame:
    """构造滚动百分位数上下轨

    Args:
        signal (pd.Series): index-price
        window (int): 时间窗口
        bound (Tuple): 0-上轨百分位,1-下轨百分位

    Returns:
        pd.DataFrame: index-date columns-ub,signal,lb
    """
    up, lw = bound
    ub: pd.Series = signal.rolling(window).apply(
        lambda x: np.percentile(x, up), raw=True)
    lb: pd.Series = signal.rolling(window).apply(
        lambda x: np.percentile(x, lw), raw=True)

    df: pd.DataFrame = pd.concat((ub, signal, lb), axis=1)
    df.columns = ['ub', 'signal', 'lb']

    return df.dropna()


def get_hold_series(price: pd.Series, signal: pd.Series, window: int,
                    bound: Tuple) -> pd.Series:
    """获取持仓信号
       当signal小于下轨时开仓,大于上轨时平仓
    Args:
        price (pd.Series): _description_
        signal (pd.Series): _description_
        window (int): _description_
        bound (Tuple): _description_

    Returns:
        pd.Series: _description_
    """
    signal_df: pd.DataFrame = create_quantile_bound(signal, window, bound)
    hold = pd.Series(index=signal.index)

    previous_flag = 0

    for trade, rows in signal_df.iterrows():

        s: float = rows['signal']
        ub: float = rows['ub']
        lb: float = rows['lb']

        if s <= lb:

            hold.loc[trade] = 1
            previous_flag = 1

        elif s >= ub:

            hold.loc[trade] = 0
            previous_flag = 0

        else:

            hold.loc[trade] = previous_flag

    return hold

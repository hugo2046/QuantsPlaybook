'''
Author: shen.lan123@gmail.com
Date: 2022-04-27 22:54:24
LastEditTime: 2022-05-20 17:23:08
LastEditors: hugo2046 shen.lan123@gmail.com
Description: 用于因子合成

参考:《20190104-华泰证券-因子合成方法实证分析》
-------------
传入T期因子及收益数据 使用T-1至T-N期数据计算因子的合成权重

现有方法：
1. fac_eqwt 等权法
2. fac_ret_half 历史因子收益率（半衰）加权法
3. fac_ic_half 历史因子 IC(半衰)加权法
4. fac_maxicir_samp 最大化 IC_IR 加权法 样本协方差
    fac_maxicir  Ledoit压缩估计方法计算协方差
5. fac_maxic 最大化IC加权法 Ledoit压缩估计方法计算协方差
------
输入参数:
    factor:MuliIndex level0为date,level1为code,columns为因子值
        -----------------------------------
            date    |    asset   |
        -----------------------------------
                    |   AAPL     |   0.5
                    -----------------------
                    |   BA       |  -1.1
                    -----------------------
        2014-01-01  |   CMG      |   1.7
                    -----------------------
                    |   DAL      |  -0.1
                    -----------------------
                    |   LULU     |   2.7
                    -----------------------

    next_returns:下期收益率,结构与factor相同
'''

import numpy as np
import pandas as pd
from multiprocessing import Pool
import functools

import warnings
from scipy import stats
from scipy import optimize
import statsmodels.api as sm
from sklearn.decomposition import (PCA, IncrementalPCA)
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import ledoit_wolf
from scr.my_scr import (get_factor_columns, get_factor_rank)

from scr.utils import (calculate_best_chunk_size, rolling_windows)
from typing import (Tuple, List, Union, Dict, Callable)

CPU_WORKER_NUM = 6
"""utils"""


def compute_forward_returns(prices, periods=(1, 5, 10), filter_zscore=None):
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.

    Parameters
    ----------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float
        Sets forward returns greater than X standard deviations
        from the the mean to nan.
        Caution: this outlier filtering incorporates lookahead bias.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    """

    forward_returns = pd.DataFrame(index=pd.MultiIndex.from_product(
        [prices.index, prices.columns], names=['date', 'asset']))

    for period in periods:
        delta = prices.pct_change(period).shift(-period)

        if filter_zscore is not None:
            mask = abs(delta - delta.mean()) > (filter_zscore * delta.std())
            delta[mask] = np.nan

        forward_returns[period] = delta.stack()

    forward_returns.index = forward_returns.index.rename(['date', 'asset'])

    return forward_returns


def calc_information_coefficient(factors: pd.DataFrame) -> pd.DataFrame:
    """计算因子IC

    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret

    Returns:
        pd.DataFrame: index-date columns-code values-IC
    """
    def src_ic(group):
        group = group.fillna(0)
        f = group['next_ret']
        _ic = group[get_factor_columns(factors.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic

    ic = factors.groupby(level='date').apply(src_ic)
    return ic


def calc_ols(factors: pd.DataFrame) -> pd.DataFrame:
    """计算因子收益率

    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret

    Returns:
        pd.DataFrame: index-date columns-code values-IC
    """
    def _ols(x, y) -> float:
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=-1)[0]
        return m

    def src_ols(group):
        group = group.fillna(0)
        f = group['next_ret']

        ols = group[get_factor_columns(factors.columns)] \
            .apply(lambda x: _ols(x, f))
        return ols

    ols = factors.groupby(level='date').apply(src_ols)
    return ols


def _build_halflife_wight(T: int, H: int) -> np.array:
    '''
    生成半衰期权重

    $w_t = 2^{\frac{t-T-1}{H}}(t=1,2,...,T)$
    实际需要归一化,w^{'}_{t}=\frac{w_t}{\sumw_t}
    ------

    输入参数:
        T:期数
        H:半衰期参数
    '''

    periods = np.arange(1, T + 1)

    return np.power(2, np.divide(periods - T - 1, H)) * 0.5


def _explicit_solutions_icir(ic: pd.DataFrame, window: int,
                             fill_Neg: str) -> pd.Series:
    """计算ic ir的显示解

    Args:
        ic (pd.DataFrame): 过去一段时间的ic数据,index-date columns-code values IC
        window (int): ic的窗口
        fill_Neg (str): 空缺值的填充,normal小于0的部分使用0填充;mean小于0的部分使用均值填充

    Returns:
        pd.Series: index-date columns-code values-权重
    """
    mean_ic = ic.rolling(window).mean()
    std_ic = ic.rolling(window).std()

    ic_ir = mean_ic / std_ic

    if fill_Neg == 'normal':

        ic_ir = pd.DataFrame(np.where(ic_ir < 0, 0, ic_ir),
                             index=ic.index,
                             columns=ic.columns)

    elif fill_Neg == 'mean':

        ic_ir = pd.DataFrame(np.where(ic_ir < 0, mean_ic, ic_ir),
                             index=ic.index,
                             columns=ic.columns)

    weight = ic_ir.div(ic_ir.sum(axis=1), axis=0)
    return weight


def _opt_icir(ic: pd.DataFrame, target_func: Callable) -> pd.Series:
    """约束条件下优化失败时调用,_explicit_solutions_icir函数

    Args:
        ic (pd.DataFrame): index-因子名 value-因子在一段时间内得ic均值
        target_func (Callable): 求解目标函数

    Returns:
        pd.Series: index-factor_name values-权重
    """

    size = ic.shape[1]
    weight = np.random.random(size)
    # s.t w >= 0
    bounds = tuple((0, None) for _ in range(size))
    cons = {'type': 'eq', 'fun': lambda weight: np.sum(weight) - 1}

    res = optimize.minimize(fun=target_func,
                            x0=weight,
                            args=ic,
                            bounds=bounds,
                            constraints=cons)

    if res['success']:

        if isinstance(ic, np.ndarray):

            return res['x']

        if isinstance(ic, pd.DataFrame):

            return pd.Series(res['x'], index=ic.columns.tolist())

    else:
        warnings.warn('求解失败')
        return np.array([np.nan] * size)


def _target_cov_func(w: np.array, ic: pd.DataFrame) -> float:
    '''
    使用样本协方差
    最大化IC IR的目标函数
    ------
    输入参数:
        w:因子合成的权重
        ic:IC均值向量 数据为因子在过去一段时间的IC均值
    '''

    mean_ic = ic.mean(axis=0)

    return -np.divide(w.T @ mean_ic, np.sqrt(w @ np.cov(ic.T) @ w.T))


def _target_ledoit_func(w: np.array, ic: pd.DataFrame) -> float:
    '''
    使用ledoit协方差
    最大化IC IR的目标函数
    ------
    输入参数:
        w:因子合成的权重
        ic:IC均值向量 数据为因子在过去一段时间的IC均值
    '''
    mean_ic = ic.mean(axis=0)

    return -np.divide(w.T @ mean_ic, np.sqrt(w @ ledoit_wolf(ic)[0] @ w.T))


"""函数"""


def fac_eqwt(factors: pd.DataFrame) -> pd.DataFrame:
    """equal因子等权

    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret.

    Returns:
        pd.DataFrame: MultiIndex level0-date level1-code score
    """
    ind_name = get_factor_columns(factors.columns)

    score = factors[ind_name].mean(axis=1)

    return score.to_frame('score')


def fac_ret_half(factors: pd.DataFrame,
                 window: int,
                 halflife: bool = True) -> pd.Series:
    """历史因子收益率(半衰)加权法

    最近一段时期内历史因子收益率的算术平均值（或半衰权重下的加权平均值）作为权重进行相加
    如果这六个因子的历史因子收益率均值分别是 1、2、3、4、5、6,则每个因子的权重分别为：
    1/(1+2+3+4+5+6)= 1/21、2/(1+2+3+4+5+6)= 2/21、3/21、4/21、5/21、
    6/21,即为 4.76%、9.52%、14.29%、19.05%、23.81%、28.57%

    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret
        window (int): ic计算的窗口
        halflife (bool, optional): 默认为True使用半衰期加权,False为等权 . Defaults to True.

    Returns:
        pd.Series: MultiIndex level0-date level1-code score
    """

    # 获取因子收益率
    factor_returns = calc_ols(factors)

    ret_mean = factor_returns.rolling(window).mean()

    # 使用半衰期
    if halflife:

        weight = ret_mean / ret_mean.rolling(window).sum()

    else:
        # 未使用半衰期
        weight = ret_mean

    # 因子合成
    factors_ = factors[get_factor_columns(
        factors.columns)].transform(lambda x: x.shift(-1))
    score = factors_.mul(weight, axis=0).sum(axis=1)
    idx = score.index.levels[0][window - 1:]
    score = score.to_frame('score')
    return score.loc[idx]


def fac_ic_half(factors: pd.DataFrame,
                window: int,
                halflife: int = None) -> pd.Series:
    """历史因子 IC(半衰)加权法

    按照最近一段时期内历史RankIC的算术平均值(或半衰权重下的加权平均值)作为权重进行相加，
    得到新的合成后因子

    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret
        window (int): ic计算的窗口
        halflife (int, optional): 半衰期,1,2,4等 通常使用2. Defaults to None.

    Returns:
        pd.Series: MultiIndex level0-date level1-code score
    """
    if window > len(ic):
        raise ValueError('window参数不能大于%s' % len(ic))

    ic = calc_information_coefficient(factors)
    factors_ = factors[get_factor_columns(factors.columns)].groupby(
        level='date').transform(lambda x: x.shift(-1))

    if halflife:

        # 构造半衰期
        ic_weight = _build_halflife_wight(window, halflife)

        weight = ic.rolling(window).apply(
            lambda x: np.average(x, weights=ic_weight))

    else:

        weight = ic.rolling(window).mean()

    score = factors_.mul(weight).sum(axis=1)
    score = score.to_frame('score')
    idx = score.index.levels[0][window - 1:]
    return score.loc[idx]


def fac_maxicir_ledoit(factors: pd.DataFrame, window: int) -> pd.Series:
    """最大化 IC_IR 加权法ledoit
    以历史一段时间的复合因子平均IC值作为对复合因子下一期IC值的估计,
    以历史 IC 值的协方差矩阵作为对复合因子下一期波动率的估计
    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret
        window (int): ic计算的窗口
    Returns:
        pd.Series: MultiIndex level0-date level1-code score
    """
    if window > len(factors):
        raise ValueError('window参数不能大于%s' % len(factors))

    ic = calc_information_coefficient(factors)
    factors_ = factors[get_factor_columns(factors.columns)].groupby(
        level='date').transform(lambda x: x.shift(-1))

    ic_roll_mean = ic.rolling(window).mean()

    rolls = rolling_windows(ic_roll_mean.iloc[window - 1:], window)
    weights: Tuple[np.ndarray] = tuple(
        _opt_icir(x, _target_ledoit_func) for x in rolls)
    weights: pd.DataFrame = pd.DataFrame(
        weights,
        index=factors_.index.levels[0][window * 2 - 2:],
        columns=factors_.columns)

    score = factors_.mul(weights).sum(axis=1)
    score = score.to_frame('score')
    idx = score.index.levels[0][window * 2 - 2:]
    return score.loc[idx]


def fac_maxicir_cov(factors: pd.DataFrame, window: int) -> pd.Series:
    """最大化IC_IR加权法
    以历史一段时间的复合因子平均IC值作为对复合因子下一期IC值的估计,
    以历史 IC 值的协方差矩阵作为对复合因子下一期波动率的估计
    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret
        window (int): ic计算的窗口
    Returns:
        pd.Series: MultiIndex level0-date level1-code score
    """
    if window > len(factors):
        raise ValueError('window参数不能大于%s' % len(factors))

    ic = calc_information_coefficient(factors)
    factors_ = factors[get_factor_columns(factors.columns)].groupby(
        level='date').transform(lambda x: x.shift(-1))

    ic_roll_mean = ic.rolling(window).mean()

    rolls = rolling_windows(ic_roll_mean.iloc[window - 1:], window)
    weights: Tuple[np.ndarray] = tuple(
        _opt_icir(x, _target_cov_func) for x in rolls)
    weights: pd.DataFrame = pd.DataFrame(
        weights,
        index=factors_.index.levels[0][window * 2 - 2:],
        columns=factors_.columns)

    score = factors_.mul(weights).sum(axis=1)
    score = score.to_frame('score')
    idx = score.index.levels[0][window * 2 - 2:]
    return score.loc[idx]


def fac_maxic(factors: pd.DataFrame, window: int) -> pd.Series:
    '''
    最大化 IC 加权法,ledoit_wolf z_score

    $max IC = \frac{w.T * IC}{\sqrt{w.T * V *w}}

    𝑉是当前截面期因子值的相关系数矩阵(由于因子均进行过标准化,自身方差为1,因此相关系数矩阵亦是协方差阵)
    协方差使用压缩协方差矩阵估计方式

    使用约束解
    '''

    if window > len(factors):
        raise ValueError('window参数不能大于%s' % len(factors))

    ic = calc_information_coefficient(factors)
    factors_ = factors[get_factor_columns(factors.columns)].groupby(
        level='date').transform(lambda x: x.shift(-1))

    ic_roll_mean = ic.rolling(window).mean()

    z_score = (ic.fillna(0) - ic_roll_mean) / ic.rolling(window).std()
    rolls = rolling_windows(z_score.iloc[window - 1:].fillna(0), window)
    weights: Tuple[np.ndarray] = tuple(
        _opt_icir(x, _target_ledoit_func) for x in rolls)
    weights: pd.DataFrame = pd.DataFrame(
        weights,
        index=factors_.index.levels[0][window * 2 - 2:],
        columns=factors_.columns)

    score = factors_.mul(weights).sum(axis=1)
    score = score.to_frame('score')
    idx = score.index.levels[0][window * 2 - 2:]
    return score.loc[idx]


# def fac_pca2pool(factors: pd.DataFrame, window: int) -> pd.Series:
#     """pca

#     Parameters
#     ----------
#     factors : pd.DataFrame
#         MutliIndex level0-date level1-code
#         columns factors_name
#     window : int
#         滚动窗口

#     Returns
#     -------
#     pd.Series
#         MutliIndex level0-date level1-code
#     """

#     periods = factors.index.levels[0]
#     func = functools.partial(_calc_roll_pca, df=factors)
#     roll_idx = rolling_windows(periods.to_numpy(), window)
#     chunk_size = calculate_best_chunk_size(len(roll_idx), CPU_WORKER_NUM)

#     with Pool(processes=CPU_WORKER_NUM) as pool:

#         df = pd.concat((pool.imap(func, roll_idx, chunksize=chunk_size)))
#         # res_tuple: Tuple[pd.Series] = tuple(
#         #     pool.imap(func, roll_idx, chunksize=chunk_size))

#     return df  # pd.concat(res_tuple)


def fac_pca(factors: pd.DataFrame, window: int) -> pd.Series:
    """pca

    Parameters
    ----------
    factors : pd.DataFrame
        MutliIndex level0-date level1-code
        columns factors_name
    window : int
        滚动窗口

    Returns
    -------
    pd.Series
        MutliIndex level0-date level1-code
    """

    periods = factors.index.levels[0]
    factors_ = factors[get_factor_columns(factors.columns)]
    func = functools.partial(_calc_roll_pca, df=factors_)
    roll_idx = rolling_windows(periods.to_numpy(), window)
    ser = pd.concat(
        (func(idxs=idx).loc[slice(idx[-1], None)] for idx in roll_idx))

    ser = ser.to_frame('score')
    return ser.loc[periods[window:]]


def _calc_roll_pca(idxs: List, df: pd.DataFrame) -> pd.Series:

    return get_pca(df.loc[idxs])


def get_pca(df: pd.DataFrame) -> pd.Series:
    """获取PCA
       因子进行了标准化
    Parameters
    ----------
    df : pd.DataFrame
        MutliIndex-level0 date level1 code
        column 因子名称

    Returns
    -------
    pd.Series
        MutliIndex-level0 date level1 code
        values factor
    """
    pca = IncrementalPCA(n_components=1)  # PCA(n_components=1)
    scaler = StandardScaler()
    # 这里进行了标准化
    factor_scaler = scaler.fit_transform(df.fillna(0).values)

    ser = pd.Series(data=pca.fit_transform(factor_scaler).flatten(),
                    index=df.index)

    return ser


def factor_score_indicators(factors: pd.DataFrame,
                            score_method: str,
                            direction: Union[str, Dict] = 'ascending',
                            window: int = 5,
                            is_rank: bool = True) -> pd.DataFrame:
    """打分法中：多因子组合分析与单因子分析主要多出了以下两个过程：
        因子选择的过程：静态选择和动态选择

        单因子得分到多因子组合得分的过程，这个过程涉及到了各单因子得分该如何加总的问题

        主要的组合得分计算有以下几种方法：

        等权法：该方法对所有因子同等看待，不论其有效性的优劣

        IC加权:根据IC均值的大小决定因子的权重,IC高的因子,权重就大,IC的均值为滚动计算

        ICIR加权:根据因子ICIR的大小决定因子的权重,ICIR越大,权重越大,ICIR的值为滚动计算

    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret
        score_method (str): 打分方法，可选有'equal':因子等权，'ic':因子ic加权,'icir':因子icir加权
        direction (Union[str, Dict], optional):置所有因子的排序方向，
        'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高;
        当为dict时,可以分别对不同因子的排序方向进行设置. Defaults to 'ascending'.
        window (int, optional): ic或icir打分法时ic计算均值及标准差的数据量. Defaults to 5.
        is_rank (bool): 是否排序,False不排序,当为False时
    Returns:
        pd.DataFrame: MultiIndex level0-date level1-code score
    """

    score_method_func = {
        'equal': fac_eqwt,
        'ret_half': fac_ret_half,
        'ic_half': fac_ic_half,
        'maxicir_ledoit': fac_maxicir_ledoit,
        'maxicir_cov': fac_maxicir_cov,
        'maxic': fac_maxic,
        'pca': fac_pca
    }

    if is_rank:
        rank = get_factor_rank(factors, direction)
    else:
        rank = factors
    score = score_method_func[score_method](rank, window)
    score['next_ret'] = rank['next_ret'].loc[score.index]
    return score

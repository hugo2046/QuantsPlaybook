'''
Author: shen.lan123@gmail.com
Date: 2022-04-22 13:21:17
LastEditTime: 2022-05-20 17:22:35
LastEditors: hugo2046 shen.lan123@gmail.com
Description: 
'''
import functools
import alphalens as al
import pandas as pd
import empyrical as ep
import copy

import factor_tools.composition_factor as comp_factor
from factor_tools.composition_factor import compute_forward_returns
from .my_scr import (calc_group_ic, add_group, get_group_return)
from scipy import stats
from typing import (List, Tuple, Dict, Callable, Union)
from collections import namedtuple
import warnings

def get_information_table(ic_data: pd.DataFrame) -> pd.DataFrame:
    """计算IC的相关信息

    Args:
        ic_data (pd.DataFrame): index-date columns-IC

    Returns:
        pd.DataFrame: index - num columns-IC_Mean|IC_Std|Risk-Adjusted IC|t-stat(IC)|...
    """
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)
    return ic_summary_table


class analyze_factor_res(object):
    def __init__(self,
                 factors: pd.DataFrame,
                 ind_name: Union[str, List] = None,
                 direction: Union[str, Dict] = 'ascending') -> None:
        '''
        输入:factors MuliIndex level0-date level1-asset columns-factors
        ind_name (Union[str, List]): 需要分组的因子名
        direction (Union[str, Dict], optional):设置所有因子的排序方向，'ascending'表示因子值越大分数越高，
        'descending'表示因子值越小分数越;Defaults to 'ascending'.
        '''
        self.factors = factors.copy()
        self.ind_name = ind_name
        self.direction = direction

    def get_calc(self,
                 pricing: pd.DataFrame,
                 quantiles: int = 5) -> pd.DataFrame:
        """数据准备

        Args:
            pricing (pd.DataFrame): index-date columns-code value-close
            quantiles (int, optional): group_num (int, optional): 当为大于等于2的整数时,对股票平均分组;当为(0,0.5)之间的浮点数,
                                对股票分为3组,前group_num%为G01,后group_num%为G02,中间为G03. Defaults to 5.
        """
        next_returns: pd.DataFrame = compute_forward_returns(pricing, (1, ))

        # 分组
        group_factor = add_group(self.factors,
                                 ind_name=self.ind_name,
                                 group_num=quantiles,
                                 direction=self.direction)

        group_factor['next_ret'] = next_returns[1]
        self.factors['next_ret'] = next_returns[1]
        self.next_returns = next_returns

        # 因子分组
        self.group_factor = group_factor

        # 分组收益
        self.group_returns = get_group_return(group_factor)
        self.group_returns.columns.name = '分组'
        # 分组累计收益
        self.group_cum_returns = self.group_returns.groupby(
            level='factor_name').transform(lambda x: ep.cum_returns(x))

    def calc_ic(self) -> pd.DataFrame:

        ic_frame = calc_group_ic(self.factors, self.group_factor)
        return ic_frame


"""生成结果"""


def get_factor_res(dichotomy: pd.DataFrame, factors: pd.DataFrame,
                   pricing: pd.DataFrame, cat_type: Dict, **kws) -> Dict:
    """获取因子分析报告

    Args:
        dichotomy (pd.DataFrame): 象限区分表 MultiIndex level0-date level1-code
        factors (pd.DataFrame): 因子 MultiIndex level0-date level1-code
        pricing (pd.DataFrame): 价格数据 index-date columns-code values-price
        cat_type (Dict): k-label v- 0-查询 1-选择的因子

    Returns:
        Dict: _description_
    """
    res = {}

    ind_name = kws.get('ind_name', None)
    # k-与cat_type的key一致,v-direction
    direction = kws.get('direction', 'ascending')
    group_num = kws.get('group_num', 5)
    # 获取因子复合方法字典
    comp_params = kws.get('comp_params', None)

    func = functools.partial(get_factor_res2namedtuple,
                             factor_df=factors,
                             pricing=pricing,
                             categories_df=dichotomy,
                             comp_params=comp_params)

    for name, v in cat_type.items():

        if isinstance(direction, dict):

            des: dict = direction[name]

        else:

            des = direction

        res[name] = func(
            categories_dic={
                'cat_tuple': v,
                'ind_name': ind_name,
                'direction': des,
                'group_num': group_num
            })

    return res


def get_factor_res2namedtuple(factor_df: pd.DataFrame,
                              pricing: pd.DataFrame,
                              categories_df: pd.DataFrame,
                              categories_dic: Dict,
                              comp_params: Dict = None) -> namedtuple:
    """计算每个象限的因子收益情况

    Args:
        factors_df (pd.DataFrame): 因子分值
        pricing (pd.DataFrame): 价格 index-date columns-codes
        categories_df(pd.DataFrame):MultiIndex level0-date level1-asset columns-分类情况
        categories_dic (Dict):
            1. cat_tuple (Tuple): 0-分类筛选表达 1-因子组
            2. ind_name同add_group
            3. group_num同add_group
            4. direction同add_group
        comp_params (Dict):是否复合因子
            1. method:因子复合的方法
            2. window:计算IC均值的均值窗口
            3. is_rank:是否排序再复合因子
    Returns:
        namedtuple
    """
    factors_res = namedtuple(
        'factor_res',
        'quantile_returns,quantile_cum_returns,ic_info_table,group_factor,factor_frame'
    )

    # 从categories_dic获取参数
    cat_tuple = categories_dic['cat_tuple']
    ind_name = categories_dic.get('ind_name', None)
    direction = categories_dic.get('direction', 'ascending')
    group_num = categories_dic.get('group_num', 5)

    # 获取查询及所需因子名称
    q, factor_cols = cat_tuple

    sel_idx = categories_df.query(q).index
    test_factor = factor_df.loc[sel_idx, factor_cols]

    if isinstance(direction, dict):
        direction = copy.deepcopy(direction)
        direction['复合因子'] = 'ascending'

    # 因子复合参数
    if comp_params is not None:

        comp_method = comp_params['method']
        ic_window = comp_params['window']
        is_rank = comp_params.get('is_rank', True)
        test_factor['next_ret'] = compute_forward_returns(pricing, (1, ))[1]
        score_ser = comp_factor.factor_score_indicators(
            test_factor, comp_method, direction, ic_window, is_rank)['score']
        test_factor = test_factor.loc[score_ser.index]
        test_factor['复合因子'] = score_ser

    # 因子计算
    afr = analyze_factor_res(test_factor,
                             ind_name=ind_name,
                             direction=direction)

    afr.get_calc(pricing, group_num)

    # 计算ic
    ic_df = afr.calc_ic()

    ic_info_table = ic_df.groupby(
        level='factor_name').apply(lambda x: get_information_table(x.dropna()))

    ic_info_table['mean_ret'] = afr.group_returns.groupby(
        level='factor_name').mean().stack()

    ic_info_table.index.names = ['factor_name', 'group_num']

    quantile_returns = afr.group_returns
    quantile_cum_returns = afr.group_cum_returns

    return factors_res(quantile_returns, quantile_cum_returns, ic_info_table,
                       afr.group_factor, afr.factors)


"""策略收益率风险指标"""


def value_at_risk(returns: pd.Series,
                  period: str = None,
                  sigma: float = 2.0) -> float:
    """
    Get value at risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        策略每日收益率(不是累计收益率).
    period : str, optional
        计算风险指标的周期,weekly,monthly,yearly,daily.
    sigma : float, optional
        Standard deviations of VaR, default 2.
    """
    if period is not None:
        returns_agg = ep.aggregate_returns(returns, period)
    else:
        returns_agg = returns.copy()

    value_at_risk = returns_agg.mean() - sigma * returns_agg.std()
    return value_at_risk


SIMPLE_STAT_FUNCS = [
    ep.annual_return, ep.cum_returns_final, ep.annual_volatility,
    ep.sharpe_ratio, ep.calmar_ratio, ep.stability_of_timeseries,
    ep.max_drawdown, ep.omega_ratio, ep.sortino_ratio, stats.skew,
    stats.kurtosis, ep.tail_ratio, value_at_risk
]

FACTOR_STAT_FUNCS = [
    ep.alpha,
    ep.beta,
]

STAT_FUNC_NAMES = {
    'annual_return': 'Annual return',
    'cum_returns_final': 'Cumulative returns',
    'annual_volatility': 'Annual volatility',
    'sharpe_ratio': 'Sharpe ratio',
    'calmar_ratio': 'Calmar ratio',
    'stability_of_timeseries': 'Stability',
    'max_drawdown': 'Max drawdown',
    'omega_ratio': 'Omega ratio',
    'sortino_ratio': 'Sortino ratio',
    'skew': 'Skew',
    'kurtosis': 'Kurtosis',
    'tail_ratio': 'Tail ratio',
    'common_sense_ratio': 'Common sense ratio',
    'value_at_risk': 'Daily value at risk',
    'alpha': 'Alpha',
    'beta': 'Beta',
}


def perf_stats(returns,
               benchmark_returns: pd.Series = None,
               period: str = 'daily') -> pd.Series:
    """
    计算收益表

    Parameters
    ----------
    returns : pd.Series
        策略每日收益率(不是累计收益率).
    benchmark_returns : pd.Series, optional
    period:周期

    Returns
    -------
    pd.Series
        Performance metrics.
    """

    stats = pd.Series()

    for stat_func in SIMPLE_STAT_FUNCS:

        if 'period' in stat_func.__code__.co_varnames:

            stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(
                returns, period=period)

        else:

            stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)

    if benchmark_returns is not None:

        for stat_func in FACTOR_STAT_FUNCS:

            if stat_func.__name__ == 'alpha':

                res = stat_func(returns, benchmark_returns, period=period)

            else:

                res = stat_func(returns, benchmark_returns)

            stats[STAT_FUNC_NAMES[stat_func.__name__]] = res

    return stats


"""复合各象限因子收益"""


def get_com_returns(res: Dict,
                    factor_name: str,
                    group_num: int,
                    hold_num: float = None,
                    des: bool = True) -> namedtuple:
    """获取每个象限的对应情况

    Args:
        res (Dict):  
        factor_name (str): _description_
        group_num (int): _description_

    Returns:
        namedtuple: _description_
    """
    group_res = namedtuple('group_returns',
                           'factor_original_frame,factor_returns')

    bt_dict = {}
    sub_df_list = []
    for name, sub_res in res.items():

        bt_res = get_factor_target_group_returns(sub_res, factor_name,
                                                 group_num, hold_num, des)
        bt_dict[name] = bt_res

        sub_df_list.append(bt_res.factor_original_frame)

    df = pd.concat(sub_df_list)
    df.sort_index(inplace=True)
    idxs = df.index.drop_duplicates()
    df = df.loc[idxs]

    if des:
        df = df.groupby(level='date', group_keys=False).apply(
            lambda x: x.loc[x[factor_name].nlargest(hold_num).index])
    else:
        df = df.groupby(level='date', group_keys=False).apply(
            lambda x: x.loc[x[factor_name].nsmallest(hold_num).index])

    bt_dict[factor_name] = group_res(
        df,
        df.groupby(level='date')['next_ret'].mean())
    return bt_dict


def get_com_frame(com_dict: Dict) -> pd.DataFrame:

    dic = {}
    sub_list = []
    for name, res in com_dict.items():
        dic[name] = res.factor_returns
        sub_list.append(res.factor_original_frame)

    return pd.concat(dic)


def get_factor_target_group_returns(res: namedtuple,
                                    factor_name: str,
                                    group_num: int,
                                    hold_num: int = None,
                                    desc: bool = True) -> namedtuple:
    """
        提取res回测结果中某一个因子的特定组合
    Args:
        res (namedtuple): 回测结果
        factor_name (str): 因子名称
        group_num (int): 分组数量
        hold_num (int, optional): 当为None时择时整个分组,当有数值时,根据因子分值提取前N或后N的股票. Defaults to None.
        desc (bool, optional): True提取最大,False为最后. Defaults to True.

    Returns:
        namedtuple: factor_original_frame,factor_returns
    """
    group_res = namedtuple('group_returns',
                           'factor_original_frame,factor_returns')

    # 分组 MultiIndex level0-date level1-code
    filter_df: pd.DataFrame = res.group_factor.query(
        f'{factor_name} == @group_num').copy()

    # 因子数值 MultiIndex level0-date level1-code
    factor_df = res.factor_frame.copy()

    size = len(filter_df)

    # 防止重复
    duplicates_idx = filter_df.index.drop_duplicates(keep='last')

    # 去重后的因子值表格
    duplicates_frame = factor_df.loc[duplicates_idx]

    duplicates_size = len(duplicates_idx)

    if size != duplicates_size:

        warnings.warn('具有重复项,原始数据为%s,去重后为:%s' % (size, duplicates_size))

    if hold_num is not None:

        if desc:
            hold_frame = duplicates_frame.loc[duplicates_idx].groupby(
                level='date', group_keys=False).apply(
                    lambda x: x.loc[x[factor_name].nlargest(hold_num).index])

        else:
            hold_frame = duplicates_frame.loc[duplicates_idx].groupby(
                level='date', group_keys=False)[factor_name].apply(
                    lambda x: x.loc[x[factor_name].nsmallest(hold_num).index])

    else:

        hold_frame = duplicates_frame

    factor_returns = hold_frame.groupby(level='date')['next_ret'].mean()

    return group_res(hold_frame, factor_returns)

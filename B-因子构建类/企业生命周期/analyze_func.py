'''
Author: shen.lan123@gmail.com
Date: 2022-04-22 13:21:17
LastEditTime: 2022-05-05 14:46:50
LastEditors: hugo2046 shen.lan123@gmail.com
Description: 
'''
import functools
import alphalens as al
import pandas as pd
import empyrical as ep
import copy
from composition_factor import compute_forward_returns
from my_scr import (calc_group_ic, add_group, get_group_return,
                    get_information_table)
import composition_factor as comp_factor
from typing import (List, Tuple, Dict, Callable, Union)
from collections import namedtuple


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
        'factor_res', 'quantile_returns,quantile_cum_returns,ic_info_table')

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

    quantile_returns = afr.group_returns
    quantile_cum_returns = afr.group_cum_returns

    return factors_res(quantile_returns, quantile_cum_returns, ic_info_table)

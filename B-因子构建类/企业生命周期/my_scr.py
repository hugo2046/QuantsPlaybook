'''
Author: shen.lan123@gmail.com
Date: 2022-04-18 16:53:10
LastEditTime: 2022-05-06 17:06:42
LastEditors: hugo2046 shen.lan123@gmail.com
Description: 
'''
import _imports
from typing import (List, Tuple, Dict, Callable, Union)

# from Hugos_tools.Tdays import (Tdaysoffset, get_trade_period)
# from Hugos_tools.BuildStockPool import Filter_Stocks

# from my_factor import (quadrant, VolAvg, VolCV, RealizedSkewness, ILLIQ,
#                        Operatingprofit_FY1, BP_LR, EP_Fwd12M, Sales2EV,
#                        Gross_profit_margin_chg, Netprofit_chg)

# from jqfactor import calc_factors
# from jqdata import *

from tqdm import tqdm_notebook

import alphalens as al
import pandas as pd
import numpy as np
import empyrical as ep
from scipy import stats
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

__all__ = ['quadrant_dic', 'dichotomy_dic']
"""划分象限及高低端"""

# CATEGORY = {
#     'roe端==0':
#     ['VolAvg_20D_240D', 'VolCV_20D', 'RealizedSkewness_240D', 'ILLIQ_20D'],
#     'roe端==1': ['Operatingprofit_FY1_R20D'],
#     '增长端==1': ['BP_LR', 'EP_Fwd12M', 'Sales2EV'],
#     '增长端==0': ['Gross_profit_margin_chg', 'Netprofit_chg']
# }


# def _get_quadrant_dic():

#     dic = {
#         'cat_type == 2': ['roe端==0', '增长端==1'],
#         'cat_type == 1': ['roe端==1', '增长端==1'],
#         'cat_type == 3': ['增长端==0', 'roe端==0'],
#         'cat_type == 4': ['roe端==1', '增长端==0']
#     }

#     # 导入期 cat_type = 2
#     # 成长期 cat_type = 1
#     # 衰退期 cat_type = 3
#     # 成熟期 cat_type = 4

#     sub_dic = defaultdict(list)
#     out_put = {}
#     for label, (name, v) in zip(['导入期', '成长期', '衰退期', '成熟期'], dic.items()):

#         factor_set = set()
#         for i in v:

#             factor_set |= set(CATEGORY[i])

#         out_put[label] = (name, list(factor_set))

#     return out_put


# def _get_dichotomy_dic():

#     label = ['低roe端', '高roe端', '高增长端', '低增长端']

#     out_put = {name: (k, v) for name, (k, v) in zip(label, CATEGORY.items())}

#     return out_put

# 因子类别
FACTOR_DIC = {
    '量价指标':
    ['VolAvg_20D_240D', 'VolCV_20D', 'RealizedSkewness_240D', 'ILLIQ_20D'],
    '一致预期指标': ['Operatingprofit_FY1_R20D'],
    '价值稳定指标': ['BP_LR', 'EP_Fwd12M', 'Sales2EV'],
    '成长质量指标': ['Gross_profit_margin_chg', 'Netprofit_chg']
}
# 二元分类对应的因子
DICHOTOMY2FACTOR = {'低ROE端':'量价指标',
                    '高ROE端':'一致预期指标',
                    '低增长端':'价值稳定指标',
                    '高增长端':'成长质量指标'}

# 四象限分类对应的因子
QUADRANT2FACTOR = {'导入期':('量价指标','成长质量指标'),
                   '成长期':('一致预期指标','成长质量指标'),
                   '成熟期':('一致预期指标','价值稳定指标'),
                   '衰退期':('量价指标','价值稳定指标')}

# 二元分类的条件
DICHOTOMY_QUERY = {'低ROE端':'roe端==0',
                   '高ROE端':'roe端==1',
                   '低增长端':'增长端==0',
                   '高增长端':'增长端==1'}

# 四象限分类对应的条件
QUANDANT_QUERY = {'导入期':2,'成长期':1,'衰退期':3,'成熟期':4}

# 二元转为四象限
DICHOTOMY2QUANDANT = {'导入期':('高增长端','低ROE端'),
                      '成长期':('高ROE端','高增长端'),
                      '成熟期':('高ROE端','低增长端'),
                      '衰退期':('低增长端','低ROE端')}


def _get_dichotomy_dic()->Dict:
    dic = {}
    for k,v in DICHOTOMY2FACTOR.items():
        
        dic[k] = (DICHOTOMY_QUERY[k],FACTOR_DIC[v])
    return dic

def _get_quadrant_dic()->Dict:
    
    dic = {}

    for k,v in QUANDANT_QUERY.items():
        f_ls = []
        for i in QUADRANT2FACTOR[k]:
            
            f_ls += FACTOR_DIC[i]
        dic[k] = (f'cat_type == {v}',f_ls)
        
    return dic


quadrant_dic = _get_quadrant_dic()
dichotomy_dic = _get_dichotomy_dic()

"""划分象限"""


def get_daily_quadrant(watch_date: str,
                       method: str = 'ols',
                       is_scaler: bool = False) -> pd.DataFrame:
    """获取当日象限划分

    Parameters
    ----------
    watch_date : str
        观察日
    method : str, optional
        划分方式-主要是营收增长率的处理方式,quadrant类,详见, by default 'ols'
    is_scaler : bool, optional
        是否归一化处理, by default False

    Returns
    -------
    pd.DataFrame
        象限划分
    """
    trade = pd.to_datetime(watch_date)
    stock_pool_func = Filter_Stocks('A', trade)
    factor = quadrant()
    factor.method = method
    factor.is_scaler = is_scaler

    return calc_factors(stock_pool_func.securities, [factor], trade,
                        trade)['quadrant']


def get_daily_dichotomy(watch_date: str,
                        method: str = 'ols',
                        is_scaler: bool = False) -> pd.DataFrame:
    """获取二分象限

    Parameters
    ----------
    watch_date : str
        观察日
    method : str, optional
        划分方式-主要是营收增长率的处理方式,quadrant类,详见, by default 'ols'
    is_scaler : bool, optional
        是否归一化处理, by default False

    Returns
    -------
    pd.DataFrame
        划分
    """
    trade = pd.to_datetime(watch_date)
    stock_pool_func = Filter_Stocks('A', trade)
    factor = quadrant()
    factor.method = method
    factor.is_scaler = is_scaler

    calc_factors(stock_pool_func.securities, [factor], trade,
                 trade)['quadrant']
    return factor.dichotomy


def get_dichotomy(start: str,
                  end: str,
                  method: str = 'ols',
                  is_scaler: bool = False) -> pd.DataFrame:
    """获取区间象限划分合集

    Parameters
    ----------
    start : str
        开始日
    end : str
        结束日
    method : str, optional
        划分方式-主要是营收增长率的处理方式,quadrant类,详见, by default 'ols'
    is_scaler : bool, optional
        是否归一化处理, by default False
        
    Returns
    -------
    pd.DataFrame
        象限划分
    """
    periods = get_trade_period(start, end, 'ME')

    tmp = {}
    for trade in tqdm_notebook(periods, desc='划分高低端象限'):

        tmp[trade] = get_daily_dichotomy(trade, method, is_scaler)

    df = pd.concat(tmp, sort=True)
    return df


def get_quadrant(start: str,
                 end: str,
                 method: str = 'ols',
                 is_scaler: bool = False) -> pd.DataFrame:
    """获取区间象限划分合集

    Parameters
    ----------
    start : str
        开始日
    end : str
        结束日
    method : str, optional
        划分方式-主要是营收增长率的处理方式,quadrant类,详见, by default 'ols'
    is_scaler : bool, optional
        是否归一化处理, by default False
        
    Returns
    -------
    pd.DataFrame
        象限划分
    """
    periods = get_trade_period(start, end, 'ME')

    tmp = []
    for trade in tqdm_notebook(periods, desc='划分四象限'):

        tmp.append(get_daily_quadrant(trade, method, is_scaler))

    df = pd.concat(tmp, sort=True)
    return df


"""因子生成"""


def get_pricing(factor_df: pd.DataFrame,
                last_periods: str = None) -> pd.DataFrame:
    """获取价格数据

    Args:
        factor_df (pd.DataFrame): 因子数据  MultiIndex levels-0 date levels-1 code
        last_periods (str, optional): 最后一期数据. Defaults to None.

    Returns:
        pd.DataFrame
    """
    if last_periods is not None:
        periods = factor_df.index.levels[0].tolist() + [
            pd.to_datetime(last_periods)
        ]
    else:
        periods = factor_df.index.levels[0]

    securities = factor_df.index.levels[1].tolist()

    # 获取收盘价
    price_list = list(get_freq_price(securities, periods))
    price_df = pd.concat(price_list)
    pivot_price = pd.pivot_table(price_df,
                                 index='time',
                                 columns='code',
                                 values='close')
    return pivot_price


def get_freq_price(security: Union[List, str], periods: List) -> pd.DataFrame:
    """获取对应频率价格数据

    Args:
        security (Union[List, str]): 标的
        periods (List): 频率

    Yields:
        Iterator[pd.DataFrame]
    """
    for trade in tqdm_notebook(periods, desc='获取收盘价数据'):

        yield get_price(security,
                        end_date=trade,
                        count=1,
                        fields='close',
                        fq='post',
                        panel=False)


def trans2frame(dic: Dict) -> pd.DataFrame:
    def _func(df, col):
        ser = df.iloc[-1]
        ser.name = col
        return ser

    return pd.concat((_func(df, col) for col, df in dic.items()), axis=1)


def get_factors(quandrant_df: pd.DataFrame):

    periods = quandrant_df.index.tolist()
    codes = quandrant_df.columns.tolist()

    tmp = {}

    factors = [
        VolAvg(),
        VolCV(),
        RealizedSkewness(),
        ILLIQ(),
        Operatingprofit_FY1(),
        BP_LR(),
        EP_Fwd12M(),
        Sales2EV(),
        Gross_profit_margin_chg(),
        Netprofit_chg()
    ]

    for trade in tqdm_notebook(periods, desc='获取因子'):

        dic = calc_factors(codes, factors, trade, trade)
        tmp[trade] = trans2frame(dic)

    factor_df = pd.concat(tmp)
    factor_df.index.names = ['date', 'asset']

    return factor_df


"""构建因子分析相关数据"""


def get_factor_columns(columns: pd.Index) -> List:
    """获取因子名称

    Args:
        columns (pd.Index): _description_

    Returns:
        List: _description_
    """
    return [col for col in columns if col not in ['next_return', 'next_ret']]


def calc_group_ic(factor_data: pd.DataFrame,
                  group_data: pd.DataFrame) -> pd.DataFrame:
    """计算分组IC

    Args:
        factor_data (pd.DataFrame): MultiIndex level-0 date level-1 code columns
        group_data (pd.DataFrame): MultiIndex level-0 date level-1 code columns
        两表要对齐
    Returns:
        pd.DataFrame: _description_
    """
    def src_ic(group):

        group = group.dropna()

        f = group['next_ret']

        _ic = stats.spearmanr(group[factor_name], f)[0]

        return _ic

    cols = get_factor_columns(group_data.columns)
    dic = defaultdict(pd.DataFrame)
    for factor_name, group_ser in group_data[cols].items():

        for group_num, ser in group_ser.groupby(group_ser):

            idx = ser.index
            ic_ser = factor_data.loc[idx].groupby(level='date').apply(src_ic)

            dic[factor_name][group_num] = ic_ser

    ic_frame = pd.concat(dic)
    ic_frame.index.names = ['factor_name', 'date']
    return ic_frame


def get_group_return(df: pd.DataFrame, cols: List = None) -> pd.DataFrame:
    """计算分组收益率

    Args:
        df (pd.DataFrame): MultiIndex level0-date level1-code 
        cols (List, optional): 需要计算的因子. Defaults to None.

    Returns:
        pd.DataFrame
    """
    if cols is None:
        cols = get_factor_columns(df.columns)
    if isinstance(cols, str):
        cols = [cols]
    dic = defaultdict(pd.DataFrame)
    for col in cols:

        dic[col] = pd.pivot_table(df.reset_index(level=0),
                                  index='date',
                                  columns=col,
                                  values='next_ret')

    rets = pd.concat(dic)
    rets.index.names = ['factor_name', 'date']

    return rets


# def add_group(factors: pd.DataFrame,
#               ind_name: Union[str, List] = None,
#               group_num: int = 5,
#               direction: Union[str, Dict] = 'ascending') -> pd.DataFrame:
#     """分组函数

#     Args:
#         factors (pd.DataFrame): _description_
#         ind_name (Union[str, List]): 需要分组的因子名
#         group_num (int, optional): 当为大于等于2的整数时,对股票平均分组;当为(0,0.5)之间的浮点数,
#                                    对股票分为3组,前group_num%为G01,后group_num%为G02,中间为G03. Defaults to 5.
#         direction (Union[str, Dict], optional):设置所有因子的排序方向，'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高；
#                                                 Defaults to 'ascending'.

#     Returns:
#         pd.DataFrame: _description_
#     """

#     if ind_name is None:
#         ind_name = factors.columns.tolist()

#     if isinstance(ind_name, str):

#         ind_name = [ind_name]

#     if isinstance(direction, str):

#         direction = [direction] * len(ind_name)

#         direction = dict(zip(ind_name, direction))

#     dfs: List = []
#     for name, des in direction.items():

#         labels = list(map(int, range(1, group_num + 1)))
#         # 当降序时 扭转labels
#         if des == 'descending':
#             labels_dic = dict(zip(labels, labels[::-1]))

#         rank_ser: pd.Series = factors.groupby(
#             level='date')[name].transform(lambda x: pd.qcut(
#                 x, group_num, duplicates='drop', labels=False)) + 1

#         rank_ser.name = name
#         try:
#             labels_dic
#         except UnboundLocalError:
#             dfs.append(rank_ser)
#             continue

#         rank_ser = rank_ser.map(labels_dic)
#         dfs.append(rank_ser)

#     return pd.concat(dfs, axis=1)


def get_factor_rank(factors: pd.DataFrame,
                    direction: Union[str, Dict] = 'ascending') -> pd.DataFrame:
    """对因子进行排序

    Args:
        factors (pd.DataFrame): MultiIndex level0-date level1-code columns中需要含有next_ret
        direction (Union[str, Dict], optional):置所有因子的排序方向，
        'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高;
        当为dict时,可以分别对不同因子的排序方向进行设置. Defaults to 'ascending'. Defaults to 'ascending'.

    Returns:
        pd.DataFrame: MultiIndex level0-date level1-code columns-factors_name及next_ret value-ranke
    """
    rank = factors.copy()

    asc_dic = {"ascending": 1, 'descending': -1}

    if isinstance(direction, str):

        ind_name = get_factor_columns(rank)

        direction = [asc_dic[direction]] * len(ind_name)

        direction = dict(zip(ind_name, direction))

    if isinstance(direction, dict):

        ind_name = list(direction.keys())
        direction = {k: asc_dic[v] for k, v in direction.items()}

    rank[ind_name] = factors[ind_name].mul(direction, axis=1)

    return rank


def add_group(factors: pd.DataFrame,
              ind_name: Union[str, List] = None,
              group_num: int = 5,
              direction: Union[str, Dict] = 'ascending') -> pd.DataFrame:
    """分组函数

    Args:
        factors (pd.DataFrame): _description_
        ind_name (Union[str, List]): 需要分组的因子名
        group_num (int, optional): 当为大于等于2的整数时,对股票平均分组;当为(0,0.5)之间的浮点数,
                                   对股票分为3组,前group_num%为G01,后group_num%为G02,中间为G03. Defaults to 5.
        direction (Union[str, Dict], optional):设置所有因子的排序方向，'ascending'表示因子值越大分数越高，'descending'表示因子值越小分数越高；
                                                Defaults to 'ascending'.

    Returns:
        pd.DataFrame
    """

    asc_dic = {'ascending': 1, 'descending': -1}

    # 当未指定字段时 获取因子字段
    if ind_name is None:

        ind_name = get_factor_columns(factors)  # factors.columns.tolist()

    if isinstance(ind_name, str):

        ind_name = [ind_name]

    if isinstance(direction, str):

        direction = [direction] * len(ind_name)

        direction = dict(zip(ind_name, direction))

    rank = get_factor_rank(factors, direction)

    rank[ind_name] = rank[ind_name].groupby(level='date').transform(
        lambda x: pd.qcut(x, group_num, duplicates='drop', labels=False)) + 1

    return rank


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


# def get_quantile_ic(factor_frame: pd.DataFrame) -> pd.DataFrame:
#     """获取分组IC相关信息

#     Args:
#         factor_frame (pd.DataFrame): MultiIndex level0-date level1-asset

#     Returns:
#         pd.DataFrame
#     """
#     tmp = []
#     for num, df in factor_frame.groupby('factor_quantile'):

#         ic = al.performance.factor_information_coefficient(df, False)
#         ic_info = get_information_table(ic)
#         #ic_info = ic_info.T
#         ic_info.index = [num]
#         tmp.append(ic_info)

#     return pd.concat(tmp)

'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-11-10 14:42:37
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-11-14 19:46:46
Description: 读取data中的excel文件
'''
import functools
from pathlib import PurePath
from typing import Dict, List, Tuple, Union

import pandas as pd


def query_data(codes: Union[str, List],
               start_date: str,
               end_date: str,
               method: str,
               fields: Union[str, List] = None) -> pd.DataFrame:

    # 标准化时间格式
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    # 检查字段
    fields: Tuple = _check_fields(fields)
    # 检查codes
    codes: List = _check_codes(codes)
    # 数据读取
    data: pd.DataFrame = load_excel(method, fields)
    # 标准化日期
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    sel_codes: pd.Series = data['code'].isin(codes)
    
    return data[sel_codes].query(
        'trade_date>=@start_date & trade_date<=@end_date')


def _check_codes(codes: Union[str, List]) -> List:

    if codes:
        return [codes] if isinstance(codes, str) else codes
    else:

        raise ValueError('codes不能为空')


def _check_fields(fields: Union[str, List]) -> Tuple:

    if fields:
        BASE_FIELDS: List = ['trade_date', 'code']
        fields: List = [field for field in fields if field not in BASE_FIELDS]

        return tuple(BASE_FIELDS + fields)
    else:

        raise ValueError('fields参数不能为空')


@functools.lru_cache()
def query_sw_classify() -> Dict:
    """查询申万一级行业列表
    
    Returns:
        Dict:key-code values-industry_name
    """
    df: pd.DataFrame = load_excel('sw', ('code', 'industry_name'))
    return df.set_index('code')['industry_name'].to_dict()


@functools.lru_cache()
def query_stock_index_classify() -> Dict:
    """查询宽基指数列表
    
    Returns:
        Dict:key-code values-index_name
    """
    df: pd.DataFrame = load_excel('index', ('code', 'index_name'))
    return df.set_index('code')['index_name'].to_dict()


@functools.lru_cache()
def load_excel(method: str, col_names: Tuple) -> pd.DataFrame:
    """加载excel

    Args:
        method (str): sw-为行业数据 index-为宽基指数
        col_names (Tuple): 字段名
    Returns:
        pd.DataFrame: index-pd.IndexTime
    """
    data_path: str = PurePath(__file__)
    file_name_dic: Dict = {
        'sw': 'sw2021_level1.xlsx',
        'index': 'stock_index.xlsx'
    }

    workbook_name: str = file_name_dic[method]

    return pd.read_excel(rf'{data_path.parents[1]}\data\{workbook_name}',
                         usecols=col_names)

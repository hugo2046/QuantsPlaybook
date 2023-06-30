'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-04-04 10:49:17
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-30 14:20:31
Description: 
'''

from alphalens.utils import quantize_factor
import pandas as pd
from typing import Dict,List


def drop_col_mulitindex2index(factor_data: pd.DataFrame) -> pd.DataFrame:
    """预处理因子数据,将多层索引转换为单层索引

    Args:
        factor_data (pd.DataFrame): MultiIndex level0:datetime level1:instrument MultiColumns level0:feature level1:label

    Returns:
        pd.DataFrame: MultiIndex level0:date level1:assert columns->factor next_ret
    """
    clean_factor: pd.DataFrame = factor_data.copy()
    if isinstance(clean_factor.columns,pd.MultiIndex):
        clean_factor.columns = clean_factor.columns.droplevel(0)
        
    clean_factor.index.names = ["date", "assert"]

    return clean_factor


def get_factor_group_returns(
    clean_factor: pd.DataFrame, quantile: int, no_raise: bool = False
) -> pd.DataFrame:
    """获取单因子分组收益

    Args:
        clean_factor (pd.DataFrame): MultiIndex level0:date level1:assert columns->factor next_ret
        quantile (int): 分组
        no_raise (bool, optional):Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    sel_cols: List = [col for col in clean_factor.columns.tolist() if col != "next_ret"]

    returns_dict: Dict = {}
    for col in sel_cols:
        clean_factor[f"{col}_group"] = quantize_factor(
            clean_factor.rename(columns={col: "factor"})[["factor"]].dropna(),
            quantiles=quantile,
            no_raise=no_raise,
        )
        returns_dict[col] = pd.pivot_table(
            clean_factor.reset_index(),
            index="date",
            columns=f"{col}_group",
            values="next_ret",
        )

    df: pd.DataFrame = pd.concat(returns_dict, axis=1)
    df.index.names = ["date"]
    df.columns.names = ["factor_name", "group"]
    return df




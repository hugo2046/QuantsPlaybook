'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-22 13:22:49
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-28 11:13:04
Description: 
'''
from typing import Dict, List

import numpy as np
import pandas as pd
import talib


# 构造HMA
def HMA(price: pd.Series, window: int) -> pd.Series:
    """HMA均线

    Args:
        price (pd.Series): 价格数据. index-date values
        window (int): 计算窗口

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values
    """
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    hma = talib.WMA(
        2 * talib.WMA(price, int(window * 0.5)) - talib.WMA(price, window),
        int(np.sqrt(window)))

    return hma


def get_exchange_set(billboard: pd.DataFrame) -> Dict:
    """划分机构席位类型
       营业部席位,机构席位,交易单元席位,沪股通席位,其他,券商总部席位
    Args:
        billboard (pd.DataFrame): _description_

    Returns:
        Dict: k-分类类型
              v-布尔值
    """
    # 营业部
    cond1 = billboard['exalter'].str.contains('营业?|分公司', regex=True)
    # 机构专用
    cond2 = billboard['exalter'].str.contains('机构专用|机构投资者', regex=True)
    # 交易单元
    cond3 = billboard['exalter'].str.contains('交易单元')
    # 深股通
    cond4 = billboard['exalter'].str.contains('深股通专用|沪股通专用|深股通投资者', regex=True)
    # 其他
    cond5 = billboard['exalter'].str.contains('自然人|其他自然人|中小投资者|投资者分类',
                                              regex=True)

    # 营业部总部
    cond6 = (~cond1) & (~cond2) & (~cond3) & (~cond4) & (~cond5)

    return {
        '营业部席位': cond1,
        '机构席位': cond2,
        '交易单元席位': cond3,
        '沪股通席位': cond4,
        '其他': cond5,
        '券商总部席位': cond6
    }


def calc_netbuy(billboard: pd.DataFrame, index_amount: pd.Series) -> pd.Series:
    """计算全市场机构席位绝对净流入金额

    Args:
        billboard (pd.DataFrame): 
        hs300_vol (pd.Series): index-date volume

    Returns:
        pd.Series: index-date values-netbuy
    """
    # 计算当日所有单个席位净流入金额(单个席位买入 金额-单个席位卖出金额)的总和
    netbuy: pd.Series = billboard.groupby('trade_date')['net_buy'].sum()
    # IS_NetBuy/沪深 300 指数当日成交金额
    is_netbuy_s: pd.Series = netbuy / (index_amount * 1000)
    is_netbuy_s: pd.Series = is_netbuy_s.dropna()

    return is_netbuy_s
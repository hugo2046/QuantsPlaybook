'''
Author: Hugo
Date: 2022-04-16 21:44:26
LastEditTime: 2022-04-16 22:33:39
LastEditors: Please set LastEditors
Description: 生成结果页面
'''
import build_timing_signal
import _imports
import pyvisflow as pvf
from typing import (List, Tuple, Union, Dict)
from collections import (namedtuple, defaultdict)
from plotting import plot_subplots
from tqdm import tqdm

import pandas as pd
import numpy as np
import json


# 形态分类
CATEGORIES_DIC:Dict = {
    '头肩顶(HS)': '下跌形态',
    '头肩底(IHS)': '上涨形态',
    '顶部发散(BTOP)': '震荡形态',
    '底部发散(BBOT)': '震荡形态',
    '顶部收敛三角形(TTOP)': '震荡形态',
    '底部收敛三角形(TBOT)': '震荡形态',
    '顶部矩形(RTOP)': '下跌形态',
    '底部矩形(RBOT)': '上涨形态'
}


def load_data()->pd.DataFrame:
    """读取形态识别所需数据

    Returns
    -------
    pd.DataFrame
    """
    with open(r'Data\code2secname.json', 'r') as file:
        code2secname = json.loads(file.read())

    price = pd.read_csv(r'Data\industry_zx_l2.csv', index_col=[0], parse_dates=[0])
    # price.index.names = ['trade']
    price['symbol'] = price['symbol'].map(code2secname)

    price.columns = [col.lower() for col in price.columns]

    return price


def get_last_patterns(res: namedtuple) -> namedtuple:
    """获取最后一期的形态识别

    Args:
        res (namedtuple): _description_

    Returns:
        namedtuple:patterns - 形态名称
                   points - 匹配符合的数据点
    """
    points_dic = res.points.copy()

    re_dic = {}

    for k, v in points_dic.items():
        if len(v) > 1:
            re_dic[k] = np.sort(v)[-1]
        else:
            re_dic[k] = v

    try:
        last_key = sorted(re_dic)[-1]
    except IndexError:

        return {}
    last_res = namedtuple('last_res', 'patterns,points')

    if isinstance(re_dic[last_key], list):
        tmp = re_dic[last_key]
    else:
        tmp = [re_dic[last_key]]
    last_res = last_res._make([last_key, {last_key: tmp}])

    return last_res

def pattern_recognition(price:pd.DataFrame)->Tuple[pd.DataFrame,Dict]:
    """获取结果所需

    Parameters
    ----------
    price : pd.DataFrame
        _description_

    Returns
    -------
    Tuple[pd.DataFrame,Dict]
        分类结果,画图所需
    """
    res = {}

    price.columns = [col.lower() for col in price.columns]

    for name, df in tqdm(price.groupby('symbol'), desc='形态识别中'):

        slice_df = df.copy()
        slice_df.drop(columns=['symbol'], inplace=True)
        res[name] = build_timing_signal.get_shorttimeseries_pattern(
            slice_df['close'],
            save_all=True,
            smooth_func=build_timing_signal.calc_smooth)

    re_res = {k: get_last_patterns(v) for k, v in res.items() if v.patterns}

    cluster_list = []

    for k, v in re_res.items():

        cp = CATEGORIES_DIC[v.patterns]

        cluster_list.append((k, v.patterns, cp))

    patterns_df = pd.DataFrame(cluster_list, columns=['行业指数', '识别形态', '方向'])

    return patterns_df,re_res



def block_zx_level2_pattern():

    price = load_data()
    patterns_df,re_res = pattern_recognition(price)
    
    

    pvf.markdown("""# 行业指数形态匹配情况""")

    table = pvf.dataTable(patterns_df.sort_values('方向'))
    page = len(re_res)//5 + 1
    table.page_size = page # 设置页面
    
    pvf.markdown("""*形态匹配结果*""")
    box = pvf.box()
    fig = plot_subplots(re_res,price)
    box.plotly().from_dict(fig.to_dict())
  
    box.styles.set_height('150vh').set('overflow-y', 'auto')
    pvf.to_html('形态识别结果.html')

block_zx_level2_pattern()
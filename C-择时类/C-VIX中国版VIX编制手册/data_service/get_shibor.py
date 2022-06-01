'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-30 11:35:26
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-01 10:38:29
FilePath: 
Description: 使用爬虫获取shibor
    上海银行业同业拆借报告:
    https://datacenter.jin10.com/reportType/dc_shibor
    
    上海银行业同业拆借报告, 数据区间从20170317-至今,
    shibor_data中储存的2006至20170316的数据
    
'''
import functools
import json
import time

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d


@functools.lru_cache()
def query_china_shibor_all() -> pd.DataFrame:
    """    
    上海银行业同业拆借报告, 数据区间从20170317-至今
    https://datacenter.jin10.com/reportType/dc_shibor
    https://cdn.jin10.com/dc/reports/dc_shibor_all.js?v=1578755058

    Returns:
        pd.DataFrame:
        | indx      | O/N   | 1W    | 2W    | 1M     | 3M     | 6M     | 9M    | 1Y     |
        | :-------- | :---- | :---- | :---- | :----- | :----- | :----- | :---- | :----- |
        | 2017/3/17 | 2.633 | 2.725 | 3.236 | 4.2775 | 4.3507 | 4.2909 | 4.134 | 4.1246 |

        O/N隔夜 
        单位:%
    """
    t = time.time()
    params = {"_": t}
    res = requests.get("https://cdn.jin10.com/data_center/reports/il_1.json",
                       params=params)
    json_data = res.json()
    temp_df = pd.DataFrame(json_data["values"]).T
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df = temp_df.applymap(lambda x: x[0])
    temp_df = temp_df.astype(float)
    return temp_df


@functools.lru_cache()
def _load_csv() -> pd.DataFrame:
    """加载shibor数据前期数据
    Returns:
        pd.DataFrame:
        | indx      | O/N   | 1W    | 2W    | 1M     | 3M     | 6M     | 9M    | 1Y     |
        | :-------- | :---- | :---- | :---- | :----- | :----- | :----- | :---- | :----- |
        | 2006/01/01 | 2.633 | 2.725 | 3.236 | 4.2775 | 4.3507 | 4.2909 | 4.134 | 4.1246 |
    """

    return pd.read_csv(r'data_service/shibor_data/shibor_db.csv',
                       index_col=[0],
                       parse_dates=True,
                       usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])


def get_shibor_data(start: str, end: str) -> pd.DataFrame:
    """获取区间shibor数据(单位:%)

    Args:
        start (str): 起始日
        end (str): 截止日

    Returns:
        pd.DataFrame
        | indx      | O/N   | 1W    | 2W    | 1M     | 3M     | 6M     | 9M    | 1Y     |
        | :-------- | :---- | :---- | :---- | :----- | :----- | :----- | :---- | :----- |
        | 2017/3/17 | 2.633 | 2.725 | 3.236 | 4.2775 | 4.3507 | 4.2909 | 4.134 | 4.1246 |
    """

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    df2 = query_china_shibor_all()  # 爬虫数据
    df1 = _load_csv()  # 用于补充数据

    df = pd.concat((df1, df2)).sort_index()
    df = df.loc[~df.index.duplicated(keep='last')]

    return df.loc[start:end]


def get_interpld_shibor(shibor_df: pd.DataFrame) -> pd.DataFrame:
    """获取差值后的shibor数据
       采用三次样条插值法补全利率曲线
     Args:
         shibor_df (pd.DataFrame): 
            | indx      | O/N   | 1W    | 2W    | 1M     | 3M     | 6M     | 9M    | 1Y     |
            | :-------- | :---- | :---- | :---- | :----- | :----- | :----- | :---- | :----- |
            | 2017/3/17 | 2.633 | 2.725 | 3.236 | 4.2775 | 4.3507 | 4.2909 | 4.134 | 4.1246 |

     Returns:
         pd.DataFrame:
            | index    | 1      | 2        | 3        | ...      | 356      | 357   | 358     | 360      |
            | :------- | :----- | :------- | :------- | :------- | :------- | :---- | :------ | :------- |
            | 2015/1/4 | 0.0364 | 0.038687 | 0.040898 | 0.043026 | 0.045063 | 0.047 | 0.04883 | 0.050544 |
     """
    def _interpld_fun(r):
        """用于差值"""
        y_vals = r.values / 100

        daily_range = np.arange(1, 365)
        periods = [1, 7, 14, 30, 90, 180, 270, 365]

        # 插值三次样条插值法补全利率曲线
        f = interp1d(periods, y_vals, kind='cubic')
        t_ser = pd.Series(data=f(daily_range), index=daily_range)

        return t_ser

    shibor_df = shibor_df.apply(lambda x: _interpld_fun(x), axis=1)

    shibor_df.index = pd.DatetimeIndex(shibor_df.index)

    return shibor_df


# if __name__ == '__main__':

#     df = _load_csv()
#     print(df)

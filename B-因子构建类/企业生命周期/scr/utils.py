'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-20 12:24:32
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-05-20 12:51:57
Description: 
'''
from typing import (List, Tuple, Dict, Union)
import pandas as pd
import numpy as np


def get_max_factor_name(ic_info_table: pd.DataFrame,
                        max_num: int,
                        group_num: int = 3) -> List:
    """获取ic_table中强势表现得因子
      

    Args:
        ic_info_table (pd.DataFrame): 获取其中得ic_table MultiIndex:level0-factor_name level1-group_num
        max_num (int): 获取因子的前N
        group_num (int):因子分组
    Returns:
        List: 因子列表
    """
    idx = pd.IndexSlice
    # 每个因子中第三组收益中最强的
    top = ic_info_table.loc[idx[:, group_num], 'mean_ret'].nlargest(max_num)
    return top.index.get_level_values(0).tolist()


def rolling_windows(df: Union[pd.DataFrame, pd.Series, np.ndarray],
                    window: int) -> List:
    """Creates rolling-window 'blocks' of length `window` from `a`.
    Note that the orientation of rows/columns follows that of pandas.
    Example
    -------
    import numpy as np
    onedim = np.arange(20)
    twodim = onedim.reshape((5,4))
    print(twodim)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]
     [16 17 18 19]]
    print(rwindows(onedim, 3)[:5])
    [[0 1 2]
     [1 2 3]
     [2 3 4]
     [3 4 5]
     [4 5 6]]
    print(rwindows(twodim, 3)[:5])
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
     [[ 4  5  6  7]
      [ 8  9 10 11]
      [12 13 14 15]]
     [[ 8  9 10 11]
      [12 13 14 15]
      [16 17 18 19]]]
    """

    if window > df.shape[0]:
        raise ValueError("Specified `window` length of {0} exceeds length of"
                         " `a`, {1}.".format(window, df.shape[0]))
    if isinstance(df, (pd.Series, pd.DataFrame)):
        df = df.values
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    shape = (df.shape[0] - window + 1, window) + df.shape[1:]
    strides = (df.strides[0], ) + df.strides
    windows = np.squeeze(
        np.lib.stride_tricks.as_strided(df, shape=shape, strides=strides))
    # In cases where window == len(a), we actually want to "unsqueeze" to 2d.
    #     I.e., we still want a "windowed" structure with 1 window.
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


def calculate_best_chunk_size(data_length: int, n_workers: int) -> int:

    chunk_size, extra = divmod(data_length, n_workers * 5)
    if extra:
        chunk_size += 1
    return chunk_size


def get_factor_columns(columns: pd.Index) -> List:
    """获取因子名称

    Args:
        columns (pd.Index): _description_

    Returns:
        List: _description_
    """
    return [col for col in columns if col not in ['next_return', 'next_ret']]
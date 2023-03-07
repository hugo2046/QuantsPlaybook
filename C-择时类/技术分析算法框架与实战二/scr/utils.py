"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-02-07 16:01:09
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-02-07 16:01:24
Description: 
"""

import numpy as np
import pandas as pd


def calc_smooth(price: pd.Series, h: float, **kw) -> pd.Series:
    """使用KernelRidge平滑数据

    Args:
        price (pd.Series): index-date values
        h (float): 带宽参数 str->cv_ls aic

    Returns:
        pd.Series: index-date values-smooth
    """
    from statsmodels.nonparametric.kernel_regression import KernelReg

    X: np.ndarray = np.arange(len(price))
    if isinstance(h, (float, int)):
        h = [h]

    kr = KernelReg(price, X, reg_type="ll", var_type="c", bw=h)
    return pd.Series(kr.fit(X)[0], index=price.index)

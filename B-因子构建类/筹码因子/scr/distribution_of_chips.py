"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-03-27 15:02:44
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-03-29 10:50:17
Description: 筹码分布算法(distribution_of_chips)
"""
import numpy as np
from numba import jit
# from scipy.stats import triang, uniform

#################### 历史换手衰减系数 ####################


def calc_adj_turnover(turnover_arr: np.ndarray) -> np.ndarray:
    """调整换手率

    Args:
        turnover_arr (np.ndarray): 换手率数组

    Returns:
        np.ndarray: 调整换手率
    """
    if not isinstance(turnover_arr, np.ndarray):
        raise TypeError("turnover_arr must be np.ndarray")

    if turnover_arr.ndim != 1:
        turnover_arr = turnover_arr.flatten()

    arr = turnover_arr.copy()
    arr = np.roll(arr, -1)
    arr[-1] = 0

    return np.multiply(np.flip(np.flip(np.subtract(1, arr)).cumprod()), turnover_arr)


def calc_normalization_turnover(turnover_arr: np.ndarray) -> np.ndarray:
    """计算归一化换手率权重"""
    if not isinstance(turnover_arr, np.ndarray):
        raise TypeError("turnover_arr must be np.ndarray")
    if turnover_arr.ndim != 1:
        turnover_arr = turnover_arr.flatten()
    adj_turnover: np.ndarray = calc_adj_turnover(turnover_arr)
    return adj_turnover / adj_turnover.sum()


#################### 三角分布 ####################

# scipy.stats.triang.pdf的实现
# 使用scipy速度过慢 这里使用numba加速
# 这里减少了大量的数据检查
@jit(nopython=True)
def triang_pdf(x, c, loc, scale):
    """
    自定义实现的三角分布概率密度函数

    参数：
    x: float or array_like
        输入变量
    c: float
        三角分布的众数
    loc: float
        三角分布的下限
    scale: float
        三角分布的长度

    返回：
    pdf: float or ndarray
        对于每一个x值的三角分布概率密度函数值
    """
    # 设置输出数组
    pdf = np.empty_like(x)

    if (c > 1 or c < 0) or (scale < 0):

        return pdf * np.nan

    peak: float = loc + scale * c
    upper: float = loc + scale

    square_scale: float = scale**2

    # 在区间内计算输出
    for i in range(len(x)):

        if loc <= x[i] <= peak:

            if c == 0:

                pdf[i] = np.nan # 2 - 2 * np.divide(x[i] - loc, c * square_scale)

            else:

                pdf[i] = 2 * np.divide(x[i] - loc, c * square_scale)

        elif peak < x[i] <= upper:

            pdf[i] = 2 * np.divide(upper - x[i], square_scale * (1 - c))

        else:

            pdf[i] = 0.0

    return pdf


def calc_triang_pdf(
    close: float,
    high: float,
    low: float,
    vol: float,
    min_p: float = None,
    max_p: float = None,
    step: float = 0.01,
) -> np.ndarray:
    """三角分布

    Args:
        close (float): close或者avg
        high (float): 最高价
        low (float): 最低价
        vol (float): 成交量
        min_p (float, optional): N日的最低价. Defaults to None.
        max_p (float, optional): N日的最高价. Defaults to None.
        step (float, optional): min_p至max_p的步长. Defaults to 0.01.

        注意:min_p或max_p为None时,则取low或high

    Returns:
        np.ndarray: 成交量分布
    """
    if (min_p is None) or (max_p is None):
        min_p, max_p = low, high
    c: float = np.divide(close - low, high - low)
    x: np.ndarray = np.arange(min_p, max_p + step, step)
    # c表示三角分布峰位,scale表示三角分布的宽度,loc表示三角分布的左边界
    # x表示生成x的概率密度
    # pdf: np.ndarray = triang.pdf(x, c, loc=low, scale=high - low)

    try:
        pdf: np.ndarray = triang_pdf(x, c, loc=low, scale=high - low)
    except ZeroDivisionError as e:
        print(f"c:{c},loc:{low},low:{low},high:{high},scale:{high - low}")
        raise e
    return pdf / np.sum(pdf) * vol


#################### 平均分布 ####################

@jit(nopython=True)
def uniform_pdf(x,loc=0,scale=1)->np.ndarray:
    if scale == 0:
        return np.zeros_like(x) * np.nan
    a:np.ndarray = (x - loc) / scale
    return np.where((a >= 0) & (a <= 1), 1 / scale, 0)

def calc_uniform_pdf(
    close: float,
    high: float,
    low: float,
    vol: float,
    min_p: float = None,
    max_p: float = None,
    step: float = 0.01,
) -> np.ndarray:
    """平均分布

    close (float): close或者avg
        high (float): 最高价
        low (float): 最低价
        vol (float): 成交量
        min_p (float, optional): N日的最低价. Defaults to None.
        max_p (float, optional): N日的最高价. Defaults to None.
        step (float, optional): min_p至max_p的步长. Defaults to 0.01.

        注意:min_p或max_p为None时,则取low或high

    Returns:
        np.ndarray: 成交量分布
    """

    if (min_p is None) or (max_p is None):
        min_p, max_p = low, high
    x: np.ndarray = np.arange(min_p, max_p + step, step)
    # loc表示均匀分布的左边界,scale表示均匀分布的宽度
    # pdf: np.ndarray = uniform.pdf(x, loc=close, scale=high - low)
    pdf: np.ndarray = uniform_pdf(x, loc=close, scale=high - low)
    return pdf / np.sum(pdf) * vol

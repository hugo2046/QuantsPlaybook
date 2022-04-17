'''
Author: Hugo
Date: 2022-02-11 22:42:44
LastEditTime: 2022-02-22 10:33:13
LastEditors: Please set LastEditors
Description: 复现的金工研报择时指标
'''

from email.policy import default
from typing import (List, Tuple, Dict, Union, Callable, Any)
import math
import warnings

import pandas as pd
import numpy as np

import talib

from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.rolling import (RollingWLS, RollingOLS)


"""
RSRS

from:
    https://www.joinquant.com/view/community/detail/1f0faa953856129e5826979ff9b68095
    https://www.joinquant.com/view/community/detail/32b60d05f16c7d719d7fb836687504d6
    https://www.joinquant.com/view/community/detail/e855e5b3cf6a3f9219583c2281e4d048
"""

# RSRS计算的类


class RSRS(object):

    def __init__(self, data: pd.DataFrame) -> None:
        """数据加载

        Args:
            data (pd.DataFrame): index-date columns-OCHLV 
        """
        self.data = data

    def calc_basic_rsrs(self, N: int, method: str, weight: pd.Series = None) -> pd.Series:
        """计算基础的RSRS

        Args:
            N (int): 计算窗口
            method (str): 使用ols或者wls模型计算
            weight (pd.Series):当方法为wls时有效. Defaults to None.
                               为None时,权重设置为令每个数据点的权重等于
                               当日成交额除以回归样本内N日的总成交额
        Returns:
            pd.Series: 基础rsrs index-date value
        """

        func: Dict = {'ols': RollingOLS, 'wls': RollingWLS}

        endog: pd.Series = self.data['high']
        exog: pd.DataFrame = self.data[['low']].copy()
        exog['const'] = 1.

        if (method == 'wls'):

            if weight is None:

                weight = self.data['volume'] / \
                    self.data['volume'].rolling(N).sum()

            mod = func[method](endog, exog, window=N, weights=weight)

        else:
            mod = func[method](endog, exog, window=N)

        self.rolling_res = mod.fit()  # 将回归结果储存在rolling_res中
        self._basic_rsrs = self.rolling_res.params['low']

        return self._basic_rsrs

    def calc_zscore_rsrs(self, N: int, M: int, method: str, weight: pd.Series = None) -> pd.DataFrame:
        """计算标准分RSRS

        Args:
            N (int): 基础RSRS的计算窗口
            M (int): 标准分的计算窗口
            method (str): 使用ols或者wls模型计算

        Returns:
            pd.DataFrame: 标准分RSRS index-date value
        """
        # 计算基础RSRS
        basic_rsrs: pd.Series = self.calc_basic_rsrs(N, method, weight)

        return (basic_rsrs - basic_rsrs.rolling(M).mean()) / basic_rsrs.rolling(M).std()

    def calc_revise_rsrs(self, N: int, M: int, method: str, weight: pd.Series = None) -> pd.Series:
        """计算修正标准分RSRS

        Args:
            N (int): 基础RSRS的计算窗口
            M (int): 标准分的计算窗口
            method (str): 使用ols或者wls模型计算

        Returns:
            pd.Series: 修正标准分RSRS index-date value
        """
        zscore_rsrs: pd.Series = self.calc_zscore_rsrs(
            N, M, method, weight)  # 计算标准分RSRS
        rsquared: pd.Series = self.rolling_res.rsquared  # 获取R方

        return zscore_rsrs * rsquared

    def calc_right_skewed_rsrs(self, N: int, M: int, method: str, weight: pd.Series = None) -> pd.Series:
        """计算右偏标准分RSRS

        Args:
            N (int): 基础RSRS的计算窗口
            M (int): 标准分的计算窗口
            method (str): 使用ols或者wls模型计算

        Returns:
            pd.Series: 右偏标准分RSRS index-date value
        """
        revise_rsrs: pd.Series = self.calc_revise_rsrs(
            N, M, method, weight)  # 计算修正标准分RSRS
        return revise_rsrs * self._basic_rsrs

    def calc_insensitivity_rsrs(self, N: int, M: int, method: str, volatility: pd.Series = None, weight: pd.Series = None) -> pd.Series:
        """计算钝化RSRS

        原理:由于R大于0小于1,当分位数越大时,震荡水平越高,此时RSRS指标将得到更大的钝化效果。
        这里使用volatility来控制R方指数部分,来达到钝化的效果
        Args:
            N (int): 基础RSRS的计算窗口
            M (int): 标准分的计算窗口
            method (str): 使用ols或者wls模型计算
            volatility (pd.Series, optional): 控制钝化效果的指标. 默认为波动率排名 研报的原始构造.

        Returns:
            pd.Series: [description]
        """
        if volatility is None:
            ret = self.data['close'].pct_change()
            ret_std = ret.rolling(N).std()
            quantile = ret_std.rolling(M).apply(
                lambda x: x.rank(pct=True)[-1], raw=False) * 2
        else:
            quantile = volatility

        zscore_rsrs: pd.Series = self.calc_zscore_rsrs(
            N, M, method, weight)  # 计算标准分RSRS
        rsquared: pd.Series = self.rolling_res.rsquared

        return zscore_rsrs * rsquared.pow(quantile)


"""
低延时性均线

from:https://www.joinquant.com/view/community/detail/f011921f2398c593eee3542a6069f61c
"""


def calc_LLT_MA(price: pd.Series, alpha: float) -> pd.Series:
    """计算低延迟趋势线

    Args:
        price (pd.Series): 价格数据. index-date values
        alpha (float): 窗口期的倒数.比如想要窗口期为5,则为1/5

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values
    """
    if not isinstance(price, pd.Series):
        raise ValueError('price必须为pd.Series')

    llt_ser: pd.Series = pd.Series(index=price.index)
    llt_ser[0], llt_ser[1] = price[0], price[1]

    for i, e in enumerate(price.values):

        if i > 1:

            v = (alpha - alpha**2 * 0.25) * e + (alpha ** 2 * 0.5) * price.iloc[i - 1] - (
                alpha - 3 * (alpha**2) * 0.25) * price.iloc[i - 2] + 2 * (
                    1 - alpha) * llt_ser.iloc[i - 1] - (1 - alpha)**2 * llt_ser.iloc[i - 2]

            llt_ser.iloc[i] = v

    return llt_ser


def calc_OLSTL(price: pd.Series, window: int) -> pd.Series:
    """
    广发提出了一种新的低延迟均线系统。
    OLSTL(Ordinary Least Square Trend Line)是基于普通最小二乘法的思想构建均线指标。
    普通最小二乘法的思想是通过对自变量和因变量序列进行拟合,找寻使所有观察值残差平方和
    最小的拟合曲线及对应参数。

    from:https://www.joinquant.com/view/community/detail/25005955f7b98b52ae99fb9fad9a6758
    Args:
        price (pd.Series): 价格数据. index-date values
        window (int): 窗口期的倒数

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values
    """
    if not isinstance(price, pd.Series):
        raise ValueError('price必须为pd.Series')

    def _func(arr: np.ndarray) -> float:

        size = len(arr)
        weights = np.arange(1, size+1) - (size + 1) / 3

        avg = weights * arr
        constant = 6 / (size * (size + 1))
        return constant * np.sum(avg)

    return price.rolling(window).apply(_func, raw=True)


def FRAMA(price: pd.Series, window: int, clip: bool = True) -> pd.Series:
    """分形自适应移动平均(FRactal Adaptive Moving Average,FRAMA)利用了投资品价格序列的分形特征

    Args:
        price (pd.Series): 价格数据. index-date values
        window (int): 时间窗口
        clip (bool, optional): 是否截断. Defaults to True.

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values
    """
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    T = int(np.ceil(window * 0.5))
    ser = price.copy()

    # 1.用窗口 W1 内的最高价和最低价计算 N1 = (最高价 – 最低价) / T
    N1 = (ser.rolling(T).max()-ser.rolling(T).min())/T

    # 2.用窗口 W2 内的最高价和最低价计算 N2 = (最高价 – 最低价) / T
    n2_df = ser.shift(T)
    N2 = (n2_df.rolling(T).max()-n2_df.rolling(T).min())/T

    # 3.用窗口 T 内的最高价和最低价计算 N3 = (最高价 – 最低价) / (2T)
    N3 = (ser.rolling(window).max() -
          ser.rolling(window).min()) / window

    # 4.计算分形维数 D = [log(N1+N2) – log(N3)] / log(2)
    D = (np.log10(N1+N2)-np.log10(N3))/np.log10(2)

    # 5.计算指数移动平均的参数 alpha = exp(-4.6(D-1))
    alpha = np.exp(-4.6*(D-1))

    # 设置上线
    if clip:
        alpha = np.clip(alpha, 0.01, 0.2)

    FRAMA = []
    idx = np.argmin(alpha)
    for row, data in enumerate(alpha):
        if row == (idx):
            FRAMA.append(ser.iloc[row])
        elif row > (idx):
            FRAMA.append(data * ser.iloc[row] +
                         (1-data)*FRAMA[-1])
        else:
            FRAMA.append(np.nan)

    FRAMA_se = pd.Series(FRAMA, index=ser.index)

    return FRAMA_se


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

    hma = talib.WMA(2 * talib.WMA(price, int(window * 0.5)) -
                    talib.WMA(price, window), int(np.sqrt(window)))

    return hma


"""
高阶距择时


from: https://www.joinquant.com/view/community/detail/e585df64077e4073ece0bcaa6b054bfa
"""


def calc_moment(price: pd.Series, cal_m_winodw: int = 20, moment: int = 5, rol_window: int = 90, alpha: Union[float, np.ndarray] = None) -> pd.DataFrame:
    """
    1. 计算每天日收益率的五阶矩,计算公式下式所示,计算数据长度为20。
    $$v_k=\frac{\sum^n_{i=1}x^k_i}{N}$$
    2. 在T日收盘后,计算出T日(含)之前的五阶矩。 
    3. 对五阶矩进行指数移动平均处理,具体计算公式如下:
    $$EMA=\sum^{120}_{i=1}\alpha*(1-\alpha)^{i-1}*momentt_{T-i+1}$$
    参数alpha取值范围为从0.05至0.5,间隔0.05,𝑚o𝑚ent代表t日的高阶矩,这样我们就得到了不同参数下的T日(含)之前的平滑五阶矩序列。 
    4. 滚动窗口样本外推。每隔90个交易日,利用 T 日之前 90 个交易日的窗口期数据进行参数确定,需要确定的参数为指数移动平均系数alpha。通过窗口期数据对不同alpha的指数移动平均得到的结果进行测试，按照切线法（详见短线择时策略研究之三《低 延迟趋势线与交易性择时》）确定 T日使得窗口期累积收益最大的指数移动平均参数 $\alpha_{max}$（该值每次可能会发生变化），得到的参数$\alpha_{max}$有效期为90天，直至下一次参 数确定前。 
    5. 按照切线法,如果T日五阶矩的 EMA$\alpha_{max}$大于 T-1 日的 EMA($\alpha_{max}$),那么T+1 日的信号为+1,T+1日看多,建仓价为 T 日收盘价;否则信号为-1,T+1日看空。 
    6. 计算过程设臵 10%止损线,如果单次择时亏损超过 10%即保持空仓位,直至择时信号变化。

    Args:
        price (pd.Series): 价格数据. index-date values
        cal_m_winodw (int, optional): 收益率阶距的计算窗口. Defaults to 20.
        moment (int, optional): 阶距的距数. Defaults to 5.
        rol_window (int, optional): 阶距的ema计算窗口数. Defaults to 90.
        alpha (Union[float,np.ndarray], optional): ema的参数. Defaults to None.

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.DataFrame: index-date columns ema参数alpha value-处理后的结果
    """
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    if isinstance(alpha, (float, int)):

        alpha = np.array([alpha])

    # 计算收益率
    pct_chg: pd.Series = price.pct_change()

    # 计算收益率阶距
    moment_ser: pd.Series = pct_chg.rolling(cal_m_winodw).apply(
        stats.moment, kwargs={'moment': moment})

    ema_momentt = pd.concat(
        (moment_ser.ewm(alpha=x, adjust=False).mean() for x in alpha), axis=1)
    ema_momentt.columns = ['{}'.format(round(i, 4)) for i in alpha]

    return ema_momentt


"""
相对强弱RPS指标

from:
    https://www.joinquant.com/view/community/detail/ddf35e24e9dbad456d3e6beaf0841262
"""

# 相对强弱指标
def calc_RPS(price: pd.Series, window: int=10, default_window: int = 250) -> pd.Series:
    """
    "强者恒强、弱者恒弱"常为市场所证实。个股或市场的强弱表现其本身就是基本面、资金面、投资者情绪等多种因素的综合作用下的体现。
    通常市场强势与否,可以用市场相对强弱 RPS 指标来表示。

    计算方法:
    RPS_1 = (当前收盘价 - min(过去250日收盘价))/(max(过去250日收盘价)-min(过去250日收盘价))

    RPS = RPS_1的10日移动平均值 

    Args:
        price (pd.Series): 价格数据. index-date values
        window (int): 时间窗口. Defaults to 10.
        default_window (int):默认的计算窗口. Defaults to 250.
    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values
    """
    if not isinstance(price, pd.Series):
        raise ValueError('price必须为pd.Series')

    size = len(price)
    limit = min(default_window, window)
    if size < limit:

        warnings.warn(
            "price长度低于最低窗口长度%s." % limit
        )

        min_periods = 0

    else:

        min_periods = None

    rps = (price - price.rolling(250, min_periods=min_periods).min()) / (
        price.rolling(250, min_periods=min_periods).max() - price.rolling(250, min_periods=min_periods).min())

    return rps.rolling(window, min_periods=min_periods).mean()

# 强弱 RPS下波动率差值
def calc_volatility_rpc(price:pd.Series,window:int,default_window:int=250)->pd.Series:
    """
    相对强弱 RPS下波动率差值
    
    1. 计算相应指数相对强弱RPS 
    2. 计算相应指数上行波动率、下行波动率,并计算二者差值 
    3. 计算当天波动率差值的移动均值天数由 RPS 值确定、RPS 值越大相就取的天数越多
    4. 观察前一天的(波动率差值的移动均值),如为正就保持持有(或卖入)、否则就保持空仓(或卖出)。

    Args:
        price (pd.Series): 价格数据. index-date values
        window (int): 时间窗口. Defaults to 10.
        default_window (int):默认的计算窗口. Defaults to 250.
    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values

    """
    rps = calc_RPS(price, window,default_window)
    pct_chg = price.pct_change()
    
    up:np.ndarray = np.where(pct_chg > 0,rps,0)
    down:np.ndarray = np.where(pct_chg <= 0,rps,0)
    
    dif = pd.Series(data=up - down,index=rps.index)
    dif = dif.rolling(window).mean()

    return dif

"""牛熊线指标

from:https://www.joinquant.com/view/community/detail/6a77f468b6f996fcd995a8d0ad8c939c
     https://www.joinquant.com/view/community/detail/d0b0406c2ad2086662de715c92d518cd
"""

# 华泰-熊牛线
def calc_ht_bull_bear(price:pd.Series,turnover:pd.Series,window:int)->pd.Series:
    """华泰熊牛熊
       使用收益率的波动率与自由流通换手率构造
    Args:
        price (pd.Series): 价格数据
        turnover (pd.Series): 自由流通换手率(也可使用普通换手率,但自由流通换手率效果更好)
        window (int): 观察窗口

    Returns:
        pd.Series
    """
    if (not isinstance(price,pd.Series)) or (not isinstance(turnover,pd.Series)):
        raise ValueError('price和turnover必须为pd.Series')
        
    pct_chg = price.pct_change()
    vol = pct_chg.rolling(window).std()
    turnover_avg = turnover.rolling(window).mean()

    return turnover_avg / turnover_avg

# 量化投资:策略于技术-熊牛线
def calc_bull_curve(price: pd.Series, alpha:float, n: int, T: int, method:str='bull') -> pd.Series:
    """
    from 《量化投资:策略于技术》丁鹏
          策略篇-量化择时-熊牛线
    Args:
        price (pd.Series): 价格数据
        n (int): 采样点数
        T (int): 采样的间隔交易日
        method (str): 计算bull curve或者bear curve. Defaults to 'bull'.
    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: [description]
    """
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    window = n * T # 时间窗口
    epsilon = stats.t.ppf(1 - alpha * 0.5, n)  # 落入牛熊价格区间的置信度为(1-alpha)
    log_ret = np.log(price / price.shift(-1))
    mu = log_ret.rolling(window).mean()
    sigma = log_ret.rolling(window).std()
    close_t = price.shift(T)

    return geometric_mrownian_motion(close_t, mu, sigma, T, epsilon)


def geometric_mrownian_motion(price: pd.Series, mu: pd.Series, sigma: pd.Series, T: int, epsilon: float, method: str = 'bull') -> float:
    """geometric_mrownian_motion过程

    Args:
        price (pd.Series): 价格数据
        mu (pd.Series): 均值
        sigma (pd.Series): 波动率
        T (int): 采样的间隔交易日
        epsilon (float): 执行区间
        method (str, optional): 计算bull curve或者bear curve. Defaults to 'bull'.

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        float: [description]
    """
    if method == 'bull':

        return price * np.exp(T * mu + np.sqrt(T) * sigma * epsilon)

    elif method == 'bear':

        return price * np.exp(T * mu - np.sqrt(T) * sigma * epsilon)

    else:

        raise ValueError('method参数仅能为bull或者bear')

"""Hurst指数

from:https://github.com/Mottl/hurst
"""


def __to_inc(x):
    incs = x[1:] - x[:-1]
    return incs


def __to_pct(x):
    pcts = x[1:] / x[:-1] - 1.
    return pcts


def __get_simplified_RS(series, kind):
    """
    Simplified version of rescaled range
    Parameters
    ----------
    series : array-like
        (Time-)series
    kind : str
        The kind of series (refer to compute_Hc docstring)
    """

    if kind == 'random_walk':
        incs = __to_inc(series)
        R = max(series) - min(series)  # range in absolute values
        S = np.std(incs, ddof=1)
    elif kind == 'price':
        pcts = __to_pct(series)
        R = max(series) / min(series) - 1.  # range in percent
        S = np.std(pcts, ddof=1)
    elif kind == 'change':
        incs = series
        _series = np.hstack([[0.], np.cumsum(incs)])
        R = max(_series) - min(_series)  # range in absolute values
        S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due the undefined R/S ratio

    return R / S


def __get_RS(series, kind):
    """
    Get rescaled range (using the range of cumulative sum
    of deviations instead of the range of a series as in the simplified version
    of R/S) from a time-series of values.
    Parameters
    ----------
    series : array-like
        (Time-)series
    kind : str
        The kind of series (refer to compute_Hc docstring)
    """

    if kind == 'random_walk':
        incs = __to_inc(series)
        mean_inc = (series[-1] - series[0]) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    elif kind == 'price':
        incs = __to_pct(series)
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    elif kind == 'change':
        incs = series
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due undefined R/S

    return R / S


def compute_Hc(series,
               kind="random_walk",
               min_window=10,
               max_window=None,
               simplified=True):
    """
    Compute H (Hurst exponent) and C according to Hurst equation:
    E(R/S) = c * T^H
    Refer to:
    https://en.wikipedia.org/wiki/Hurst_exponent
    https://en.wikipedia.org/wiki/Rescaled_range
    https://en.wikipedia.org/wiki/Random_walk
    Parameters
    ----------
    series : array-like
        (Time-)series
    kind : str
        Kind of series
        possible values are 'random_walk', 'change' and 'price':
        - 'random_walk' means that a series is a random walk with random increments;
        - 'price' means that a series is a random walk with random multipliers;
        - 'change' means that a series consists of random increments
            (thus produced random walk is a cumulative sum of increments);
    min_window : int, default 10
        the minimal window size for R/S calculation
    max_window : int, default is the length of series minus 1
        the maximal window size for R/S calculation
    simplified : bool, default True
        whether to use the simplified or the original version of R/S calculation
    Returns tuple of
        H, c and data
        where H and c — parameters or Hurst equation
        and data is a list of 2 lists: time intervals and R/S-values for correspoding time interval
        for further plotting log(data[0]) on X and log(data[1]) on Y
    """

    if len(series) < 100:
        raise ValueError("Series length must be greater or equal to 100")

    ndarray_likes = [np.ndarray]
    if "pandas.core.series" in sys.modules.keys():
        ndarray_likes.append(pd.core.series.Series)

    # convert series to numpy array if series is not numpy array or pandas Series
    if type(series) not in ndarray_likes:
        series = np.array(series)

    if "pandas.core.series" in sys.modules.keys() and type(
            series) == pd.core.series.Series:
        if series.isnull().values.any():
            raise ValueError("Series contains NaNs")
        series = series.values  # convert pandas Series to numpy array
    elif np.isnan(np.min(series)):
        raise ValueError("Series contains NaNs")

    if simplified:
        RS_func = __get_simplified_RS
    else:
        RS_func = __get_RS

    err = np.geterr()
    np.seterr(all='raise')

    max_window = max_window or len(series) - 1
    window_sizes = list(
        map(lambda x: int(10**x),
            np.arange(math.log10(min_window), math.log10(max_window), 0.25)))
    window_sizes.append(len(series))

    RS = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(series), w):
            if (start + w) > len(series):
                break
            _ = RS_func(series[start:start + w], kind)
            if _ != 0:
                rs.append(_)
        RS.append(np.mean(rs))

    A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
    np.seterr(**err)

    c = 10**c
    return H, c, [window_sizes, RS]


def random_walk(length,
                proba=0.5,
                min_lookback=1,
                max_lookback=100,
                cumprod=False):
    """
    Generates a random walk series
    Parameters
    ----------
    proba : float, default 0.5
        the probability that the next increment will follow the trend.
        Set proba > 0.5 for the persistent random walk,
        set proba <  0.5 for the antipersistent one
    min_lookback: int, default 1
    max_lookback: int, default 100
        minimum and maximum window sizes to calculate trend direction
    cumprod : bool, default False
        generate a random walk as a cumulative product instead of cumulative sum
    """

    assert (min_lookback >= 1)
    assert (max_lookback >= min_lookback)

    if max_lookback > length:
        max_lookback = length
        warnings.warn(
            "max_lookback parameter has been set to the length of the random walk series."
        )

    if not cumprod:  # ordinary increments
        series = [0.] * length  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback,
                                             min(i - 1, max_lookback) + 1)
                direction = np.sign(series[i - 1] - series[i - 1 - lookback]
                                    ) * np.sign(proba - np.random.uniform())
            series[i] = series[i - 1] + np.fabs(np.random.randn()) * direction
    else:  # percent changes
        series = [1.] * length  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback,
                                             min(i - 1, max_lookback) + 1)
                direction = np.sign(series[i - 1] / series[i - 1 - lookback] -
                                    1.) * np.sign(proba - np.random.uniform())
            series[i] = series[i - 1] * np.fabs(1 + np.random.randn() / 1000. *
                                                direction)

    return series

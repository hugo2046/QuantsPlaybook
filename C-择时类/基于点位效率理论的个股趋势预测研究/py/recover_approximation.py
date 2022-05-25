'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-25 10:50:52
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-05-25 11:27:51
Description: 高低点划分
'''
from collections import (defaultdict, namedtuple)
from typing import (List, Tuple, Dict, Union, Any)

import pandas as pd
import numpy as np
from talib import (MACD, ATR)

from sklearn.base import (BaseEstimator, TransformerMixin)
from sklearn.pipeline import Pipeline


def _approximation_method(dif: pd.Series, dea: pd.Series, atr: pd.Series,
                          rate: float, method: str) -> pd.Series:
    """生成上下行划分

    Args:
        dif (pd.Series): index-date value
        dea (pd.Series): index-date value
        atr (pd.Series): index-date value
        rate (float): index-date value
        method (str): 方法选择

    Returns:
        pd.Series: _description_
    """
    if not isinstance(method, str):

        raise ValueError('method参数只能为str')

    method = method.upper()

    func_dic = {
        'A': create_method_a,
        'B': create_method_b,
        'C': create_method_c
    }
    return func_dic[method](dif, dea, atr, rate)


def create_method_a(dif: pd.Series, dea: pd.Series, **kw) -> pd.Series:
    """方式一
    
        $$
        Dir[t]
        \begin{cases}
        \1, &如果DIF[t] \le DEA[t]\\
        -11, &如果DIF[t] \gt DEA[t]
        \end{cases}
        $$
        
    Args:
        dif (pd.Series): index-date value
        dea (pd.Series): index-date value

    Returns:
        pd.Series: index-date value-dir
    """
    return (dif -
            dea).apply(lambda x: np.where(x >= 0, 1, np.where(x < 0, -1, 0)))


def create_method_b(dif: pd.Series, dea: pd.Series, atr: pd.Series,
                    rate: float) -> pd.Series:
    """方式二
    
        $$
        Dir[t]
        \begin{cases}
        1, &如果DIF[t] - DEA[t] \ge \delta\\
        -1, &如果DIF[t] - DEA[t] \le -\delta
        \end{cases}
        $$
        其中\delta:
        $$\delta=Rate_{\delta}*ATR[t]$$
        
    Args:
        dif (pd.Series): index-date value
        dea (pd.Series): index-date value
        atr (pd.Series): index-date value
        rate (float):

    Returns:
        pd.Series: index-date values
    """

    delta: pd.Series = atr * rate

    dir_ = np.where((dif - dea - delta) >= 0, 1, 0) + np.where(
        (dif - dea + delta <= 0), -1, 0)

    dir_ser = pd.Series(dir_, index=dif.index)

    return dir_ser


def get_integral(dif: pd.Series, dea: pd.Series) -> pd.Series:
    """同号累加

    Args:
        dif (pd.Series): index-date value
        dea (pd.Series): index-date value

    Returns:
        pd.Series: index-date value
    """
    ser = pd.Series(index=dif.index)

    for i, v in enumerate(dif - dea):

        if i:

            curret_sign = np.sign(v)

            if curret_sign == previous_sign:

                ser.iloc[i] = v + previous_v

            else:

                ser.iloc[i] = 0

            previous_sign = curret_sign
            previous_v = v

        else:

            ser.iloc[i] = 0
            previous_sign = np.sign(v)
            previous_v = v
            continue

    return ser


def create_method_c(dif: pd.Series, dea: pd.Series, atr: pd.Series,
                    rate: float) -> pd.Series:
    """方法三"""

    integral = get_integral(dif, dea)

    delta = atr * rate

    dir_ser = pd.Series(index=dif.index)

    for num, (integral_v, delta) in enumerate(zip(integral, delta)):

        if num == 0:

            dir_ser.iloc[num] = 0
            previous_dir = 0
            continue

        if (integral_v >= delta) or ((previous_dir == 1) and
                                     (integral_v >= -delta)):

            dir_ser.iloc[num] = 1

        elif (integral_v <= -delta) or ((previous_dir == -1) or
                                        (integral_v <= delta)):

            dir_ser.iloc[num] = -1

        else:
            dir_ser.iloc[num] = 0

        previous_dir = dir_ser.iloc[num]

    return dir_ser


# TODO:从这里开始
# 划分上下行
class Approximation(BaseEstimator, TransformerMixin):
    '''
    用于划分上下行
    ------
    输入参数：
        price:含有CLH
        rate:方法2,3所需参数当method为A时无效
        method:A,B,C对应方法1至3,忽略大小写
        fastperiod,slowperiod,signalperiod为MACD参数
        N:为ATR参数
    '''
    def __init__(self,
                 rate: float,
                 method: str,
                 fastperiod: int = 12,
                 slowperiod: int = 26,
                 signalperiod: int = 9,
                 N: int = 100) -> None:

        self.rate = rate
        self.method = method.upper()
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod
        self.N = N

    def fit(self, X, y=None) -> pd.Series:

        return self

    def transform(self, price: pd.DataFrame, y=None) -> pd.DataFrame:

        dif, dea, histogram = MACD(price['close'],
                                   fastperiod=self.fastperiod,
                                   slowperiod=self.slowperiod,
                                   signalperiod=self.signalperiod)

        atr = ATR(price['high'], price['low'], price['close'], self.N)

        df = price[['close']].copy()

        original = _approximation_method(dif, dea, atr, self.rate, self.method)
        df['original'] = dif - dea
        df['dir'] = original
        df['dif'] = dif
        df['dea'] = dea
        df['atr'] = atr

        # 过滤前序期
        max_periods = max(self.fastperiod, self.slowperiod, self.signalperiod,
                          self.N)

        return df.iloc[max_periods:]


class Mask_dir_peak_valley(BaseEstimator, TransformerMixin):
    '''
    根据上下行方式，标记高低点
    ------
    输入参数:
        flag_df:含有上下行标记的df
        flag_col:含有标记的列,根据此列进行标记
        show_tmp:中间过程是否保存
    ------
    return 在原表上添加PEAL-阶段高点,VELLEY-阶段低点及POINT标记的合并点
    '''
    def __init__(self, flag_col: str) -> None:

        self.flag_col = flag_col

    def fit(self, X, y=None):

        return self

    def transform(self, flag_df: pd.DataFrame, y=None) -> pd.DataFrame:

        # 过滤nan值
        dropna_df = flag_df.dropna(subset=[self.flag_col]).copy()

        DROP_COL = [
            'PEAK', 'VALLEY', 'PEAK_DATE', 'VALLEY_DATE', 'POINT', 'POINT_DATE'
        ]

        SELECT_COL = [
            col for col in dropna_df.columns if col.upper() in DROP_COL
        ]

        try:
            dropna_df.drop(columns=SELECT_COL, inplace=True)
        except KeyError:
            pass

        flag_ser = dropna_df[self.flag_col]

        dropna_df['g'] = (flag_ser != flag_ser.shift(1)).cumsum()

        for k, slice_df in dropna_df.groupby('g'):

            if slice_df[self.flag_col][0] == 1:

                idx = slice_df['close'].idxmax()
                dropna_df.loc[idx, 'PEAK'] = slice_df.loc[idx, 'close']
                dropna_df.loc[idx, 'PEAK_DATE'] = idx
                dropna_df.loc[idx, 'POINT'] = dropna_df.loc[idx, 'PEAK']
                dropna_df.loc[idx, 'POINT_DATE'] = idx

            if slice_df[self.flag_col][0] == -1:

                idx = slice_df['close'].idxmin()
                dropna_df.loc[idx, 'VALLEY'] = slice_df.loc[idx, 'close']
                dropna_df.loc[idx, 'VALLEY_DATE'] = idx
                dropna_df.loc[idx, 'POINT'] = dropna_df.loc[idx, 'VALLEY']
                dropna_df.loc[idx, 'POINT_DATE'] = idx

        # 端点不标记
        idx = dropna_df.index[-1]

        dropna_df.loc[idx, SELECT_COL] = [np.nan] * len(SELECT_COL)

        # 舍弃中间过程列
        dropna_df.drop(columns=['g'], inplace=True)

        return dropna_df


# 修正上下行
class Except_dir(BaseEstimator, TransformerMixin):
    '''
    获取修正后的status值,依赖于高低点标记
    ------
    输入参数:
        flag_df:index-date,columns-close|需要修正的列(flag_col)
        flag_col:需要修正的目标列
    ------
    return 在flag_df(副本)上添加status及except的df
    '''
    def __init__(self, flag_col: str) -> None:

        self.flag_col = flag_col

    def fit(self, X, y=None):

        return self

    def transform(self, flag_df: pd.DataFrame, y=None) -> pd.DataFrame:

        flag_col = self.flag_col

        def _except_1(row: pd.Series) -> int:
            '''当t-1为1时'''
            if (row[flag_col] == 1) and (row['close'] <= row['VALLEY_CLONE']):

                return -1

            elif (row[flag_col] == -1) and (row['close'] >= row['PEAK_CLONE']):

                return -1

            else:

                return 1

        def _except_2(row: pd.Series) -> int:
            '''当t-1 为-1时'''
            if (row[flag_col] != row['pervious_dir']):

                return 1

            elif (row[flag_col] == 1) and (row['close'] >= row['PEAK_CLONE']):

                return 1

            elif (row[flag_col]
                  == -1) and (row['close'] <= row['VALLEY_CLONE']):

                return 1

            else:
                return -1

        pervious_except = 1  # 初始值

        df = flag_df.copy()  # 备份

        df['pervious_dir'] = df[flag_col].shift(1)

        df[['PEAK_CLONE',
            'VALLEY_CLONE']] = df[['PEAK', 'VALLEY']].fillna(method='ffill')

        df['excet_g'] = (df[flag_col] != df[flag_col].shift(1)).cumsum()

        dropna_df = df.dropna(subset=[flag_col])

        for idx, row in dropna_df.iterrows():

            if pervious_except == 1:

                sign = _except_1(row)
                dropna_df.loc[idx, 'except'] = sign

            else:

                sign = _except_2(row)
                dropna_df.loc[idx, 'except'] = sign

            perivous_except = sign

        df['except'] = dropna_df['except']
        df['status'] = df[flag_col] * df['except']

        # 删除中间过程
        df.drop(
            columns=['pervious_dir', 'excet_g', 'PEAK_CLONE', 'VALLEY_CLONE'],
            inplace=True)

        return df


class Mask_status_peak_valley(BaseEstimator, TransformerMixin):
    '''标记修正后的高低点'''
    def __init__(self, flag_col: str) -> None:

        self.flag_col = flag_col

    def fit(self, X, y=None):

        return self

    def transform(self, flag_df: pd.DataFrame, y=None) -> pd.DataFrame:

        drop_tmp = flag_df.dropna(subset=[self.flag_col]).copy()

        DROP_COL = [
            'PEAK', 'VALLEY', 'PEAK_DATE', 'VALLEY_DATE', 'POINT', 'POINT_DATE'
        ]

        SELECT_COL = [
            col for col in drop_tmp.columns if col.upper() in DROP_COL
        ]
        try:
            drop_tmp.drop(columns=SELECT_COL, inplace=True)
        except KeyError:
            pass

        peak_valley_dic = get_status_peak_valley(flag_df, self.flag_col)

        # 将dic转为DataFrame
        point_df: pd.DataFrame = pd.DataFrame(
            [i._asdict() for i in peak_valley_dic.status_dic.values()])

        point_df.index = point_df['point_date']
        point_df.columns = [i.upper() for i in point_df.columns]

        # 与传入表合并
        return pd.merge(drop_tmp,
                        point_df,
                        left_index=True,
                        right_index=True,
                        how='left')


# 标记修正后的高低点
class peak_valley_record():
    '''记录波峰波谷'''
    def __init__(self) -> None:

        self.status_dic: Dict = defaultdict(namedtuple)

        self.P = namedtuple(
            'P', 'peak,peak_date,valley,valley_date,point,point_date')

    def add(self,
            key: int,
            peak=None,
            peak_date=None,
            valley=None,
            valley_date=None) -> Dict:

        if peak:
            point = peak
            point_date = peak_date
        else:
            point = valley
            point_date = valley_date

        self.status_dic[key] = self.P(peak=peak,
                                      peak_date=peak_date,
                                      valley=valley,
                                      valley_date=valley_date,
                                      point=point,
                                      point_date=point_date)

    def query(self, key: Any):

        if key in self.status_dic:

            return self.status_dic[key]

        else:

            raise KeyError(f'{key}不在字典中')


def get_status_peak_valley(except_end: pd.DataFrame,
                           flag_col: str) -> peak_valley_record:
    '''
    使用status标记波段,依赖修正标记
    ------
    输入参数：
        except_trend:index-date columns-close|status
    '''

    drop_tmp = except_end.dropna(subset=[flag_col]).copy()

    drop_tmp['mark_num'] = (drop_tmp[flag_col] !=
                            drop_tmp[flag_col].shift(1)).cumsum()

    peak_valley_dic = peak_valley_record()

    for trade, row in drop_tmp.iterrows():

        status = row[flag_col]
        price = row['close']
        mark_num = row['mark_num']

        try:

            pervious_status

        except NameError:

            max_price, min_price = price, price
            max_date, min_date = trade, trade
            pervious_status = status

            continue

        if status != pervious_status:

            # 记录上一段的高低点位
            if pervious_status == 1:

                peak_valley_dic.add(mark_num - 1,
                                    peak=max_price,
                                    peak_date=max_date)

            else:

                peak_valley_dic.add(mark_num - 1,
                                    valley=min_price,
                                    valley_date=min_date)

            #
            if status == 1:

                valley_date = peak_valley_dic.query(mark_num - 1).valley_date

                slice_frame = drop_tmp.loc[valley_date:trade, 'close']

                # 更新区间最大最小值
                max_date = slice_frame.idxmax()

                max_price = slice_frame.max()

                # min_date = slice_frame.idxmin()

                # min_price = slice_frame.min()

            elif status == -1:

                peak_date = peak_valley_dic.query(mark_num - 1).peak_date

                slice_frame = drop_tmp.loc[peak_date:trade, 'close']

                # 更新区间最大最小值
                min_date = slice_frame.idxmin()

                min_price = slice_frame.min()

                # max_date = slice_frame.idxmax()

                # max_price = slice_frame.max()

            else:

                raise ValueError(f'错误的status值:{status}')

        else:

            if price >= max_price:

                max_price = price

                max_date = trade

            if price <= min_price:

                min_price = price

                min_date = trade

        pervious_status = status

    return peak_valley_dic


# 获取价格效率及时间效率
class Relative_values(BaseEstimator, TransformerMixin):
    def __init__(self, flag_col: str, is_drop: bool = True) -> None:

        self.is_drop = is_drop
        self.flag_col = flag_col

    def fit(self, X, y=None) -> None:

        return self

    def transform(self, flag_df: pd.DataFrame, y=None) -> pd.DataFrame:

        if self.is_drop:

            flag_df = flag_df.dropna(subset=[self.flag_col])

        col = ['PEAK', 'VALLEY', 'PEAK_DATE', 'VALLEY_DATE', 'POINT']

        fillna_slice = flag_df.copy()

        # 端点
        # idx = fillna_slice.index[-1]
        # fillna_slice.loc[idx,'PEAK_DATE'] = np.nan
        # fillna_slice.loc[idx,'PEAK'] = np.nan
        # fillna_slice.loc[idx,'VALLEY'] = np.nan
        # fillna_slice.loc[idx,'VALLEY_DATE'] = np.nan

        fillna_slice[col] = fillna_slice[col].fillna(method='ffill')

        fillna_slice['relative_time'] = fillna_slice.apply(calc_relative_time,
                                                           axis=1)
        fillna_slice['relative_price'] = fillna_slice.apply(
            calc_relative_price, axis=1)

        # 还原
        fillna_slice[col] = flag_df[col]
        return fillna_slice


# 时间效率
def calc_relative_time(df: pd.DataFrame) -> float:

    # 谁小离谁近
    current_dt = df.name
    if estimate_distance(df):

        return (current_dt - df['PEAK_DATE']) / (df['PEAK_DATE'] -
                                                 df['VALLEY_DATE'])

    else:

        return (current_dt - df['VALLEY_DATE']) / (df['VALLEY_DATE'] -
                                                   df['PEAK_DATE'])


# 价格效率
def calc_relative_price(df: pd.DataFrame) -> float:

    current_df = df.name
    current_price = df['close']

    if estimate_distance(df):

        return abs(current_price - df['PEAK']) / abs(df['PEAK'] - df['VALLEY'])
    else:

        return abs(current_price - df['VALLEY']) / abs(df['VALLEY'] -
                                                       df['PEAK'])


def calc_pct_distance(df: pd.DataFrame):

    current_dt = df.name
    peak_date = df['PEAK_DATE']
    valley_date = df['VALLEY_DATE']
    current_price = df['close']

    if estimate_distance(df):

        pass
    else:
        pass


def estimate_distance(df: pd.DataFrame) -> bool:
    '''
    判断当期点与前期的波峰和波谷哪更近
    ------
    return true 距离波峰近 false 距离波谷近
    '''

    current_dt = df.name
    time_h = current_dt - df['PEAK_DATE']
    time_l = current_dt - df['VALLEY_DATE']

    return time_h.days <= time_l.days
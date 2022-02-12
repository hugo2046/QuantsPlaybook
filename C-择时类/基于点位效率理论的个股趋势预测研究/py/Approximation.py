'''
Author: Hugo
Date: 2021-10-19 16:30:21
LastEditTime: 2021-10-29 13:51:38
LastEditors: Please set LastEditors
Description:
    1. 基于点位效率理论的个股趋势预测研究的点位划分-兴业证券
    2. 趋与势的量化研究-国泰证券
'''
from collections import (defaultdict, namedtuple)
from typing import (List, Tuple, Dict, Union, Callable, Any)

import datetime as dt
import numpy as np
import pandas as pd
import talib

from sklearn.base import (BaseEstimator, TransformerMixin)
from sklearn.pipeline import Pipeline

############################# 兴业研报部分 #############################


def _approximation_method(dif: pd.Series,
                          dea: pd.Series,
                          atr: pd.Series,
                          rate: float,
                          method: str,
                          window: int = 12) -> pd.Series:
    '''划分上下行方式
    ------
    输入参数：
        dif/dea:MACD参数
        atr:ATR
        rate:阈值使用的比率
        method:a,b,c三种划分方式
        window:仅对方法c有效
    '''
    if not isinstance(method, str):

        raise ValueError('method参数只能为str')

    method = method.upper()

    if method == 'A':

        return dif - dea

    elif method == 'B':

        return dif - dea - atr * rate

    elif method == 'C':

        tmp: np.array = np.vstack([dif.values, dea.values]).T

        # 判断是否同号
        cond = np.apply_along_axis(lambda x: estimate_sign(x[0], x[1]), 1, tmp)

        # 同号差异值累加
        v_diff = dif - dea

        # 设置累加窗口期 如果直接cumsum在时序上容易出现"端点效应",即新的数据会导致模型历史状态发生变化
        # 在对沪深300全样本划分时在2008年底部没有识别到低点,需要将window改为>=60以上的周期才行,但如果这样
        # 做的话前面图4,图8,图9基本上就对不上了
        intergral = (cond * v_diff).rolling(window).sum()
        intergral = intergral * np.where(v_diff == 0, 0, 1)

        return intergral + atr * rate

    else:

        raise ValueError('method参数只能为A,B,C')


# 判断是否同号
def estimate_sign(a: float, b: float) -> bool:
    '''判断数字是否同号'''

    return np.signbit(a) == np.signbit(b)


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

        dif, dea, histogram = talib.MACD(price['close'],
                                         fastperiod=self.fastperiod,
                                         slowperiod=self.slowperiod,
                                         signalperiod=self.signalperiod)

        atr = talib.ATR(price['high'], price['low'], price['close'], self.N)

        df = price[['close']].copy()

        original = _approximation_method(dif, dea, atr, self.rate, self.method)
        df['original'] = original
        df['dir'] = np.sign(original)
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


#############################  国泰研报部分 #############################


class Normalize_Trend(object):
    '''
    标准化价格位移
    
    注意:位移向量比状态变化向量多一个初始单元0
    '''
    def __init__(self, close_ser: pd.Series) -> None:

        if not isinstance(close_ser, pd.Series):

            raise ValueError('输入参数类型必须为pd.Series')

        self.close_ser = close_ser

    def normalize_monotone(self) -> pd.Series:
        '''单调性标准化'''

        sign = self.close_ser.pct_change().apply(np.sign)
        sign = sign.cumsum().fillna(0)

        return sign

    def normalize_movingaverage(self, window: int = 5) -> pd.Series:
        '''5周期均线的标准化'''

        close_ser = self.close_ser
        size = len(close_ser)

        if size < window:

            raise ValueError('输入数据长度小于窗口期')

        ma = close_ser.rolling(window).mean()
        sign = (close_ser - ma).apply(np.sign).iloc[window - 2:]
        sign = sign.cumsum().fillna(0)

        return sign

    def normalize_compound(self, window: int = 5):

        close_ser = self.close_ser

        size = len(close_ser)

        if size < window:

            raise ValueError('输入数据长度小于窗口期')

        sign_monotone = close_ser.pct_change().apply(np.sign)

        ma = close_ser.rolling(window).mean()
        sign_ma = (close_ser - ma).apply(np.sign)

        # auth:@jqz1226
        # 可以按照4种情形分别分析：
        # 1. 前一个交易日收盘价位于均线之下，当前收盘价站上均线，状态记为1；分析：当前sign_ma = 1，
        # 收盘价能从均线下跃到均线上，必然是由于价格上涨，故sign_monotone = 1, 于是 (1+1)/2 = 1
        # 2. 前一个交易日收盘价位于均线之上，当前收盘价跌破均线，状态记为-1；分析：当前sign_ma=-1，
        # 收盘价能从均线上掉到均线下，必然是由于价格下跌，故sign_monotone = -1, 于是((-1)+(-1))/2 = -1
        # 3. 3a) 前一个交易日收盘价位于均线之上，当前收盘价位于均线之上，当前收盘价大于或等于前一个交易日收盘价，状态记为1；
        # 分析：当前sign_ma = 1，收盘价上升，sign_monotone = 1, 于是 (1+1)/2 = 1
        # 3b) 前一个交易日收盘价位于均线之上，当前收盘价位于均线之上，当前收盘价小于前一个交易日收盘价，状态记为0；
        # 分析：当前sign_ma = 1，收盘价下降，sign_monotone = -1, 于是 ((1)+(-1))/2 = 0
        # 4. 4a) 前一个交易日收盘价位于均线之下，当前收盘价位于均线之下，当前收盘价大于前一个交易日收盘价，状态记为0，
        # 分析：当前sign_ma = -1，收盘价上升，sign_monotone = 1, 于是 (-1+1)/2 = 0
        # 4b) 前一个交易日收盘价位于均线之下，当前收盘价位于均线之下，当前收盘价小于或等于前一个交易日收盘价，状态记为-1。
        # 分析：当前sign_ma = -1，收盘价下降，sign_monotone = -1, 于是 ((-1)+(-1))/2 = -1

        sign_compound = (sign_monotone + sign_ma) / 2  # 简单平均
        sign_compound = sign_compound.iloc[window - 2:].cumsum().fillna(0)

        return sign_compound


class Tren_Score(object):
    '''
    根据标准化后的价格数据计算趋势得分
    ------
    输入参数：
        normalize_trend_ser:pd.Series index-date values-标准化后的价格数据

    方法：
        评分方法均有两种计算模式区别是划分波段的方法不同
        分别是opposite/absolute 即【相对波段划分】和【绝对波段划分】

        calc_trend_score:计算“趋势”得分
            score Dict
                - trend_score 势得分
                - act_score 趋得分
            - point_frame Dict 标记表格
            - point_mask Dict 标记点
        calc_absolute_score:计算混合模式得分
    '''
    def __init__(self, normalize_trend_ser: pd.Series) -> None:

        if not isinstance(normalize_trend_ser, pd.Series):

            raise ValueError('输入参数类型必须为pd.Series')

        self.normalize_trend_ser = normalize_trend_ser

        # 储存标记点表格
        self.point_frame: Dict[pd.DataFrame] = defaultdict(pd.DataFrame)
        self.score_record = namedtuple('ScoreRecord', 'trend_score,act_score')
        self.score: Dict = defaultdict(namedtuple)

        # 储存标记点标记
        self.point_mask: Dict[List] = defaultdict(list)

        self.func_dic: Dict = {
            'opposite': self._get_opposite_piont,
            'absolute': self._get_absolute_point
        }

    def calc_trend_score(self, method: str) -> float:
        '''势'''

        func: Callable = self.func_dic[method]

        # 趋势极值点得标记
        cond: pd.Series = func()

        # 势得分
        trend_score = np.square(self.normalize_trend_ser[cond].diff()).sum()
        # 趋得分
        act_score = self.normalize_trend_ser.diff().sum()

        self.score[method] = self.score_record(trend_score=trend_score,
                                               act_score=act_score)

        self.point_frame[method] = self.normalize_trend_ser[cond]

        self.point_mask[method] = cond

    def calc_absolute_score(self) -> float:
        '''势的终极定义'''

        opposite = self.calc_trend_score('opposite')
        absolute = self.calc_trend_score('absolute')

        N = len(self.normalize_trend_ser)

        return max(opposite, absolute) / (N * (3 / 2))

    def _get_opposite_piont(self) -> List:
        '''
        获取相对拐点的位置
        ------
        return np.array([True,..False,...True])
            True表示为拐点，False表示不是
        '''
        ser = self.normalize_trend_ser
        flag_ser = pd.Series(index=ser.index, dtype=ser.index.dtype)

        dif = ser.diff().fillna(method='bfill')

        for idx, i in dif.items():

            try:
                previous_i
            except NameError:

                previous_idx = idx
                previous_i = i
                flag_ser[idx] = True
                continue

            if i != previous_i:

                flag_ser[previous_idx] = True
            else:
                flag_ser[previous_idx] = False

            previous_idx = idx
            previous_i = i

        flag_ser.iloc[0] = True
        flag_ser.iloc[-1] = True

        # 拐点索引

        return flag_ser.values.tolist()

    def _get_absolute_point(self) -> List:
        '''
        获取绝对拐点的位置
        ------
        return np.array([True,..False,...True])
            True表示为拐点，False表示不是
        '''
        arr = self.normalize_trend_ser.values
        size = len(arr)

        # TODO:不知道我是不是没理解研报算法
        # 如果使用下面算法找最大最小 在[0,-1,-1,0,1,0,-1,-1,-2]这种情况下
        # 最大值会被标记在下标为8的元素上

        # distances = np.abs(arr.reshape(-1, 1) - np.tile(arr, (size, 1)))

        # d_arr = np.tril(distances)[:, 0]
        # # 获取最大/小值
        # ind_max = np.argmax(d_arr)
        # ind_min = np.argmin(d_arr)

        # # 最大/小值索引下标
        # idx_max = np.argwhere(d_arr == ind_max).reshape(1, -1)[0]
        # idx_min = np.argwhere(d_arr == ind_min).reshape(1, -1)[0]

        ind_max = np.max(arr)
        ind_min = np.min(arr)

        idx_max = np.argwhere(arr == ind_max).reshape(1, -1)[0]
        idx_min = np.argwhere(arr == ind_min).reshape(1, -1)[0]
        point = np.append(idx_min, idx_max)
        point = np.append(point, [0, size - 1])
        point = np.unique(point)
        cond = [True if i in point else False for i in range(size)]

        return cond
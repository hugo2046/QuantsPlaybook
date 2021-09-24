# 常用计算
import talib
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import svm   

from scipy import stats 
import scipy.fftpack as fftpack # 希尔伯特变换

import pywt # 小波分析
from typing import List

# 时间操作
import datetime as dt
from dateutil.parser import parse

# 数据获取
from jqdata import *
import tushare as ts

# 研究文档数据读取
from six import BytesIO  # 读取研究环境文件用

enable_profile()  # 开启性能分析

########################  初始化  #############################


def initialize(context):
    
    set_params()
    set_variables()
    set_backtest()  # 设置回测条件
   


# 设置策参数
def set_params():
    
   
   g.M = 5 # 计算指标的窗口
   g.N = 50 # 滚动学习计算窗口
   g.index_code = '000016.XSHG'
   g.etf_code = '000016.XSHG'#'510300.XSHG'
   
   
# 中间变量
def set_variables():
    pass


# 设置回测条件
def set_backtest():

    set_option("avoid_future_data", True)  # 避免未来数据
    set_option('use_real_price', True)  #用真实价格交易
    set_benchmark('000300.XSHG')  # 设置基准收益
    log.set_level('order', 'debug')


# 每日盘前运行
def before_trading_start(context):

    # 设置手续费
    set_slip_fee(context)


# 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 根据不同的时间段设置手续费
    dt = context.current_dt

    if dt > datetime.datetime(2013, 1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    elif dt > datetime.datetime(2011, 1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))

    elif dt > datetime.datetime(2009, 1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))
        
        
        
################################################## 信号获取 #########################################################
# 信号去噪
class DenoisingThreshold(object):
    '''
    获取小波去噪的阈值
    1. CalSqtwolog 固定阈值准则(sqtwolog)
    2. CalRigrsure 无偏风险估计准则（rigrsure）
    3. CalMinmaxi 极小极大准则（ minimaxi）
    4. CalHeursure
    
    参考：https://wenku.baidu.com/view/63d62a818762caaedd33d463.html
    
    对股票价格等数据而言，其信号频率较少地与噪声重叠因此可以选用sqtwolog和heursure准则，使去噪效果更明显。 
    但对收益率这样的高频数据，尽量采用保守的 rigrsure 或 minimaxi 准则来确定阈值，以保留较多的信号。
    '''
    

    def __init__(self,signal: np.array):

        self.signal = signal

        self.N = len(signal)

    # 固定阈值准则(sqtwolog)
    @property
    def CalSqtwolog(self) -> float:

        return np.sqrt(2 * np.log(self.N))

   
    # 无偏风险估计准则（rigrsure）
    @property
    def CalRigrsure(self)->float:

        N = self.N
        signal = np.abs(self.signal)
        signal = np.sort(signal)
        signal = np.power(signal, 2)

        risk_j = np.zeros(N)

        for j in range(N):

            if j == 0:
                risk_j[j] = 1 + signal[N - 1]
            else:
                risk_j[j] = (N - 2 * j + (N - j) *
                             (signal[N - j]) + np.sum(signal[:j])) / N

        k = risk_j.argmin()

        return np.sqrt(signal[k])

    # 极小极大准则（ minimaxi）
    @property
    def CalMinmaxi(self)->float:
        
        if self.N > 32:
            # N>32 可以使用minmaxi阈值 反之则为0
            return 0.3936 + 0.1829 * (np.log(self.N) / np.log(2))
        
        else:
            
            return 0

    @property
    def GetCrit(self)->float:

        return np.sqrt(np.power(np.log(self.N) / np.log(2), 3) * 1 / self.N)

    @property
    def GetEta(self)->float:

        return (np.sum(np.abs(self.signal)**2) - self.N) / self.N
    
    #混合准则（heursure）
    @property
    def CalHeursure(self):

        if self.GetCrit > self.GetEta:

            #print('推荐使用sqtwolog阈值')
            return self.CalSqtwolog
            
        else:

            #print('推荐使用 Min(sqtwolog阈值,rigrsure阈值)')
            return min(self.CalRigrsure,self.CalSqtwolog)
            
class wavelet_svm_model(object):
    '''对数据进行建模预测
    --------------------
    输入参数：

        data:必须包含OHLC money及预测字段Y(ovo标记) 其余字段为训练数据
        M:train数据的滚动计算窗口
        window:滚动窗口 即T至T-window日 预测T-1至T-window日数据 预测T日数据
        wavelet\wavelet_mode:同pywt.wavedec的参数
        filter_num:需要过滤小波的细节组 比如(3,4)对三至四组进行过滤 为空则是1-4组全过滤 
        whether_wave_process:是否使用小波处理
    --------------------
    方法：
        wave_process:过滤阈值 采用 混合准则（heursure）
        preprocess:生成训练用字段
        rolling_svm:使用svm滚动训练
    '''

    def __init__(self,
                 Data: pd.DataFrame,
                 M: int,
                 window: int,
                 wavelet: str,
                 wavelet_mode: str,
                 filter_num=None,
                 whether_wave_process: bool = False):

        self.Data = Data
        self.__M = M
        self.__window = window
        self.__wavelet = wavelet
        self.__wavelet_mode = wavelet_mode
        self.__filter_num = filter_num
        self.__whether_wave_process = whether_wave_process

        self.__train_col = [col for col in self.Data.columns if col != 'Y'
                           ]  # 训练的字段

        self.train_df = pd.DataFrame()  # 储存训练数据
        self.predict_df = Data[['Y']].copy()  # 储存预测数据及真实Y

    def wave_process(self):
        '''对数据进行小波处理(可选)'''
        
        if self.__filter_num:
            
            a = self.__filter_num[0]
            b = self.__filter_num[1]
            
            #self.__filter_num = range(a,b + 1)
            
        else:
            a = 1
            b = 5
            #self.__filter_num = range(1,5)

    
        Data = self.Data.copy()  # 复制
        
        for col in self.__train_col:

            #res1 = pywt.wavedec(
            #    data[col].values, wavelet=self.__wavelet, mode=self.__wavelet_mode, level=4)

            #for j in self.__filter_num:

            #    threshold = DenoisingThreshold(res1[j]).CalHeursure
            #    res1[j] = pywt.threshold(res1[j], threshold, 'soft')
            
            denosed_ser = wave_transform(Data[col], 
                           wavelet=self.__wavelet, 
                           wavelet_mode=self.__wavelet_mode, 
                           level=4,
                           n=a,
                           m=b)
            
            Data[col] = denosed_ser

        self.train_df = Data

    def preprocess(self):
        '''生成相应的特征'''

        if self.__whether_wave_process:

            self.wave_process()  # 小波处理

            Data = self.train_df

        else:

            Data = self.Data.copy()

        Data['近M日最高价'] = Data['high'].rolling(self.__M).max()
        Data['近M日最低价'] = Data['low'].rolling(self.__M).min()
        Data['成交额占比'] = Data['money'] / Data['money'].rolling(self.__M).sum()
        Data['近M日涨跌幅'] = Data['close'].pct_change(self.__M)
        Data['近M日均价'] = Data['close'].rolling(self.__M).mean()
        
        # 上面新增了需要训练用的字段 这里更新字段
        self.__train_col = [col for col in Data.columns if col not in  self.__train_col + ['Y']]
        self.train_df = Data[self.__train_col]
        self.train_df = self.train_df.iloc[self.__M:]
    
    
    def standardization(self):
        '''对所有特征进行标准化处理'''

        Data = preprocessing.scale(self.train_df[self.__train_col])
        Data = pd.DataFrame(
            Data, index=self.train_df.index, columns=self.__train_col)
        Data['Y'] = self.predict_df['Y']
        self.train_df = Data

    def rolling_svm(self):
        
        '''利用SVM模型进行建模预测'''
        predict_ser = rolling_apply(self.train_df, self.model_fit,
                                    self.__window)
        
        self.predict_df['predict'] = predict_ser
        
        self.predict_df = self.predict_df.iloc[self.__window + self.__M:]

    def model_fit(self, df: pd.DataFrame) -> pd.Series:
        
        idx = df.index[-1]

        train_x = df[self.__train_col].iloc[:-1]
        train_y = df['Y'].shift(-1).iloc[:-1]  # 对需要预测的y进行滞后一期处理

        test_x = df[self.__train_col].iloc[-1:]

        model = svm.SVC(gamma=0.001)

        model.fit(train_x, train_y)

        return pd.Series(model.predict(test_x), index=[idx])
        
# 定义rolling_apply理论上应该比for循环快
# pandas.rolling.apply不支持多列
def rolling_apply(df, func, win_size) -> pd.Series:

    iidx = np.arange(len(df))

    shape = (iidx.size - win_size + 1, win_size)

    strides = (iidx.strides[0], iidx.strides[0])

    res = np.lib.stride_tricks.as_strided(
        iidx, shape=shape, strides=strides, writeable=True)

    # 这里注意func返回的需要为df或者ser
    return pd.concat((func(df.iloc[r]) for r in res), axis=0)  # concat可能会有点慢
    
# 小波变换
def wave_transform(data_ser:pd.Series,wavelet:str,wavelet_mode:str,level:int,n:int,m:int)->pd.Series:
    '''
    参数：
        data_ser:pd.Series
        wavelet\wavelet_mode\level：同pywt.wavedec
        n,m:需要过了的层级范围
    '''
    res1 = pywt.wavedec(
                    data_ser.values, wavelet=wavelet, mode=wavelet_mode, level=level)

    for j in range(n,m+1):

        threshold = DenoisingThreshold(res1[j]).CalHeursure
        res1[j] = pywt.threshold(res1[j], threshold, 'soft')
    
    # 有时候重构后的长度不等 
    n = len(data_ser)
    
    return pd.Series(pywt.waverec(res1, wavelet)[-n:],index=data_ser.index)
    
    
################################### 数据获取 ####################################################
def GetSingal(context)->int:
    
    price_df = attribute_history(g.index_code, count=g.M + g.N + 25, unit='1d',
            fields=['open', 'close', 'high', 'low', 'pre_close', 'money'])
            
    price_df['pct_chg'] = price_df['close'] / price_df['pre_close'] - 1

    
    price_df['MACD'] = talib.MACD(price_df['close'],fastperiod=6,slowperiod=12,signalperiod=9)[0]
    price_df['RSI'] = talib.RSI(price_df['close'],timeperiod=14)        
    # 标记
    price_df['Y'] = np.sign(price_df['pct_chg'])
    
    price_df = price_df.dropna()

    # 对昨日收盘价去噪
    wsm = wavelet_svm_model(price_df,g.M,g.N,'db4','sym',True)
    # 计算训练字段
    wsm.preprocess()
    # 标准化
    wsm.standardization()
    # 滚动训练
    wsm.rolling_svm()
            
    
    return wsm.predict_df['predict'].iloc[-1] # 1为持仓 其他空仓
    
#################################  交易  ############################################

def handle_data(context,data):
    
    if GetSingal(context) == 1:
        
        BuyStock(context)
        
    else:
        
        SellStock(context)


#################################  下单  ############################################        
def BuyStock(context):
    
    
    if g.etf_code in context.portfolio.long_positions:
        
        log.info('继续持有%s'%get_security_info(g.etf_code).display_name)
        
    else:
        
        log.info('买入%s'%get_security_info(g.etf_code).display_name)
        order_target_value(g.etf_code,context.portfolio.total_value)
        
def SellStock(context):
    
    log.info('卖出%s'%get_security_info(g.etf_code).display_name)
    for hold in context.portfolio.long_positions:
        
        order_target(hold,0)
        
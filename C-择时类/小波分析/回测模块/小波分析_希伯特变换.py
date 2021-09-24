# 常用计算
import talib
import pandas as pd
import numpy as np

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
    
   g.N = 20 # 计算窗口
   g.index_code = '000300.XSHG'
   g.etf_code = '510300.XSHG'
   
   
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
    
def GetSingal(context)->int:
    
    price_df = history(count=g.N + 1, unit='1d', field='close', security_list=g.index_code)
    
    # 对昨日收盘价去噪
    denoised_price = wave_transform(price_df[g.index_code],'db4','sym',4,1,4)
    
    diff_price = denoised_price.diff()
    diff_price = diff_price.dropna()
    
    # 希尔伯特周期 滚动防止 前视偏差 
    hilbert = fftpack.hilbert(diff_price)
    
    return np.sign(hilbert)[-1] # 1为持仓 其他空仓
    
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
        
'''
Author: Hugo
Date: 2021-07-15 08:27:35
LastEditTime: 2021-11-30 14:44:35
LastEditors: Hugo
Description: In User Settings Edit
'''
from jqdata import *
import pandas as pd
import datetime as dt
from jqdata import *
from typing import (Tuple, List, Union, Callable,Dict,Any)

# 筛选股票池
class Filter_Stocks(object):
    '''
    获取某日的成分股股票
    1. 过滤st
    2. 过滤上市不足N个月
    3. 过滤当月交易不超过N日的股票
    ---------------
    输入参数：
        index_symbol:指数代码,A等于全市场,
        watch_date:日期
    '''
    
    def __init__(self,symbol:str,watch_date:str)->None:
        
        if isinstance(watch_date,str):
            
            self.watch_date = pd.to_datetime(watch_date).date()
            
        else:
            
            self.watch_date = watch_date
            
        self.symbol = symbol
        self.get_index_component_stocks()
        
    def get_index_component_stocks(self)->list:
        
        '''获取指数成分股'''
        
        if self.symbol == 'A':
            
            wd:pd.DataFrame = get_all_securities(types=['stock'],date=self.watch_date)
            self.securities:List = wd.query('end_date != "2200-01-01"').index.tolist()
        else:
            
            self.securities:List = get_index_stocks(self.symbol,self.watch_date)
    
    def filter_paused(self,paused_N:int=1,threshold:int=None)->list:
        
        '''过滤停牌股
        -----
        输入:
            paused_N:默认为1即查询当日不停牌
            threshold:在过paused_N日内停牌数量小于threshold
        '''
        
        if (threshold is not None) and (threshold > paused_N):
            raise ValueError(f'参数threshold天数不能大于paused_N天数')
            
        
        paused = get_price(self.securities,end_date=self.watch_date,count=paused_N,fields='paused',panel=False)
        paused = paused.pivot(index='time',columns='code')['paused']
        
        # 如果threhold不为None 获取过去paused_N内停牌数少于threshodl天数的股票
        if threshold:
            
            sum_paused_day = paused.sum()
            self.securities = sum_paused_day[sum_paused_day < threshold].index.tolist()
        
        else:
            
            paused_ser = paused.iloc[-1]
            self.securities = paused_ser[paused_ser == 0].index.tolist()
    
    def filter_st(self)->list:
        
        '''过滤ST'''
              
        extras_ser = get_extras('is_st',self.securities,end_date=self.watch_date,count=1).iloc[-1]
        
        self.securities = extras_ser[extras_ser == False].index.tolist()
    
    def filter_ipodate(self,threshold:int=180)->list:
        
        '''
        过滤上市天数不足以threshold天的股票
        -----
        输入：
            threhold:默认为180日
        '''
        
        def _check_ipodate(code:str,watch_date:dt.date)->bool:
            
            code_info = get_security_info(code)
            
            if (code_info is not None) and ((watch_date - code_info.start_date).days > threshold):
                
                return True
            
            else:
                
                return False

        self.securities = [code for code in self.securities if _check_ipodate(code,self.watch_date)]
    
    def filter_industry(self,industry:Union[List,str],level:str='sw_l1',method:str='industry_name')->list:
        '''过略行业'''
        ind = get_stock_ind(self.securities,self.watch_date,level,method)
        target = ind.to_frame('industry').query('industry != @industry')
        self.securities = target.index.tolist()
        
def get_stock_ind(securities:list,watch_date:str,level:str='sw_l1',method:str='industry_code')->pd.Series:
    
    '''
    获取行业
    --------
        securities:股票列表
        watch_date:查询日期
        level:查询股票所属行业级别
        method:返回行业名称or代码
    '''
    
    indusrty_dict = get_industry(securities, watch_date)

    indusrty_ser = pd.Series({k: v.get(level, {method: np.nan})[
                             method] for k, v in indusrty_dict.items()})
    
    indusrty_ser.name = method.upper()
    
    return indusrty_ser
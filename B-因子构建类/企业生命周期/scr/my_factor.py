'''
Author: Hugo
Date: 2022-04-18 16:48:40
LastEditTime: 2022-04-26 10:41:06
LastEditors: Please set LastEditors
Description: 
'''
from typing import (List, Tuple, Dict, Callable, Union)

import pandas as pd
import numpy as np
from jqfactor import Factor


def calc_ols_growth(arr: np.ndarray) -> float:
    """N年复合增速-回归法

    过去N年的财务数据，关于[0,1,2,3,...,N]回归的斜率系数，然后再除以过去N年均值的绝对值
    """
    x = np.arange(len(arr))
    A = np.vstack([x, np.ones(len(x))]).T
    beta = np.linalg.lstsq(A, arr, rcond=-1)[0][0]

    basic = np.abs(arr.mean())
    if basic == 0:
        return np.nan
    return beta / basic


def calc_growth(arr: np.ndarray) -> float:
    """N年复合增速

    """
    n = len(arr)

    return (arr[-1] / np.abs(arr[0]))**(1 / n) - 1


def StandardScaler(ser: pd.Series) -> pd.Series:
    '''z-score'''
    return (ser - ser.mean()) / ser.std()


class quadrant(Factor):
    """划分因子象限

    Parameters
    ----------
    method : ols-营收增长率-回归法
             yoy-营业收入同比增长率-年度
             growth-营收N年复合增长率
    is_scaler:是否归一化,默认为False 不归一化
              当method为yoy时 必须归一化
           
    Returns
    -------
    pd.Series
        index-code 
        values- 1:成长;2:导入;3.衰退;4.成熟
    """
    name = 'quadrant'
    max_window = 1
    method = 'ols'
    is_scaler = False  # 是否归一化
    out_put = 'quadrant'
    dependencies = [
        'roe_ttm', 'operating_revenue_y', 'operating_revenue_y1',
        'operating_revenue_y2', 'inc_revenue_year_on_year'
    ]

    def calc(self, data: Dict) -> pd.Series:

        roe_ttm: pd.DataFrame = data['roe_ttm'].iloc[-1]
        scaler_roe = (roe_ttm - roe_ttm.mean()) / roe_ttm.std()
        cols = [
            'operating_revenue_y2', 'operating_revenue_y1',
            'operating_revenue_y'
        ]
        income_frame = pd.concat((data[col].iloc[-1] for col in cols), axis=1)

        method = self.method.lower()
        if method == 'ols':
            # 三年复合增长-回归法
            income_slope = income_frame.fillna(0).apply(calc_ols_growth,
                                                        axis=1)
        elif method == 'growth':
            # 三年复合增长
            income_slope = income_frame.fillna(0).apply(calc_growth, axis=1)
        elif method == 'yoy':
            # 营业收入同比
            income_slope = data['inc_revenue_year_on_year'].iloc[-1]

        if any([self.is_scaler, method == 'yoy']):
            income_slope = StandardScaler(income_slope)

        income_yoy_mean = income_slope.mean()
        roe_mean = scaler_roe.mean()

        ser = pd.Series(index=roe_ttm.index.tolist())
        dichotomy = pd.DataFrame(index=roe_ttm.index.tolist(),
                                 columns=['roe端', '增长端'])

        ser[(income_slope > income_yoy_mean)
            & (scaler_roe > roe_mean)] = 1  # 成长
        ser[(income_slope > income_yoy_mean)
            & (scaler_roe <= roe_mean)] = 2  # 导入
        ser[(income_slope <= income_yoy_mean)
            & (scaler_roe <= roe_mean)] = 3  # 衰退
        ser[(income_slope <= income_yoy_mean)
            & (scaler_roe > roe_mean)] = 4  # 成熟

        dichotomy.loc[income_slope > income_yoy_mean, '增长端'] = 1  # 高增长端
        dichotomy.loc[income_slope <= income_yoy_mean, '增长端'] = 0  # 低增长端

        dichotomy.loc[scaler_roe > roe_mean, 'roe端'] = 1  # 高roe端
        dichotomy.loc[scaler_roe <= roe_mean, 'roe端'] = 0  # 低roe端

        self.dichotomy = dichotomy

        return ser


'''低ROE端--量价因子
低ROE端股票基本面支撑较弱，炒作风气偏重
其整体收益波动也更大，通常也更易受到技术投资者、短线投机者的关注。因此，我们认为这一类股票更适用量价指标
'''


class VolAvg(Factor):
    '''过去 20 天日均成交量 / 过去 240 天日均成交量'''

    import warnings
    warnings.filterwarnings("ignore")

    volD1 = 20
    volD2 = 240
    max_window = np.max([volD1, volD2])
    dependencies = ['volume']
    name = f'VolAvg_{volD1}D_{volD2}D'

    def calc(self, data) -> pd.Series:

        volume: pd.DataFrame = data['volume']

        return volume.iloc[-self.volD1:].mean() / volume.mean()


class VolCV(Factor):
    '''过去20天日成交量的标准差 / 过去20天日均成交量'''
    import warnings
    warnings.filterwarnings("ignore")

    max_window = 20
    dependencies = ['volume']
    name = f'VolCV_{max_window}D'

    def calc(self, data) -> pd.Series:

        volume: pd.DataFrame = data['volume']
        return volume.std() / volume.mean()


class RealizedSkewness(Factor):
    '''过去 240 天日收益率数据计算的偏度'''

    import warnings
    warnings.filterwarnings("ignore")
    D = 240
    max_window = D + 1
    dependencies = ['close']

    name = f'RealizedSkewness_{D}D'

    def calc(self, data) -> pd.Series:

        return data['close'].pct_change().iloc[1:].skew()


class ILLIQ(Factor):
    '''过去 20 天 AVERAGE(日涨跌幅绝对值 / 日成交金额)'''
    import warnings
    warnings.filterwarnings("ignore")
    D = 20
    max_window = D + 1
    name = f'ILLIQ_{D}D'
    dependencies = ['close', 'money']

    def calc(self, data) -> pd.DataFrame:

        pct_chg = data['close'].pct_change().abs().iloc[1:]
        money = data['money'].iloc[1:]

        return (pct_chg / money).mean()


'''高ROE端--一致性预期因子
高ROE端股票有较好的基本面支撑，受到价值投资者和专业的机构投资者更多的青睐。
分析师覆盖率也就越高,分析师预期边际变化因子

'''


class Operatingprofit_FY1(Factor):
    '''一致预期营业利润相对于3个月前的变化'''
    max_window = 20  # 时间长了超限....
    dependencies = ['short_term_predicted_earnings_growth']
    name = f'Operatingprofit_FY1_R{max_window}D'

    def calc(self, data) -> pd.Series:

        earnings = data['short_term_predicted_earnings_growth']
        self.frame = earnings
        return earnings.iloc[-1] / earnings.iloc[0] - 1


'''低增长端--价值稳定因子
当企业发展进入低增长的稳定期，细分市场的格局相对稳定。此时标的股票买的是否“便 宜”，
股票价格是否值得购买则变得更为重要这一阶段的股票更适用于使用 考虑“性价比”的价值稳定指标
'''


class BP_LR(Factor):

    name = 'BP_LR'
    max_window = 1
    dependencies = ['book_to_price_ratio']

    def calc(self, data) -> pd.Series:

        return data['book_to_price_ratio'].iloc[-1]


class EP_Fwd12M(Factor):

    name = 'EP_Fwd12M'
    max_window = 1
    dependencies = ['predicted_earnings_to_price_ratio']

    def calc(self, data) -> pd.Series:

        return data['predicted_earnings_to_price_ratio'].iloc[-1]


class Sales2EV(Factor):

    name = 'Sales2EV'
    max_window = 1
    dependencies = [
        'operating_revenue', 'operating_revenue_1', 'operating_revenue_2',
        'operating_revenue_3', 'market_cap', 'total_non_current_liability',
        'cash_equivalents'
    ]

    def calc(self, data) -> pd.DataFrame:

        sales_ttm = pd.concat((data[col].iloc[-1]
                               for col in self.dependencies[:4]),
                              axis=1).mean(axis=1)
        ev = (data['market_cap'].fillna(0) +
              data['total_non_current_liability'].fillna(0) +
              data['cash_equivalents'].fillna(0)).iloc[-1]
        return sales_ttm / ev


'''高增长端--成长质量
本身具有较高的成长性这一阶段股票的成长质量显得尤为重要
'''


class Gross_profit_margin_chg(Factor):
    '''销售毛利率（毛利/营业收入）同比变化'''
    name = 'Gross_profit_margin_chg'
    max_window = 1
    dependencies = ['gross_profit_margin', 'gross_profit_margin_1']

    def calc(self, data) -> pd.DataFrame:

        pct_chg = data['gross_profit_margin'] / \
            data['gross_profit_margin_1'] - 1

        return pct_chg.iloc[-1]


class Netprofit_chg(Factor):
    '''近半年利润增速变化'''
    name = 'Netprofit_chg'
    max_window = 1
    dependencies = [
        'net_profit', 'net_profit_1', 'net_profit_2', 'net_profit_3'
    ]

    def calc(self, data) -> pd.Series:

        a = pd.concat((data[col].iloc[-1] for col in self.dependencies),
                      axis=1)
        a.columns = self.dependencies
        a = a.fillna(0)
        return (a['net_profit'] + a['net_profit_1']) / a.sum(axis=1)
'''
Author: Hugo
Date: 2020-10-21 11:41:40
LastEditTime: 2020-10-21 12:00:47
LastEditors: Hugo
Description: 获取指数调仓时点
算法逻辑见:
    https://www.joinquant.com/view/community/detail/8d1dbee7c1cef8a31e988640232addeb
'''
from jqdata import *
import pandas as pd

# 时间处理
import calendar
from dateutil.parser import parse
import datetime 

import itertools  # 迭代器

###########################  时间处理 ###############################


class GetPeriodicDate(object):

    '''指定调仓周期 获取调仓时间段'''

    def __init__(self, start_date=None, end_date=None):

        if start_date and end_date:
            self._check_type(start_date, end_date)

    @property
    def get_periods(self):

        periods = self.CreatChangePos()
        periods = list(zip(periods[:-1], periods[1:]))

        return [(e[0], e[1]) if i == 0 else (OffsetDate(e[0], 1), e[1]) for i, e in enumerate(periods)]

    # 生成时间段中的各调仓时点
    def CreatChangePos(self, params: dict = {"months": (6, 12), "weekday": "Friday", 'spec_weekday': "2nd"}) -> list:
        '''
        start:YYYY-MM-DD
        end:YYYY-MM-DD
        =================
        return list[datetime.date]
        '''

        # 检查输入
        #self._check_type(start_date, end_date)
        s = self.__start_date.year
        e = self.__end_date.year

        period = list(range(s, e + 1, 1))

        c_p = []

        months = params['months']
        weekday = params['weekday']
        spec_weekday = params['spec_weekday']

        for y, m in itertools.product(range(s, e+1), months):

            c_p.append(self.find_change_day(y, m, weekday, spec_weekday))

        c_p = c_p + [self.__start_date, self.__end_date]
        c_p.sort()

        return list(filter(lambda x: ((x >= self.__start_date) & (x <= self.__end_date)), c_p))

    def _check_type(self, start_date, end_date):
        '''检查输入日期的格式'''
        if isinstance(start_date, (str, int)):
            self.__start_date = parse(start_date).date()

        if isinstance(end_date, (str, int)):
            self.__end_date = parse(end_date).date()

    # 判断某年某月的第N个周几的日期
    # 比如 2019，6月的第2个周五是几号
    # 中证指数基本上都是每年6\12月第二个周五的下个交易日

    @staticmethod
    def find_change_day(year, month, weekday, spec_weekday) -> datetime.date:
        '''
        find_day(y, 12, "Friday", "2nd")
        ================
        return datetime.date
            y年12月第二个周五
        '''
        DAY_NAMES = [day for day in calendar.day_name]
        day_index = DAY_NAMES.index(weekday)
        possible_dates = [
            week[day_index]
            for week in calendar.monthcalendar(year, month)
            if week[day_index]]  # remove zeroes

        if spec_weekday == 'teenth':

            for day_num in possible_dates:
                if 13 <= day_num <= 19:
                    return datetime.date(year, month, day_num)

        elif spec_weekday == 'last':
            day_index = -1
        elif spec_weekday == 'first':
            day_index = 0
        else:
            day_index = int(spec_weekday[0]) - 1
        return datetime.date(year, month, possible_dates[day_index])


def OffsetDate(end_date: str, count: int) -> datetime.date:
    '''
    end_date:为基准日期
    count:为正则后推，负为前推
    -----------
    return datetime.date
    '''

    trade_date = get_trade_days(end_date=end_date, count=1)[0]

    if count > 0:
        # 将end_date转为交易日

        trade_cal = get_all_trade_days().tolist()

        trade_idx = trade_cal.index(trade_date)

        return trade_cal[trade_idx + count]

    elif count < 0:

        return get_trade_days(end_date=trade_date, count=abs(count))[0]

    else:

        raise ValueError('别闹！')

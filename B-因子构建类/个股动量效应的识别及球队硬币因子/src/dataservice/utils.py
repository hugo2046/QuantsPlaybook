'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-20 16:26:04
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-20 17:03:17
Description: 
'''
from typing import Union
import datetime as dt
from dateutil.parser import parse

def format_dt(dt: Union[dt.datetime, dt.date, str], fm: str = "%Y-%m-%d") -> str:
    """格式化日期

    Args:
        dt (Union[dt.datetime, dt.date, str]):
        fm (str, optional): 格式化表达. Defaults to '%Y-%m-%d'.

    Returns:
        str
    """
    return parse(dt).strftime(fm) if isinstance(dt, str) else dt.strftime(fm)

def get_system_os() -> str:
    """获取系统 Windows/Linux"""
    import platform

    return platform.system().lower()
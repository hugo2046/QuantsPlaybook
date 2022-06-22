'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-08 13:30:08
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-22 10:47:06
Description: 
'''
import datetime as dt
from typing import Dict, Union

import pandas as pd
from dateutil.parser import parse
from IPython.display import display


def format_dt(dt: Union[dt.datetime, dt.date, str],
              fm: str = '%Y-%m-%d') -> str:
    """格式化日期

    Args:
        dt (Union[dt.datetime, dt.date, str]): 
        fm (str, optional): 格式化表达. Defaults to '%Y-%m-%d'.

    Returns:
        str
    """
    if isinstance(dt, str):

        return parse(dt).strftime(fm)

    else:

        return dt.strftime(fm)


def print_table(table: pd.DataFrame, name: str = None, fmt: str = None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)

'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-08 13:30:08
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-08 15:20:10
Description: 
'''


import pandas as pd
from IPython.display import display


def load_csv(path: str) -> pd.DataFrame:
    """获取csv

    Args:
        path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """

    price = pd.read_csv(path, index_col=[0], parse_dates=['time'])
    price.rename(columns={'time': 'datetime'}, inplace=True)
    price['openinterest'] = 0

    return price


def print_table(table:pd.DataFrame, name:str=None, fmt:str=None):
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

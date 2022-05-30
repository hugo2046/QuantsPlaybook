'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-05-27 16:31:21
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-05-27 16:31:46
Description: 
'''

import pandas as pd
import numpy as np


def trans_ser2datetime(ser: pd.Series) -> pd.Series:
    """将ser类型转为datetime

    Args:
        ser (pd.Series): _description_

    Raises:
        TypeError: _description_

    Returns:
        pd.Series: _description_
    """
    if not isinstance(ser, pd.Series):

        raise TypeError('ser必须为pd.Series')

    if (ser.dtype != np.dtype('O')) or (ser.dtype != np.dtype('<M8[ns]')):

        ser = pd.to_datetime(ser)

    return ser
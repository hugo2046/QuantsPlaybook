'''
Author: Hugo
Date: 2024-10-29 20:27:50
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-10-31 09:43:22
Description: 
'''
"""
Author: Hugo
Date: 2024-10-29 20:27:50
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-10-29 20:33:09
Description: vmacd_mtm

> 20240921-东北证券-量化择时系列之二：成交量择时指标标-VMACD_MTM
"""

from typing import Union
import pandas as pd
import numpy as np
from talib import MACD


def calc_vmacd_mtm(volume: Union[pd.DataFrame, pd.Series], period: int = 60):
    """
    计算成交量移动平均收敛/发散指标（VMACD）的动量（MTM）。

    :param volume: 成交量数据，可以是 pandas DataFrame 或 Series。
    :type volume: Union[pd.DataFrame, pd.Series]
    :param period: 计算动量的周期长度，默认为 60。
    :type period: int
    :return: 计算得到的 VMACD 动量值。
    :rtype: Union[pd.Series, pd.DataFrame]

    :raises ValueError: 如果 volume 不是 pd.Series 或 pd.DataFrame 类型。
    """

    if isinstance(volume, pd.Series):
        vmacd: pd.Series = MACD(volume, fastperiod=12, slowperiod=26, signalperiod=9)[
            -1
        ]
    elif isinstance(volume, pd.DataFrame):
        vmacd: Union[pd.Series, pd.DataFrame] = volume.apply(
            lambda ser: MACD(ser, fastperiod=12, slowperiod=26, signalperiod=9)[-1],
            raw=True,
        )
    else:
        raise ValueError("volume should be pd.Series or pd.DataFrame")

    zscore_vmacd: Union[pd.Series, pd.DataFrame] = (
        vmacd - vmacd.rolling(period).mean()
    ) / vmacd.rolling(period).std()

    return zscore_vmacd.diff(1).rolling(period).sum()


if __name__ == "__main__":

    print("vmacd_mtm")
    print("input series")
    volume: pd.Series = pd.Series(
        np.random.random(1000), index=pd.date_range("2024-01-01", periods=1000)
    )
    vmacd_mtm: pd.Series = calc_vmacd_mtm(volume)
    print(vmacd_mtm.tail())

    print("input dataframe")
    volume: pd.DataFrame = pd.DataFrame(
        np.random.random((1000, 3)),
        index=pd.date_range("2024-01-01", periods=1000),
        columns=["a", "b", "c"],
    )
    vmacd_mtm: pd.DataFrame = calc_vmacd_mtm(volume)
    print(vmacd_mtm.tail())

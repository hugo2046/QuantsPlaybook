'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-04-20 14:39:06
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-04-20 14:52:38
Description: 自定义ICU均线
'''
import backtrader as bt
import numpy as np
from scipy import stats


class IcuMaInd(bt.Indicator):
    packages = (
        ("numpy", "np"),
        ("scipy.stats", "stats"),
    )
    lines = ("icu",)
    params = (("N", 5),)  # 回看N期

    def __init__(self):
        self.addminperiod(self.p.N)
        self.high = self.data.close  # 因变量

    def next(self):
        close_N = self.high.get(ago=0, size=self.p.N)

        try:
            close_N: np.ndarray = np.array(close_N)
            res = stats.siegelslopes(close_N)
            self.lines.icu[0] = res.intercept + res.slope * (self.p.N - 1)

        except Exception as e:
            print("except")
            self.lines.icu[0] = 0
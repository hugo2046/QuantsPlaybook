"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-11 17:09:03
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-12 10:05:26
Description: 
"""
import pandas as pd


class LoadData(object):
    def __init__(self) -> None:
        import os

        this_file_path: str = os.path.split(os.path.dirname(__file__))[0]
        self._excel_file = pd.ExcelFile(f"{this_file_path}/data/data.xlsx")

    @property
    def pivot_swprice(self) -> pd.DataFrame:

        self.sw_price: pd.DataFrame = self._excel_file.parse(
            "sw_price", index_col=[0], parse_dates=["trade_date"]
        )

        return pd.pivot_table(
            self.sw_price, index="trade_date", columns="code", values="close"
        )

    @property
    def index_price(self) -> pd.DataFrame:

        return self._excel_file.parse(
            "index_price", index_col=[0], parse_dates=["trade_date"]
        )

    @property
    def sw_classify(self) -> pd.DataFrame:

        return self._excel_file.parse("sw_classify", index_col=[0])

"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-08-02 14:21:37
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-08-08 21:30:29
Description: 
"""
from typing import List, Union
import pandas as pd
import fire 
from rich.console import Console
from ts_data_service import distributed_query

console = Console()

START_DT: str = "20140101"
END_DT: str = "20230802"

CODES = ["510300.SH", "513100.SH", "159915.SZ", "510880.SH"]


def get_etf_price(
    codes: Union[str, List], start_dt: str, end_dt: str, limit: int = 800
) -> pd.DataFrame:
    price: pd.DataFrame = pd.concat(
        (
            distributed_query(
                "fund_daily",
                symbol=code,
                start_date=start_dt,
                end_date=end_dt,
                limit=limit,
            )
            for code in codes
        ),
        ignore_index=True,
    )
    price: pd.DataFrame = price.astype({"trade_date": "datetime64[D]"})
    return price


def main(codes: Union[List, str] = None, start_dt: str = None, end_dt: str = None):
    console.log("start to get data")
    if codes is None:
        codes = CODES
    if (start_dt is None) or (end_dt is None):
        start_dt = START_DT
        end_dt = END_DT
    price: pd.DataFrame = get_etf_price(codes, start_dt, end_dt)
    adj_factor: pd.DataFrame = distributed_query(
        "fund_adj", symbol=codes, start_date=start_dt, end_date=end_dt, limit=2000
    )

    adj_factor: pd.DataFrame = adj_factor.astype({"trade_date": "datetime64[D]"})

    all_data: pd.DataFrame = pd.merge(price, adj_factor, on=["ts_code", "trade_date"])

    hfq_price: pd.DataFrame = all_data.copy()
    hfq_price: pd.DataFrame = hfq_price.set_index("trade_date")

    fields: List = ["pre_close", "open", "high", "low", "close"]
    hfq_price[fields] = hfq_price[fields].mul(hfq_price["adj_factor"], axis=0)

    hfq_price.to_csv("data/hfq_price.csv")
    console.log("hfq_price saved")


if __name__ == "__main__":
    fire.Fire({'dump_all':main})

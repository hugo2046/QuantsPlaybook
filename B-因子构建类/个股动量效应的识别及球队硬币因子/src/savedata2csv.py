"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-27 08:46:46
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-27 09:21:59
Description: 
"""
from pathlib import Path
from typing import List, Union

import fire
import pandas as pd
from rich.console import Console
from rich.progress import track

from dataservice import get_price

console = Console()

FIELD_COL: List = [
    "open",
    "high",
    "low",
    "close",
    "vol",
    "adj_factor",
    "turnover_rate",
    "turnover_rate_f",
]


def save_data_to_csv(
    codes: Union[str, List] = None,
    start_date: str = None,
    end_date: str = None,
    count: int = None,
    fields: Union[str, List] = FIELD_COL,
    output_file: str = "data/cn_data",
) -> None:
    """获取数据并储存为csv

    Parameters
    ----------
    codes : Union[str, List], optional
        需要下载的标的,默认为None,表示获取当日全A, by default None
    start_date : str, optional
        起始日, by default None
    end_date : str, optional
        结束日, by default None
    fields : Union[str, List], optional
        查询字段, by default FIELD_COL
    output_file : str, optional
        数据保存位置, by default "data/cn_data"
    """
    console.log("开始获取数据...")
    price: pd.DataFrame = get_price(
        codes, start_date, end_date, count, fields, fq="hfq"
    )
    price["trade_date"] = pd.to_datetime(price["trade_date"])
    price.rename(columns={"vol": "volume", "adj_factor": "factor"}, inplace=True)
    console.print(f"数据大小: {price.shape}")
    if "volume" in price.columns:
        price["volume"] = price["volume"] * 100
    if "amount" in price.columns:
        price["amount"] = price["amount"] * 1000
    console.log("数据获取完毕!")
    for code, df in track(price.groupby("code"), description="开始保存数据"):
        df.to_csv(Path(output_file) / f"{code}.csv", index=False)
    console.log("[green]数据保存完毕!")


if __name__ == "__main__":
    fire.Fire(save_data_to_csv)

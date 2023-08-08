"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-20 16:26:04
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-08-04 14:02:32
Description: 
"""
import datetime as dt
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch


def format_dt(dt: Union[dt.datetime, dt.date, str], fm: str = "%Y-%m-%d") -> str:
    """格式化日期

    Args:
        dt (Union[dt.datetime, dt.date, str]):
        fm (str, optional): 格式化表达. Defaults to '%Y-%m-%d'.

    Returns:
        str
    """
    return pd.to_datetime(dt).strftime(fm) if isinstance(dt, str) else dt.strftime(fm)


def reduce_dimensions(arr: torch.Tensor) -> torch.Tensor:
    return arr.reshape(arr.shape[:-1])


def expand_dimensions(arr: torch.Tensor) -> torch.Tensor:
    shape: List = list(arr.shape)
    shape.append([1])
    return arr.reshape(shape)


def trans2tensor(df: pd.DataFrame, field: str) -> torch.Tensor:
    return torch.stack(
        tuple(
            torch.from_numpy(df[field].fillna(0).values).to("cuda").float()
            for _, df in df.groupby(level="ts_code")
        ),
        dim=1,
    )


def plot_pred_nan_num(pred: torch.Tensor):
    nan_num: pd.Series = pd.Series(
        pred.isnan().sum(dim=1).cpu().numpy() / pred.shape[1]
    )
    normal: pd.Series = 1 - nan_num
    fig, ax = plt.subplots(figsize=(16, 0.85))
    nan_num.plot(
        ax=ax, kind="area", stacked=True, color="red", alpha=0.25, label="预测值为空"
    )  # 红色为预测值为空
    normal.plot(
        ax=ax, kind="area", stacked=True, color="green", alpha=0.25, label="有预测值"
    )  # 绿色为有预测值
    plt.legend()
    return ax

def all_nan(tensor):
    return torch.all(torch.isnan(tensor))


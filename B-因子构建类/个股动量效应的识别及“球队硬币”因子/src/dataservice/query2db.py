"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-20 13:32:10
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-20 13:53:31
Description: 
"""

from typing import Dict, List, Tuple, Union

import pandas as pd
from sqlalchemy import select

from .db_tools import DBConn
from .trade_cal import Tdaysoffset
from .utils import get_system_os

FIELD_DICT: Dict = {
    "adj_factor": ["adj_factor"],
    "daily": [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ],
    "valuation": [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ],
}


def _preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """数据预处理"""

    return (
        df.pipe(
            pd.DataFrame.drop_duplicates, subset=["code", "trade_date"], keep="last"
        )
        .pipe(pd.DataFrame.astype, {"trade_date": "datetime64[ns]"})
        .pipe(pd.DataFrame.sort_values, "trade_date")
    )


def _check_params(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count: int = None,
    fields: Union[str, List, Tuple] = None,
    table_name: str = None,
    date_name: str = "end_date",
    dml=None,
):
    """用于检查table_name查询对应表的数据

    Args:
        codes (Union[str, Tuple, List], optional): 查询的股票列表,为空时表示全市场. Defaults to None.
        start_date (str, optional): 起始日. Defaults to None.
        end_date (str, optional): 结束日. Defaults to None.
        start_date,end_date均为空时表示全表
        fields (Union[str, List, Tuple], optional): 查询对应表格的字段数据. Defaults to None.
        table_name (str, optional): 需要查询的表名. Defaults to None.

    Returns:
        [type]: [description]
    """
    if (count is not None) and (start_date is not None):
        raise ValueError("不能同时指定 start_date 和 count 两个参数")

    if count is not None:
        end_date = pd.to_datetime(end_date)
        count = int(-count)
        start_date = Tdaysoffset(end_date, count)
    elif isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)

    if table_name is None:
        raise ValueError("table_name不能为空")

    model = model = dml.auto_db_base.classes[table_name]

    if not any([codes, start_date, end_date, fields]):
        raise ValueError("参数不能全为空")

    if fields is None:
        raise ValueError("fields不能为空")

    expr_list: List = []
    if codes is not None:
        if isinstance(codes, str):
            codes = [codes]

        expr_list.append(model.code.in_(codes))

    if (start_date is not None) and (end_date is not None):
        expr_list.extend(
            [
                getattr(model, date_name) >= start_date,
                getattr(model, date_name) <= end_date,
            ]
        )

    if isinstance(fields, str):
        fields: List = [fields]

    fields: List = list(set(fields + ["code", date_name]))
    fields: List = [getattr(model, field) for field in fields]

    return select(*fields).where(*expr_list)


def query_data(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count: int = None,
    fields: Union[str, List, Tuple] = None,
    table_name: str = None,
    date_name: str = "end_date",
) -> pd.DataFrame:
    dml = DBConn()
    # dml.connect()
    # session = dml.Session()
    stmt = _check_params(
        codes, start_date, end_date, count, fields, table_name, date_name, dml
    )

    return pd.read_sql(stmt, dml.engine.connect())


def query_adj_factor(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count: int = None,
    fields: Union[str, List, Tuple] = None,
) -> pd.DataFrame:
    """查询复权因子数据"""
    df: pd.DataFrame = query_data(
        codes, start_date, end_date, count, fields, "adj_factor", "trade_date"
    )
    df: pd.DataFrame = df.drop_duplicates(["code", "trade_date"], keep="last")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df: pd.DataFrame = df.sort_values("trade_date")
    return _preprocessing(df)


def query_daily_valuation(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count: int = None,
    fields: Union[str, List, Tuple] = None,
) -> pd.DataFrame:
    """查询每日估值数据"""
    df: pd.DataFrame = query_data(
        codes, start_date, end_date, count, fields, "daily_basic", "trade_date"
    )
    return _preprocessing(df)


def query_daily(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count: int = None,
    fields: Union[str, List, Tuple] = None,
) -> pd.DataFrame:
    """查询日线数据"""
    df: pd.DataFrame = query_data(
        codes, start_date, end_date, count, fields, "daily", "trade_date"
    )

    return _preprocessing(df)


def get_price(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count: int = None,
    fields: Union[str, List, Tuple] = None,
    fq: str = "hfq",
) -> pd.DataFrame:
    """查询价格数据

    Args:
        codes (Union[str, Tuple, List], optional): 查询标的. Defaults to None.
        start_date (str, optional): 起始日. Defaults to None.
        end_date (str, optional): 结束日. Defaults to None.
        fields (Union[str, List, Tuple], optional): 字段. Defaults to None.
        fq (str, optional): hfq为后复权,None为不复权. Defaults to "hfq".

    Returns:
        pd.DataFrame: _description_
    """

    drop_field: bool = False
    if fq is not None:
        adj_factor: pd.DataFrame = query_adj_factor(
            codes,
            start_date,
            end_date,
            count,
            fields=["code", "trade_date", "adj_factor"],
        ).set_index(["code", "trade_date"])
        if "adj_factor" in fields:
            fields.remove("adj_factor")
            drop_field: bool = True

    # daily: pd.DataFrame = query_daily(
    #     codes, start_date, end_date, count, fields
    # ).set_index(["code", "trade_date"])
    func_dict: Dict = {"daily": query_daily, "valuation": query_daily_valuation}
    daily_fields: List = [field for field in fields if field in FIELD_DICT["daily"]]
    valuation_fields: List = [
        field for field in fields if field in FIELD_DICT["valuation"]
    ]

    daily: pd.DataFrame = pd.merge(
        func_dict["daily"](codes, start_date, end_date, count, daily_fields),
        func_dict["valuation"](codes, start_date, end_date, count, valuation_fields),
        on=["code", "trade_date"],
        how="outer",
    )
    daily: pd.DataFrame = daily.set_index(["code", "trade_date"])
    if adj_fields := [
        col
        for col in daily.columns
        if col in ["open", "high", "low", "close", "pre_close"]
    ]:
        daily[adj_fields]: pd.DataFrame = daily[adj_fields].mul(
            adj_factor["adj_factor"], axis=0
        )

    if drop_field:
        daily: pd.DataFrame = pd.concat((daily, adj_factor), axis=1)

    return daily.reset_index()

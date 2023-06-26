"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-20 13:32:10
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-20 13:53:31
Description: 
"""

from typing import List, Union, Tuple

import pandas as pd
from sqlalchemy import select
from .utils import get_system_os
from .db_tools import DBConn
from .trade_cal import Tdaysoffset

def query_tag_concept(
    tag: str, watch_dt: str = None, fields: Union[List, str] = None
) -> pd.DataFrame:
    """查询数据库中的标签情况

    Parameters
    ----------
    tag : str
        pool-为自定义标签
    watch_dt : str, optional
        查询日期,yyyy-mm-dd, by default None
    fields : Union[List, str], optional
        查询字段, by default None

    Returns
    -------
    pd.DataFrame
        index-idx columns-sec_name|code|trade_date|block_name|+查询字段
    """
    db_con = DBConn(f"{get_system_os()}_conn_str", "datacenter")
    defalut_fields: List = [
        "sec_name",
        "code",
        "trade_date",
        "block_name",
    ]

    if fields is None:
        fields: List = [
            "vol_ratio",
            "turnover",
        ]
    if isinstance(fields, str):
        fields: List = [fields]
    fields: List = list(set(defalut_fields + fields))

    model = db_con.auto_db_base.classes["dashbord_plate_cons"]
    expr: List = [model.category == tag]
    if watch_dt is not None:
        expr.append(model.trade_date == watch_dt)
    stmt = select(*(getattr(model, field) for field in fields)).where(*expr)
    # db_con.engine
    return pd.read_sql(stmt, db_con.engine)


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
    count:int=None,
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
    count:int=None,
    fields: Union[str, List, Tuple] = None,
    table_name: str = None,
    date_name: str = "end_date",
) -> pd.DataFrame:
    dml = DBConn()
    dml.connect()
    # session = dml.Session()
    stmt = _check_params(
        codes, start_date, end_date,count, fields, table_name, date_name, dml
    )

    return pd.read_sql(stmt, dml.engine.connect())


def query_adj_factor(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count:int=None,
    fields: Union[str, List, Tuple] = None,
) -> pd.DataFrame:
    """查询复权因子数据"""
    df: pd.DataFrame = query_data(
        codes, start_date, end_date,count, fields, "adj_factor", "trade_date"
    )
    df: pd.DataFrame = df.drop_duplicates(["code", "trade_date"], keep="last")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df: pd.DataFrame = df.sort_values("trade_date")
    return _preprocessing(df)


def query_daily(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count:int=None,
    fields: Union[str, List, Tuple] = None,
) -> pd.DataFrame:
    """查询日线数据"""
    df: pd.DataFrame = query_data(
        codes, start_date, end_date,count, fields, "daily", "trade_date"
    )

    return _preprocessing(df)


def get_price(
    codes: Union[str, Tuple, List] = None,
    start_date: str = None,
    end_date: str = None,
    count:int=None,
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
            codes, start_date, end_date,count, fields=["code", "trade_date", "adj_factor"]
        ).set_index(["code", "trade_date"])
        if "adj_factor" in fields:
            fields.remove("adj_factor")
            drop_field: bool = True

    daily: pd.DataFrame = query_daily(codes, start_date, end_date,count, fields).set_index(
        ["code", "trade_date"]
    )

    if adj_fields := [
        col
        for col in daily.columns
        if col in ["open", "high", "low", "close", "pre_close"]
    ]:
        df: pd.DataFrame = daily[adj_fields].mul(adj_factor["adj_factor"], axis=0)

    if drop_field:
        df: pd.DataFrame = pd.concat((df, adj_factor), axis=1)

    return df.reset_index()

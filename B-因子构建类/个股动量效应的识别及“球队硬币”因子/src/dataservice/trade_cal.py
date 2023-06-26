import pandas as pd
from .db_tools import DBConn
from functools import lru_cache
from .utils import get_system_os, format_dt
from sqlalchemy import select


@lru_cache()
def get_all_trade_days() -> pd.DatetimeIndex:
    """获取全部交易日历"""

    db_con = DBConn(f"{get_system_os()}_conn_str", "datacenter")
    model = db_con.auto_db_base.classes["trade_cal"]
    stmt = select(model.cal_date).where(model.is_open == 1)
    cal_frame: pd.DataFrame = pd.read_sql(stmt, db_con.engine)

    return pd.to_datetime(cal_frame["cal_date"].unique())


def create_cal(trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """构造交易日历

    Args:
        trade_dates (pd.DatatimeIndex, optional): 交易日. Defaults to None.

    Returns:
        pd.DataFrame: 交易日历表
    """

    min_date = trade_dates.min()
    max_date = trade_dates.max()

    dates = pd.date_range(min_date, max_date)
    df = pd.DataFrame(index=dates)

    df["is_tradeday"] = False
    df.loc[trade_dates, "is_tradeday"] = True

    return df


def get_trade_days(
    start_date: str = None, end_date: str = None, count: int = None
) -> pd.DatetimeIndex:
    """获取区间交易日

    Args:
        start_date (str, optional): 起始日. Defaults to None.
        end_date (str, optional): 结束日. Defaults to None.
        count (int, optional): 便宜. Defaults to None.

    Returns:
        pd.DatetimeIndex: 交易区间
    """
    if count is not None:
        if start_date is not None:
            raise ValueError("不能同时指定 start_date 和 count 两个参数")

        count = int(-count)
        start_date = Tdaysoffset(end_date, count)

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)

    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    start_date = format_dt(start_date)
    end_date = format_dt(end_date)
    all_trade_days = get_all_trade_days()
    idx = all_trade_days.slice_indexer(start_date, end_date)

    return all_trade_days[idx]


def Tdaysoffset(watch_date: str, count: int, freq: str = "days") -> pd.Timestamp:
    """日期偏移

    Args:
        watch_date (str): 观察日
        count (int): 偏离日
        freq (str):频率,D-日度,W-周度,M-月份,Y-年度
    Returns:
        dt.datetime: 目标日期
    """

    if isinstance(watch_date, str):
        watch_date = pd.to_datetime(watch_date)

    all_trade_days = get_all_trade_days()
    cal_frame = create_cal(trade_dates=all_trade_days)

    holiday = cal_frame.query("not is_tradeday").index
    trade_days = pd.offsets.CustomBusinessDay(weekmask="1" * 7, holidays=holiday)
    # None时为Days
    if freq == "days":
        target = watch_date + trade_days * 0 + trade_days * count

    else:
        # 此处需要验证
        # TODO：验证
        target = watch_date + pd.DateOffset(**{freq: count}) + trade_days

    return target

def get_current_dt(hour: int = 18) -> pd.Timestamp:

    hour_time:int = 18 * 100
    now_dt: pd.Timestamp = pd.Timestamp.now()
    now_time:int = int(pd.Timestamp.now().strftime('%H%S'))

    today:pd.Timestamp = pd.Timestamp.today().date()

    prevoius_day:pd.Timestamp = Tdaysoffset(today, -1).date()

    if now_time >= hour_time and now_dt.weekday() not in [5, 6]:
        # 18点后 如果今天是交易日则返回当日 否则返回最近的交易日
        return today if today == prevoius_day else prevoius_day
    else:

        return prevoius_day
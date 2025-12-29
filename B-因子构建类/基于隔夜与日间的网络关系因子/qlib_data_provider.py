"""
Author: Hugo
Date: 2025-10-28 13:39:58
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-10-29 14:32:31
Description:
qlib数据提供模块

本模块基于qlib提供金融数据获取功能，支持获取股票收益率数据并进行预处理。
基于用户提供的qlib数据获取流程进行封装。

主要功能：
- 连接qlib数据库
- 获取股票池数据
- 计算多种收益率类型
- 数据格式转换
- 支持自定义时间范围和股票池
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Union

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QlibConfig:
    """
    Qlib 初始化配置。

    Attributes
    ----------
    database_uri : str
        数据库连接URI。
        支持环境变量 DOLPHINDB_URI，格式: dolphindb://username:password@host:port
        示例: export DOLPHINDB_URI="dolphindb://admin:123456@172.17.0.1:8848"
    region : str
        市场区域，例如 `qlib.constant.REG_CN`。
    """

    database_uri: str = os.getenv(
        "DOLPHINDB_URI",
        "dolphindb://username:password@host:port"  # 默认占位符，请设置环境变量
    )
    region: str = REG_CN


qlib_config = QlibConfig()


class QlibDataProvider:
    """
    一个基于Qlib的数据提供者，用于获取和处理股票收益率数据。

    该类在初始化时获取指定股票池和时间范围内的日收益率、
    日内收益率和隔夜收益率，并将其处理为透视表格式。

    Parameters
    ----------
    codes : Union[List[str], str]
        股票池。如果为字符串，则应为预定义的市场名称
        (例如, "csi300", "csi500", "ashares")。
        如果为列表，则应为股票代码列表。
    start_date : str
        数据获取的开始日期，格式为 "YYYY-MM-DD"。
    end_date : str
        数据获取的结束日期，格式为 "YYYY-MM-DD"。

    Attributes
    ----------
    instruments_ : List[str]
        经过解析后的股票代码列表。
    daily_return_df : pd.DataFrame
        日收益率数据框 (日期 x 股票)。
    daytime_return_df : pd.DataFrame
        日内收益率数据框 (日期 x 股票)。
    overnight_return_df : pd.DataFrame
        隔夜收益率数据框 (日期 x 股票)。
    """

    _qlib_initialized: bool = False

    # 日收益率表达式
    DAILY_RETURN_EXPR: str = "$close/$preclose-1"
    # 日内收益率表达式
    DAYTIME_EXPR: str = "$close/$open-1"
    # 隔夜收益率表达式
    OVERNIGHT_EXPR: str = f"(1+{DAILY_RETURN_EXPR})/(1+{DAYTIME_EXPR})-1"

    @classmethod
    def init_qlib_once(cls) -> None:
        """
        执行全局Qlib初始化（仅一次）。

        使用 `QlibConfig` 中的参数进行初始化。
        """
        if not cls._qlib_initialized:
            qlib.init(
                database_uri=qlib_config.database_uri,
                region=qlib_config.region,
            )
            cls._qlib_initialized = True
            logger.info("Qlib 初始化成功。")

    def __init__(
        self, codes: Union[List[str], str], start_date: str, end_date: str
    ) -> None:
        self.init_qlib_once()
        self.instruments_ = self._parse_instruments(codes)
        self._pivot_features = self._fetch_and_pivot_features(start_date, end_date)

    def _parse_instruments(self, codes: Union[List[str], str]) -> List[str]:
        """解析输入的股票池参数。"""
        if isinstance(codes, str):
            if codes not in ["csi300", "csi500", "ashares"]:
                raise ValueError(
                    f"当 `codes` 为字符串时，仅支持 'csi300', 'csi500', 'ashares'，但收到了 '{codes}'。"
                )
            return D.instruments(market=codes)
        elif isinstance(codes, list):
            return codes
        else:
            raise TypeError("`codes` 参数必须是 `List[str]` 或 `str` 类型。")

    def _fetch_and_pivot_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取特征数据并将其转换为透视表。"""
        features: pd.DataFrame = D.features(
            self.instruments_,
            [self.DAILY_RETURN_EXPR, self.DAYTIME_EXPR, self.OVERNIGHT_EXPR],
            start_time=start_date,
            end_time=end_date,
        )
        features.rename(
            columns={
                self.DAILY_RETURN_EXPR: "daily_return",
                self.DAYTIME_EXPR: "daytime_return",
                self.OVERNIGHT_EXPR: "overnight_return",
            },
            inplace=True,
        )

        pivot_table = pd.pivot_table(
            features,
            index="datetime",
            columns="instrument",
            values=["daily_return", "daytime_return", "overnight_return"],
        )
        return pivot_table

    @property
    def daily_return_df(self) -> pd.DataFrame:
        """
        日收益率数据框。

        Returns
        -------
        pd.DataFrame
            一个透视表，索引为日期，列为股票代码，值为日收益率。
        """
        df: pd.DataFrame = self._pivot_features["daily_return"]
        cond = df.columns.str.endswith("BJ")
        return df.loc[:,~cond]

    @property
    def daytime_return_df(self) -> pd.DataFrame:
        """
        日内收益率数据框。

        Returns
        -------
        pd.DataFrame
            一个透视表，索引为日期，列为股票代码，值为日内收益率。
        """
        df: pd.DataFrame = self._pivot_features["daytime_return"]
        cond = df.columns.str.endswith("BJ")
        return df.loc[:,~cond]

    @property
    def overnight_return_df(self) -> pd.DataFrame:
        """
        隔夜收益率数据框。

        Returns
        -------
        pd.DataFrame
            一个透视表，索引为日期，列为股票代码，值为隔夜收益率。
        """
        df: pd.DataFrame = self._pivot_features["overnight_return"]
        cond = df.columns.str.endswith("BJ")
        return df.loc[:,~cond]

def get_trade_days(end_date: str, count: int) -> List[pd.Timestamp]:
    """
    获取指定结束日期前的交易日列表。

    Parameters
    ----------
    end_date : str
        结束日期，格式为 "YYYY-MM-DD"。
    count : int
        需要获取的交易日数量。

    Returns
    -------
    pd.DatetimeIndex
        包含指定数量交易日的日期索引。
    """
    all_days = pd.date_range(start="2000-01-01", end="2060-12-31", freq=pd.tseries.offsets.BDay())
    if end_date not in all_days:
        end_date: pd.DatetimeIndex = all_days.asof(end_date)

    idx: int = all_days.get_loc(end_date)
    target_idx: int = idx + count
    if target_idx > len(all_days) or target_idx < 0:
        raise ValueError("超出日期范围")
    return [all_days[target_idx]]
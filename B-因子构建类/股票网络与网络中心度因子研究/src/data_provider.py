"""
Author: Hugo
Date: 2025-12-10 19:53:29
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-12-10 20:01:02
Description:
"""

from dataclasses import dataclass
from typing import List, Union

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader

# 配置日志 - 使用项目统一的loguru系统
from loguru import logger


@dataclass
class QlibConfig:
    """
    Qlib 初始化配置。

    Attributes
    ----------
    database_uri : str
        数据库连接URI。
    region : str
        市场区域，例如 `qlib.constant.REG_CN`。
    """

    database_uri: str = "dolphindb://admin:123456@172.17.0.1:8848"
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

        self._start_date = start_date
        self._end_date = end_date
        self.D = D

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

    def get_features(
        self,
        instruments: str | List[str],
        start_time: str,
        end_time: str,
        fields: str | List[str],
    ) -> pd.DataFrame:
        return self.D.features(instruments, fields, start_time, end_time)

    def load_data(self, config) -> pd.DataFrame:
        # 配置数据加载器
        qld = QlibDataLoader(config=config)

        # 获取数据
        df: pd.DataFrame = qld.load(
            self.instruments_, start_time=self._start_date, end_time=self._end_date
        )

        return df

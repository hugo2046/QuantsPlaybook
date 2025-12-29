"""
Author: Hugo
Date: 2025-11-21 13:28:15
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-11-21 13:37:22
Description:数据获取
"""

import sys
from pathlib import Path
# 仅在调用时添加路径
sys.path.append("/data1/hugo/workspace")
sys.path.insert(0, "/data1/hugo/workspace/qlib_ddb")

from typing import List

import pandas as pd
import qlib
from factor_pipeline import FactorPipeline
from loguru import logger
from qlib.config import REG_CN
from qlib_data_provider import qlib_config

def to_abbr(name: str) -> str:
    return "".join(w[0] for w in name.split("_") if w).lower()


def load_factor(start_dt: str, end_dt: str, network_type: str, method: str):
   
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = FactorPipeline(
        codes="ashares",
        start_dt=start_dt,
        end_dt=end_dt,
        window=60,
        network_type=network_type,
        correlation_method=method,  # pearson
        top_percentile=0.2,
        bottom_percentile=0.2,
        lead_percentile=0.5,
    )

    pipeline.run()

    long_factor_df: pd.DataFrame = pipeline.long_df
    short_factor_df: pd.DataFrame = pipeline.short_df

    long_factor_df.where(long_factor_df != 0).to_parquet(
        out_dir / f"{to_abbr(network_type)}_{method}_long.parquet"
    )

    short_factor_df.where(short_factor_df != 0).to_parquet(
        out_dir / f"{to_abbr(network_type)}_{method}_short.parquet"
    )

network_types:List[str] = ["daytime_lead_overnight", "overnight_lead_daytime","preclose_lead_close"]
correlation_method:List[str] = ["pearson", "spearman"]

if __name__ == "__main__":
    
    # 初始化Qlib环境
    qlib.init(
        database_uri=qlib_config.database_uri,
        region=REG_CN,
    )
    
    for net_type in network_types:
        for method in correlation_method:
            logger.info(f"Start loading factor for {net_type} with {method}...")
            load_factor(
                start_dt="2020-01-01",
                end_dt="2025-10-31",
                network_type=net_type,
                method=method,
            )
            logger.success(f"Finished loading factor for {net_type} with {method}.")
            
    logger.success("All factors have been loaded and saved successfully.")
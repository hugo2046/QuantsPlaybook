"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-07-18 14:20:29
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-07-19 14:27:13
Description: 
"""
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from qlib.contrib.rolling.base import Rolling
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils.data import update_config
from qlib.workflow import R

from .build_factor import get_factor_data_and_forward_return
from .utils import load2qlib

DIRNAME: Path = Path(__file__).resolve().parent


class RollingBenchmark(Rolling):
    def __init__(
        self,
        train: Union[Tuple, List],
        valid: Union[Tuple, List],
        test: Union[Tuple, List],
        factor_obj: object,
        window: int = 20,
        h_path: Union[str, Path] = None,
        horizon: int = 20,
        **kwargs,
    ) -> None:
        conf_path: Union[str, Path] = (
            DIRNAME / "config" / "workflow_config_lightgbm_Alpha158.yaml"
        )
        super().__init__(conf_path=conf_path, h_path=h_path, horizon=horizon, **kwargs)
        self.h_path = h_path
        self.horizon = horizon
        self.train = train
        self.valid = valid
        self.test = test
        self.window = window
        self.factor_obj = factor_obj

    def _update_start_end_time(self, task: Dict):
        task["dataset"]["kwargs"]["segments"]["train"] = tuple(
            pd.Timestamp(i) for i in self.train
        )
        task["dataset"]["kwargs"]["segments"]["valid"] = tuple(
            pd.Timestamp(i) for i in self.valid
        )
        task["dataset"]["kwargs"]["segments"]["test"] = tuple(
            pd.Timestamp(i) for i in self.test
        )
        return task

    def _replace_hanler_with_cache(self, task: Dict):
        """
        Due to the data processing part in original rolling is slow. So we have to
        This class tries to add more feature
        """
        if self.h_path is not None:
            h_path = Path(self.h_path)
        else:
            h_path: Path = (
                DIRNAME.parent / "data" / "pkl" / f"coin_team_factor_{self.horizon}.pkl"
            )

            if not h_path.exists():
                # 获取因子数据
                all_data: pd.DataFrame = get_factor_data_and_forward_return(
                    self.factor_obj,
                    window=self.window,
                    periods=self.horizon + 1,  # 未来期收益标签
                    general_names=["interday", "intraday", "overnight"],
                )
                # 仅获取基础因子
                sel_cols: List = [
                    factor
                    for factor in all_data.columns
                    if (factor not in ["coin_team", "coin_team_f"])
                    and (factor.find("revise") == -1)
                ]
                # 保存因子数据
                dh_pr: DataHandlerLP = load2qlib(
                    all_data[sel_cols],
                    self.train,
                    self.valid,
                    self.test,
                    output_type="DataHandlerLP",
                )

                dh_pr.to_pickle(path=h_path, dump_all=True)

        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        return task

    def basic_task(self):
        task: Dict = self._raw_conf()["task"]
        task: Dict = deepcopy(task)

        task: Dict = self._replace_hanler_with_cache(task)
        task: Dict = self._update_start_end_time(task)

        if self.task_ext_conf is not None:
            task = update_config(task, self.task_ext_conf)
        self.logger.info(task)
        return task

    def get_pred(
        self,
        recorder_name: str = None,
        recorder_id: str = None,
        experiment_id: str = None,
        experiment_name: str = "predict",
    ) -> pd.DataFrame:

        recorder = R.get_recorder(
            recorder_name=recorder_name,
            recorder_id=recorder_id,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
        )
        label_df: pd.DataFrame = recorder.load_object("label.pkl")
        label_df.columns = ["label"]
        pred_df: pd.DataFrame = recorder.load_object("pred.pkl")

        return pd.concat((label_df, pred_df), axis=1, sort=True).reindex(label_df.index)

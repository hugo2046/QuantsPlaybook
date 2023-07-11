"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-07-05 13:58:36
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-07-10 22:03:01
FilePath: 
Description: 
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Text, Union
from qlib.data.dataset import DatasetH

from qlib.model.base import Model
from qlib.model.interpret.base import FeatureInt


class LGBRanker(Model, FeatureInt):
    """LightGBM Model"""

    def __init__(
        self,
        objective: str = "lambdarank",
        early_stopping_rounds: int = 50,
        num_boost_round: int = 1000,
        **kwargs
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.params = {"object": objective}
        self.params.update(kwargs)
        self.model = None

    def fit(
        self,
        dataset: DatasetH,
        quantiles: int = 5,
        duplicates: str = "drop",
        **kwargs
    ):
        if "save_path" in kwargs:
            del kwargs["save_path"]
        # prepare dataset for lgb training and evaluation
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key="learn"
        )

        x_train, y_train = df_train["feature"], df_train["label"].groupby(
            level="datetime", group_keys=False
        )["next_ret"].apply(
            lambda x: pd.qcut(x, quantiles, labels=False, duplicates=duplicates)
        )
        x_valid, y_valid = df_valid["feature"], df_valid["label"].groupby(
            level="datetime", group_keys=False
        )["next_ret"].apply(
            lambda x: pd.qcut(x, quantiles, labels=False, duplicates=duplicates)
        )

        y_train: pd.DataFrame = y_train.to_frame()
        y_valid: pd.DataFrame = y_valid.to_frame()
        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        group: np.ndarray = (
            x_train.index.get_level_values("instrument").to_series().value_counts()
        )
        vaild_group: np.ndarray = (
            x_valid.index.get_level_values("instrument").to_series().value_counts()
        )
        dtrain = lgb.Dataset(x_train.values, label=y_train, group=group)
        dvalid = lgb.Dataset(x_valid.values, label=y_valid, group=vaild_group)

        # fit the model
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(period=50),
            ],
            **kwargs
        )

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key="infer")
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        return pd.Series(
            self.model.feature_importance(*args, **kwargs),
            index=self.model.feature_name(),
        ).sort_values(  # pylint: disable=E1101
            ascending=False
        )

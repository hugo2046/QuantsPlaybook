from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit

from .general import generate_features
from .utils import reduce_dimensions, trans2tensor


def load_csv(path: str = None) -> pd.DataFrame:
    """读取csv数据

    Parameters
    ----------
    path : str, optional
        csc路径, by default None

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path: str = "data/hfq_price.csv"

    # 读取数据
    price: pd.DataFrame = pd.read_csv(
        path, index_col=0, parse_dates=["trade_date"]
    ).sort_index()

    return price


def preparer_data(
    df: pd.DataFrame,
    vol_forward_window: int = 21,
    standardize_window: int = 21,
    vol_method: str = None,
    return_method: str = "OTO",
) -> pd.DataFrame:
    """构造特征数据及预测数据

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV数据
    vol_forward_window : int, optional
        vol of vol的计算窗口, by default 21
    standardize_window : int, optional
        标准化的计算窗口, by default 21

    Returns
    -------
    pd.DataFrame
        MultiIndex: level0-ts_code, level1-trade_date
        columns-level0-features_fields|auxiliary_fields|next_ret
    """
    feild: str = {"OTO": "open", "CTC": "close"}[return_method]

    def _zscore_standardize(arr: np.ndarray) -> np.ndarray:
        return (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=1)

    # 获取完整的索引
    idx: pd.DatetimeIndex = df.index.unique()
    # 获取用于回测的收益数据
    df["OTO"] = df.groupby("ts_code", group_keys=True)[feild].transform(
        lambda x: x.pct_change().shift(-2)
    )
    # 构造特征及预测数据
    data: pd.DataFrame = df.groupby("ts_code", group_keys=True).apply(
        generate_features, method=vol_method
    )
    # 复制波动率数据
    data["Vr_21"] = data["Raw_21"]
    # 复制收益到data中
    data["OTO"] = df.set_index("ts_code", append=True).swaplevel().sort_index()["OTO"]
    # 重置cols
    data.columns = get_multi_cols(data.columns)

    data: pd.DataFrame = pd.concat(
        {
            code: df.droplevel(level=0).reindex(idx)
            for code, df in data.groupby(level="ts_code")
        }
    )
    data.index.names = ["ts_code", "trade_date"]
    # 获取远期波动率用于预测
    data["auxiliary_fields"] = (
        data["auxiliary_fields"]
        .groupby(level="ts_code")
        .transform(lambda x: x.shift(-vol_forward_window))
    )

    # 标准化
    data["features_fields"] = (
        data["features_fields"]
        .groupby(level="ts_code")
        .transform(
            lambda x: x.rolling(standardize_window).apply(
                lambda x: _zscore_standardize(x)[-1], raw=True
            )
        )
    )

    #
    data: pd.DataFrame = pd.concat(
        {
            code: df.droplevel(level=0).iloc[:-vol_forward_window]
            for code, df in data.groupby(level="ts_code")
        }
    )
    data.index.names = ["ts_code", "trade_date"]
    return data


def get_multi_cols(cols: pd.Index) -> pd.MultiIndex:
    features_name: List[str] = [
        "ln_return_1",
        "ln_return_5",
        "ln_return_21",
        "ln_return_126",
        "ln_return_252",
        "Raw_5_avg",
        "Raw_21_avg",
        "Raw_252_avg",
        "Raw_126_avg",
        "Raw_63_avg",
        "Raw_5",
        "Raw_21",
        "Raw_252",
        "Raw_126",
        "Raw_63",
    ]

    target_fields: List[str] = [
        "Vr_21",  # 0
        "Parkinson_21",  # 1
        "GarmanKlass_21",  # 2
        "RogersSatchell_21",  # 3
        "YangZhang_21",  # 4
    ]

    next_rets_fields: List[str] = ["OTO"]

    fields: List[Tuple] = []
    for col in cols:
        if col in features_name:
            fields.append(("features_fields", col))
        elif col in target_fields:
            fields.append(("auxiliary_fields", col))
        elif col in next_rets_fields:
            fields.append(("next_ret", col))

    return pd.MultiIndex.from_tuples(fields)


def get_dataset(
    data: Tuple, test_size: int, valid_ratio: float, base_ratio: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """将data转为train_dataset, valid_dataset, test_dataset

    Parameters
    ----------
    data : Tuple
        MultiIndex: level0-ts_code, level1-trade_date
        columns-level0-features_fields|auxiliary_fields|next_ret
    test_size : int
        预测数据的长度
    valid_ratio : float
        验证集占训练集的比例

    ------
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
    """
    features: torch.Tensor = trans2tensor(data, "features_fields")

    auxiliary: torch.Tensor = trans2tensor(data, "auxiliary_fields")

    next_ret: torch.Tensor = reduce_dimensions(trans2tensor(data, "next_ret"))

    feature_size: int = features.shape[0]
    auxiliary_size: int = auxiliary.shape[0]
    next_ret_size: int = next_ret.shape[0]
    if feature_size != auxiliary_size != next_ret_size:
        raise ValueError(
            f"The size of features({feature_size}), auxiliary({auxiliary_size}) and next_ret({next_ret_size}) are not equal."
        )

    # 扩展交叉验证默认起始使用base_ratio的数据作为训练集
    tscv = TimeSeriesSplit(
        n_splits=int((features.shape[0] * (1 - base_ratio)) // test_size),
        test_size=test_size,
    )

    train_dataset: List = []
    valid_dataset: List = []
    test_dataset: List = []
    print(tscv)
    for train_index, test_index in tscv.split(features):
        valid_size: int = int(len(train_index) * valid_ratio)

        train_dataset.append(
            (
                features[train_index[:-valid_size]],
                next_ret[train_index[:-valid_size]],
                auxiliary[train_index[:-valid_size]],
            )
        )
        valid_dataset.append(
            (
                features[train_index[-valid_size:]],
                next_ret[train_index[-valid_size:]],
                auxiliary[train_index[-valid_size:]],
            )
        )
        test_dataset.append(
            (features[test_index], next_ret[test_index], auxiliary[test_index])
        )

    return train_dataset, valid_dataset, test_dataset


class DataProcessor:
    def __init__(self, path: str = None) -> None:
        self.price = load_csv(path)

    def generate(
        self,
        vol_forward_window: int = 21,
        standardize_window: int = 21,
        vol_method: str = None,
        return_method: str = "OTO",
    ) -> pd.DataFrame:
        self.frame: pd.DataFrame = preparer_data(
            self.price,
            vol_forward_window,
            standardize_window,
            vol_method=vol_method,
            return_method=return_method,
        )
        # 特征数量
        self.feature_num = self.frame["features_fields"].shape[1]
        self.features_name = self.frame["features_fields"].columns.tolist()
        # 预测数据数量
        self.auxiliary_num = self.frame["auxiliary_fields"].shape[1]
        self.auxiliary_name = self.frame["auxiliary_fields"].columns.tolist()
        self.idx: pd.DatetimeIndex = self.frame.index.levels[1]

    def build_dataset(
        self, test_size: int, valid_ratio: float, base_ratio: float = 0.3
    ):
        try:
            self.frame
        except NameError as e:
            raise NameError("You should generate data first.") from e

        self.train_dataset, self.valid_dataset, self.test_dataset = get_dataset(
            self.frame, test_size, valid_ratio, base_ratio
        )

        total_size: int = len(self.idx)

        self.test_idx: pd.DatetimeIndex = self.idx[
            -int(test_size * total_size * (1 - base_ratio) // test_size) :
        ]

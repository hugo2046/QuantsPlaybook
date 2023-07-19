from functools import partial
from typing import Dict, Tuple, Union

import pandas as pd
from qlib.backtest import backtest, executor
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config
from qlib.workflow import R  # 实验记录管理器
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
from rich.console import Console

console = Console()


################################### DATASET CONFIG ###################################


def get_tsdataset_config(
    pool: str,
    train: Tuple,
    valid: Tuple,
    test: Tuple,
    step_len: int = 20,
    calssmethod: str = "Chips",
) -> Dict:
    data_handler_config: Dict = {
        "start_time": train[0],
        "end_time": test[1],
        "fit_start_time": train[0],
        "fit_end_time": train[1],
        "instruments": pool,
    }

    return {  # 　因子数据集参数配置
        "class": "TSDatasetH",
        # 数据集类所在模块
        "module_path": "qlib.data.dataset",
        # 数据集类的参数配置
        "kwargs": {
            "handler": {  # 数据集使用的数据处理器配置
                "class": calssmethod,  # 数据处理器类，继承自DataHandlerLP
                "module_path": "scr.factor_expr",  # 数据处理器类所在模块
                "kwargs": data_handler_config,  # 数据处理器参数配置
            },
            "segments": {  # 数据集时段划分
                "train": train,  # 训练集时段
                "valid": valid,  # 验证集时段
                "test": test,  # 测试集时段
            },
            "step_len": step_len,
        },
    }


def get_dataset_config(
    pool: str, train: Tuple, valid: Tuple, test: Tuple, calssmethod: str = "Chips"
) -> Dict:
    data_handler_config: Dict = {
        "start_time": train[0],
        "end_time": test[1],
        "fit_start_time": train[0],
        "fit_end_time": train[1],
        "instruments": pool,
    }

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": calssmethod,
                "module_path": "scr.factor_expr",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": train,
                "valid": valid,
                "test": test,
            },
        },
    }


################################### MODEL CONFIG ###################################


def get_gru_config(
    d_feat: int = 6,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,
    n_epochs: int = 200,
    lr: float = 0.001,
    metric="",
    batch_size=2000,
    early_stop=20,
    loss="mse",
    optimizer="adam",
    GPU=0,
    seed=None,
    n_jobs=20,
    method: str = "normal",
) -> Dict:
    if method == "ts":
        module_path: str = "qlib.contrib.model.pytorch_gru_ts"
    else:
        module_path: str = "qlib.contrib.model.pytorch_gru"
    return {
        "class": "GRU",
        "module_path": module_path,
        "kwargs": {
            "d_feat": d_feat,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "lr": lr,
            "early_stop": early_stop,
            "batch_size": batch_size,
            "metric": metric,
            "loss": loss,
            "optimizer": optimizer,
            "n_jobs": n_jobs,
            "seed": seed,
            "GPU": GPU,
        },
    }


def get_transformer_config(
    d_feat: int = 20,
    d_model: int = 64,
    batch_size: int = 8192,
    nhead: int = 2,
    num_layers: int = 2,
    dropout: float = 0,
    n_epochs: int = 100,
    lr: float = 0.0001,
    metric: str = "",
    early_stop: int = 5,
    loss: str = "mse",
    optimizer: str = "adam",
    reg: float = 1e-3,
    n_jobs: int = 10,
    GPU: int = 0,
    seed: int = None,
    method: str = "ts",
) -> Dict:
    method: str = method.lower()
    module_path: str = {
        "ts": "qlib.contrib.model.pytorch_transformer_ts",
        "normal": "qlib.contrib.model.pytorch_transformer",
    }[method]

    return {
        "class": "TransformerModel",
        "module_path": module_path,
        "kwargs": {
            "d_feat": d_feat,
            "d_model": d_model,
            "batch_size": batch_size,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "lr": lr,
            "metric": metric,
            "early_stop": early_stop,
            "loss": loss,
            "optimizer": optimizer,
            "reg": reg,
            "n_jobs": n_jobs,
            "GPU": GPU,
            "seed": seed,
        },
    }


def get_alstm_ts_config(
    d_feat: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,
    n_epochs: int = 200,
    lr: float = 1e-3,
    early_stop: int = 20,
    batch_size: int = 800,
    metric: str = "loss",
    loss: str = "mse",
    n_jobs: int = 20,
) -> Dict:
    return {
        "class": "ALSTM",
        "module_path": "qlib.contrib.model.pytorch_alstm_ts",
        "kwargs": {
            "d_feat": d_feat,  # 步长
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "lr": lr,
            "early_stop": early_stop,
            "batch_size": batch_size,
            "metric": metric,
            "loss": loss,
            "n_jobs": n_jobs,
            "GPU": 0,
            "rnn_type": "GRU",
        },
    }


def get_adarnn_config(
    d_feat: int = 12,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,
    n_epochs: int = 200,
    lr: float = 1e-3,
    early_stop: int = 20,
    batch_size: int = 800,
    metric: str = "loss",
    loss: str = "mse",
) -> Dict:
    return {
        "class": "ADARNN",
        "module_path": "qlib.contrib.model.pytorch_adarnn",
        "kwargs": {
            "d_feat": d_feat,  # 特征维度
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "lr": lr,
            "early_stop": early_stop,
            "batch_size": batch_size,
            "metric": metric,
            "loss": loss,
            "GPU": 0,
        },
    }


def get_tcn_ts_config(
    d_feat: int = 20,
    num_layers: int = 8,
    n_chans: int = 32,
    kernel_size: int = 7,
    dropout: float = 0.5,
    n_epochs: int = 200,
    lr: float = 1e-4,
    early_stop: int = 20,
    batch_size: int = 2000,
    metric: str = "loss",
    loss: str = "mse",
    optimizer: str = "adam",
    n_jobs: int = 20,
    GPU: int = 0,
) -> Dict:
    return {
        "class": "TCN",
        "module_path": "qlib.contrib.model.pytorch_tcn_ts",
        "kwargs": {
            "d_feat": d_feat,
            "num_layers": num_layers,
            "n_chans": n_chans,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "lr": lr,
            "early_stop": early_stop,
            "batch_size": batch_size,
            "metric": metric,
            "loss": loss,
            "optimizer": optimizer,
            "n_jobs": n_jobs,
            "GPU": GPU,
        },
    }


def get_gbdt_config(
    loss: str = "mse",
    colsample_bytree: float = 0.8879,
    learning_rate: float = 0.0421,
    subsample: float = 0.8789,
    lambda_l1: float = 205.6999,
    lambda_l2: float = 580.9768,
    max_depth: int = 15,
    num_leaves: int = 210,
    num_threads: int = 20,
    early_stopping_rounds: int = 200,
    num_boost_round: int = 1000,
) -> Dict:
    return {
        # 模型类
        "class": "LGBModel",
        # 模型类所在模块
        "module_path": "qlib.contrib.model.gbdt",
        # 模型类超参数配置，未写的则采用默认值。这些参数传给模型类
        "kwargs": {  # kwargs用于初始化上面的class
            "loss": loss,
            "colsample_bytree": colsample_bytree,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "num_threads": num_threads,
            "early_stopping_rounds": early_stopping_rounds,  # 训练迭代提前停止条件
            "num_boost_round": num_boost_round,  # 最大训练迭代次数
        },
    }


def get_ranker_config(
    metric="ndcg",
    device_type="gpu",
    boosting_type="gbdt",
    num_leaves=20,
    max_depth=-1,
    learning_rate=0.05,
    ndcg_eval_at=[5],
    n_estimators=200,
    seed=42,
    n_jobs=-1,
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=20,
    reg_alpha=0.0,
    reg_lambda=0.0,
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=0,
) -> Dict:
    import sys
    from pathlib import Path

    path = Path(__file__).parent

    sys.path.append(str(path))
    return {
        "class": "LGBRanker",
        "module_path": "src.LGBRanker",  # "qlib.contrib.model.gbdt_ranker",
        "kwargs": {
            "object": "lambdarank",
            "boosting_type": boosting_type,
            "device_type": device_type,
            "metric": metric,
            "ndcg_eval_at": ndcg_eval_at,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "min_split_gain": min_split_gain,
            "min_child_weight": min_child_weight,
            "min_child_samples": min_child_samples,
            "random_state": seed,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "subsample_for_bin": subsample_for_bin,
            "subsample_freq": subsample_freq,
            "n_jobs": n_jobs,
        },
    }


MODEL_CONFIG: Dict = {
    "transformer_ts": partial(get_transformer_config, method="ts"),
    "gru_ts": partial(get_gru_config, method="ts"),
    "gru": get_gru_config,
    "transformer": get_transformer_config,
    "alstm": get_alstm_ts_config,
    "adarnn": get_adarnn_config,
    "tcn_ts": get_tcn_ts_config,
    "gbdt": get_gbdt_config,
    "ranker": get_ranker_config,
}


class QlibFlow:
    def __init__(
        self,
        dataset: DatasetH=None,
        model: str=None,
        start_time: str=None,
        end_time: str=None,
        model_kw: Dict = None,
    ) -> None:
        self.dataset = dataset
        self.model_name = model
        self.start_time = start_time
        self.end_time = end_time
        self.model_kw = model_kw
        self.R = None


    def _create_model(self):
        if self.model_kw is None:
            self.model_kw: Dict = {}

        if self.model_name is None:
            raise ValueError("请指定模型!")
        
        model_name: str = self.model_name.lower()
        model_config: Dict = MODEL_CONFIG[model_name]()
        model_config["kwargs"].update(self.model_kw)

        # if self.model_kw is not None:
        #     model_config: Dict = MODEL_CONFIG[model_name](**self.model_kw)
        # else:
        #     model_config: Dict = MODEL_CONFIG[model_name]()

        self.save_path: str = (
            "tmp.pth" if model_name not in ["ranker", "gbdt"] else None
        )

        self.model = init_instance_by_config(model_config)

    def _exsit_R(self):
        if self.R is None:
            raise ValueError("请先训练模型!")

    def fit(self, experiment_name: str = "train", **kwargs) -> None:
        """train"""
        # 生成模型
        self._create_model()
        
        if self.dataset is None:
            raise ValueError("请指定数据集!")
        self.R = R
        try:
            self.save_path
        except NameError as e:
            raise ValueError("请先初始化模型!") from e

        with self.R.start(experiment_name=experiment_name):
            self.model.fit(self.dataset, save_path=self.save_path, **kwargs)
            self.R.save_objects(**{"trained_model.pkl": self.model})
            console.print("train info")
            console.print(self.R.get_recorder().info)

    def predict(self, experiment_name: str = "predict") -> None:
        """predict"""

        self._exsit_R()

        with self.R.start(experiment_name=experiment_name):
            predict_recorder = self.R.get_recorder()
            sig_rec = SignalRecord(self.model, self.dataset, predict_recorder)
            sig_rec.generate()

            sigAna_rec = SigAnaRecord(predict_recorder)
            sigAna_rec.generate()
            console.print("predict info")
            console.print(self.R.get_recorder().info)
            self.predict_recorder = predict_recorder

    def backtest(
        self,
        pred_score: pd.DataFrame = None,
        start_time: str = None,
        end_time: str = None,
        topk: int = 30,
        n_drop: int = 5,
        account: Union[int, float] = 100000000,
        benchmark: str = "000300.SH",
        hold_thresh: int = 1,
        only_tradable: bool = True,
        risk_degree: float = 0.95,
        limit_threshold: float = 0.095,
        deal_price: str = "open",
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        min_cost: int = 5,
        freq: str = "day",
        impact_cost: float = 0.005,
        trade_unit: int = 100,
        verbose: bool = False,
    ) -> None:
        if pred_score is None:
            pred_score: Tuple = (self.model, self.dataset)

        if (start_time is None) or (end_time is None):
            start_time: str = self.start_time
            end_time: str = self.end_time

        STRATEGY_CONFIG = {
            "topk": topk,
            "n_drop": n_drop,
            # pred_score, pd.Series
            "signal": pred_score,  # 也可以是(model, dataset) 或 pred_df，得到测试集的预测值score
            "only_tradable": only_tradable,  # 仅可交易股票
            "risk_degree": risk_degree,  # 资金使用比率
            "hold_thresh": hold_thresh,  # 股票最小持有天数,默认1天
        }

        EXECUTOR_CONFIG = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "verbose": verbose,
        }

        backtest_config = {
            "start_time": start_time,  # test集开始时间
            "end_time": end_time,  # test集结束时间
            "account": account,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": freq,
                "limit_threshold": limit_threshold,  # 涨跌停板幅度
                "deal_price": deal_price,  # 以开盘价成交
                "open_cost": open_cost,  # 开仓佣金费率
                "close_cost": close_cost,  # 平仓佣金费率
                "min_cost": min_cost,  # 一笔交易的最小成本
                "impact_cost": impact_cost,  # 冲击成本费率，比如因滑点产生的冲击成本
                "trade_unit": trade_unit,  # 对应复权前的交易量为100的整数倍
            },
        }

        # strategy object，创建策略对象
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        # executor object， 策略执行器对象
        executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
        # backtest， 回测
        self.portfolio_metric_dict, self.indicator_dict = backtest(
            executor=executor_obj, strategy=strategy_obj, **backtest_config
        )

    def get_pred(
        self,
        recorder_name: str = None,
        recorder_id: str = None,
        experiment_id: str = None,
        experiment_name: str = "predict",
    ) -> pd.DataFrame:
        if self.R is None:
            self.R = R

        recorder = self.R.get_recorder(
            recorder_name=recorder_name,
            recorder_id=recorder_id,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
        )
        label_df: pd.DataFrame = recorder.load_object("label.pkl")
        label_df.columns = ["label"]
        pred_df: pd.DataFrame = recorder.load_object("pred.pkl")

        return pd.concat((label_df, pred_df), axis=1, sort=True).reindex(label_df.index)
    


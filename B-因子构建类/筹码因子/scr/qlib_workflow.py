from typing import Dict, Tuple
from functools import partial
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


MODEL_CONFIG: Dict = {
    "transformer_ts": partial(get_transformer_config, method="ts"),
    "transformer": get_transformer_config,
    "alstm": get_alstm_ts_config,
    "adarnn": get_adarnn_config,
    "tcn_ts": get_tcn_ts_config,
    "gbdt": get_gbdt_config,
}


def run_model(
    dataset: DatasetH,
    model: str,
    start_time: str,
    end_time: str,
    model_kw: Dict = None,
    experiment_name: str = "workflow",
    trained_model: str = "trained_model.pkl",
) -> Dict:
    model: str = model.lower()

    if model_kw is not None:
        print()
        model_config: Dict = MODEL_CONFIG[model](**model_kw)
    else:
        model_config: Dict = MODEL_CONFIG[model]()

    save_path = "tmp.pth" if model != "gbdt" else None

    model = init_instance_by_config(model_config)
    # R变量可以理解为实验记录管理器。
    console.log(f"实验名:{experiment_name},训练模型:{trained_model},开始运行...")

    with R.start(experiment_name=experiment_name):  # 注意，设好实验名
        ############
        # 训练
        #############

        model.fit(dataset, save_path=save_path)

        # 训练好的模型以pkl文件形式保存到本次实验运行记录目录下的artifacts子目录
        R.save_objects(**{trained_model: model})

        ###############
        # 预测
        #############
        # 本次实验的实验记录器
        recorder = R.get_recorder()
        # 生成预测结果文件
        sig_rec = SignalRecord(model, dataset, recorder)
        sig_rec.generate()

        # 生成预测结果分析文件
        sigAna_rec = SigAnaRecord(recorder)
        sigAna_rec.generate()

        # 回测所需参数配置
        port_analysis_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                    # "verbose": True, # 是否打印订单执行记录
                },
            },
            "strategy": {  # 回测策略相关超参数配置
                "class": "TopkDropoutStrategy",  # 策略类名称
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    # "model": model,  # 模型对象
                    # "dataset": dataset,  # 数据集
                    "signal": (model, dataset),  # 信号，也可以是pred_df，得到测试集的预测值score
                    "topk": 30,
                    "n_drop": 0,
                    "only_tradable": True,
                    "risk_degree": 0.95,  # 资金使用比率
                    "hold_thresh": 1,  # 股票最小持有天数,默认1天
                },
            },
            "backtest": {  # 回测数据参数
                "start_time": start_time,  # test集开始时间
                "end_time": end_time,  # test集结束时间
                "account": 100000000,
                "benchmark": "000300.SH",  # 基准
                "exchange_kwargs": {
                    "freq": "day",  # 使用日线数据
                    "limit_threshold": 0.095,  # 涨跌停板幅度
                    "deal_price": "open",  # 以开盘价成交
                    "open_cost": 0.0005,  # 开仓佣金费率
                    "close_cost": 0.0015,  # 平仓佣金费率
                    "min_cost": 5,  # 一笔交易的最小成本
                    "impact_cost": 0.005,  # 冲击成本费率，比如因滑点产生的冲击成本
                    "trade_unit": 100,  # 对应复权前的交易量为100的整数倍
                },
            },
        }

        ###############
        # 回测
        #############
        pa_rec = PortAnaRecord(recorder, port_analysis_config, "day")
        pa_rec.generate()

        # 打印本次实验记录器信息，含记录器id，experiment_id等信息
        print("info", R.get_recorder().info)

    return {
        "recorder": recorder,
        "sig_rec": sig_rec,
        "sigAna_rec": sigAna_rec,
        "pa_rec": pa_rec,
    }

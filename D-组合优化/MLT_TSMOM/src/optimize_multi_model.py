"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-08-07 10:21:35
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-08-07 10:21:56
Description: 
"""
from typing import List

import optuna
import torch
from .data_processor import DataProcessor
from .module import Multi_Task_Model

from .core import MTL_TSMOM  # train_loop

TARGET_VOL: float = 0.1
TRANSCATION_COST: float = 0.0003
FEATURE_SIZE: int = 15
BATCH_SIZE: int = 100
EARLY_STOPPING = 25
LOG_STEP: int = 50
VERBOSE: bool = False


def optimize_multi_hyperparameters(
    dataset: DataProcessor,
    target_vol: float = TARGET_VOL,
    transcation_cost: float = TRANSCATION_COST,
    batch_size: int = BATCH_SIZE,
    early_stopping: int = EARLY_STOPPING,
    log_step: int = LOG_STEP,
    verbose: bool = VERBOSE,
    n_trials: int = 20,
) -> None:
    HIDDEN_HYPER_SPACE: List[int] = [32, 64, 128, 256]
    DROP_HYPER_SPACE: List[float] = [0.05, 0.10, 0.15, 0.20]

    def objective(trial: optuna.trial.Trial) -> float:
        # We optimize the number of layers, hidden units in each layer and dropouts.
        lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
        mlp_layers = trial.suggest_int("mlp_layers", 1, 4)
        lstm_hidden_size = trial.suggest_categorical(
            "lstm_hidden_size", HIDDEN_HYPER_SPACE
        )
        mlp_hidden_size = trial.suggest_categorical(
            "mpl_hidden_size", HIDDEN_HYPER_SPACE
        )
        lstm_dropout = trial.suggest_categorical("lstm_dropout", DROP_HYPER_SPACE)
        mlp_dropout = trial.suggest_categorical("mlp_dropout", DROP_HYPER_SPACE)
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.01, 0.1, 0.0])
        lr = trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2, 0.1])

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

        mtl_tsmom = MTL_TSMOM(
            dataset=dataset,
            input_size=FEATURE_SIZE,
            lstm_hidden_size=lstm_hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            lstm_layers=lstm_layers,
            mpl_layers=mlp_layers,
            lstm_dropout=lstm_dropout,
            mpl_dropout=mlp_dropout,
            max_grad_norm=max_grad_norm,
            optimizer_name=optimizer_name,
            opt_kwargs={"lr": lr},
            target_vol=target_vol,
            transcation_cost=transcation_cost,
            num_epochs=batch_size,
            early_stopping=early_stopping,
            log_step=log_step,
            verbose=verbose,
        )

        for epoch, (train_, valid_) in enumerate(
            zip(dataset.train_dataset, dataset.valid_dataset)
        ):
            valid_loss = mtl_tsmom.loop(train_, valid_)
            trial.report(valid_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return valid_loss

    # 997min16.7s
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("best_params:")
    print(study.best_trial.params)

    return study

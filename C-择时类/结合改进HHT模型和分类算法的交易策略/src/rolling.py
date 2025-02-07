'''
Author: Hugo
Date: 2025-02-06 12:53:29
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-02-06 14:33:41
Description: 
'''
from typing import Any, List, Optional, Union

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


class Rolling:
    """
    滚动训练预测类
    
    该类实现了基于滑动窗口的模型训练和预测。支持分类和回归模型，
    可以处理sklearn、lightgbm、xgboost等主流机器学习模型。
    
    属性:
        df (pd.DataFrame): 输入数据
        feature_cols (List[str]): 特征列名列表
        target_col (str): 目标变量列名
        step (int): 滚动步长
        train_window (int): 训练窗口大小
        horizon (int): 预测周期
        return_probability (bool): 是否返回概率值
        predictions: 预测结果
        models (dict): 存储每个时间点的模型
    
    示例:
        >>> rolling = Rolling(df, ['feature1', 'feature2'], 'target')
        >>> predictions = rolling.run()
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model: Optional[Any] = None,
        model_params: Optional[dict] = None,
        step: int = 5,
        train_window: int = 60,
        horizon: int = 20,
        return_probability: bool = True,
    ):
        """
        简化版滚动训练类

        参数:
            df: 输入数据框，需要按时间排序
            feature_cols: 特征列名列表
            target_col: 目标变量列名
            model: 模型类或实例，支持sklearn、lightgbm、xgboost等
            model_params: 模型参数字典
            step: 滚动步长
            train_window: 训练窗口大小
            horizon: 预测周期
            return_probability: 是否返回概率值（仅对分类模型有效）
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.step = step
        self.train_window = train_window
        self.horizon = horizon
        self.return_probability = return_probability
        self.model_params = model_params or {}

        # 验证输入数据
        self._validate_inputs()

        # 初始化模型
        self._init_model(model)

        # 根据模型类型初始化预测结果存储
        self._init_predictions()

        self.models = {}

    def _is_classifier(self, model):
        """判断模型是否为分类器"""
        if isinstance(model, str):
            return model in [
                "rf",
                "RandomForestClassifier",
                "lgb",
                "LGBMClassifier",
                "xgb",
                "XGBClassifier",
            ]
        return isinstance(
            model, (ClassifierMixin, lgb.LGBMClassifier, xgb.XGBClassifier)
        )

    def _init_predictions(self):
        """根据模型类型初始化预测结果存储"""
        if self._is_classifier(self.model):
            if self.return_probability:
                unique_classes = sorted(self.df[self.target_col].unique())
                self.predictions = {
                    cls: pd.Series(index=self.df.index, dtype=float)
                    for cls in unique_classes
                }
            else:
                self.predictions = pd.Series(
                    index=self.df.index, dtype=self.df[self.target_col].dtype
                )
        else:
            # 回归模型
            self.predictions = pd.Series(index=self.df.index, dtype=float)

    def _init_model(self, model):
        """初始化模型"""
        if model is None:
            self.model = RandomForestClassifier(random_state=42, **self.model_params)
        elif isinstance(model, (str, type)):
            # 如果传入的是字符串或类型
            model_map = {
                # 分类器
                "rf": RandomForestClassifier,
                "RandomForestClassifier": RandomForestClassifier,
                "lgb": lgb.LGBMClassifier,
                "LGBMClassifier": lgb.LGBMClassifier,
                "xgb": xgb.XGBClassifier,
                "XGBClassifier": xgb.XGBClassifier,
                # 可以添加更多模型...
            }

            if isinstance(model, str):
                model_class = model_map.get(model)
                if model_class is None:
                    raise ValueError(f"不支持的模型名称: {model}")
            else:
                model_class = model

            self.model = model_class(**self.model_params)
        else:
            # 如果传入的是模型实例
            self.model = model

    def predict(self, model, X: pd.DataFrame):
        """预测"""
        if self._is_classifier(model) and self.return_probability:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            else:
                raise ValueError("模型不支持概率预测")
        return model.predict(X)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """训练模型"""
        if hasattr(self.model, 'get_params'):
            # 如果模型支持get_params，创建新实例
            model = self.model.__class__(**self.model.get_params())
        else:
            # 否则直接使用原始模型
            model = self.model
            
        # 直接使用feature_cols作为categorical_feature
        if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            model.fit(X, y, categorical_feature=self.feature_cols)
        else:
            model.fit(X, y)
        return model

    def _get_train_test_data(
        self, train_start_idx: int, train_end_idx: int, test_idx: int
    ):
        """获取训练和测试数据"""
        # 训练数据
        train_data = self.df.iloc[train_start_idx:train_end_idx]
        X_train = train_data[self.feature_cols]
        y_train = train_data[self.target_col]  # 使用转换后的分类标签

        # 测试数据
        test_data = self.df.iloc[test_idx : test_idx + self.horizon]
        X_test = test_data[self.feature_cols]

        return X_train, y_train, X_test

    def run(self):
        """执行滚动训练和预测"""

        total_rows: int = len(self.df)
        # 计算总迭代次数
        n_iterations: int = (
            total_rows - self.train_window - self.horizon
        ) // self.step + 1

        # 确保数据量足够
        if total_rows < self.train_window:
            raise ValueError(
                f"数据量({total_rows})小于训练窗口大小({self.train_window})"
            )

        # 初始化训练起点
        current_train_start: int = 0

        # 使用tqdm显示进度
        with tqdm(total=n_iterations, desc="Rolling prediction") as pbar:

            # 迭代滚动训练和预测
            while current_train_start + self.train_window + self.horizon <= total_rows:
                # 计算当前训练集和测试集的索引
                train_end = current_train_start + self.train_window
                test_start = train_end

                # 获取训练和测试数据
                X_train, y_train, X_test = self._get_train_test_data(
                    current_train_start, train_end, test_start
                )

                # 训练模型
                current_model = self.train(X_train, y_train)

                # 预测
                preds = self.predict(current_model, X_test)

                # 存储预测结果
                pred_indices = self.df.index[test_start : test_start + self.horizon]
                # 存储预测结果
                pred_indices = self.df.index[test_start : test_start + self.horizon]
                if self._is_classifier(self.model) and self.return_probability:
                    # 对于分类器的概率预测，需要分别存储每个类别的概率
                    for i, class_label in enumerate(sorted(self.predictions.keys())):
                        self.predictions[class_label].loc[pred_indices] = preds[:, i]
                else:
                    # 对于非概率预测或回归模型
                    self.predictions.loc[pred_indices] = preds

                # 存储模型
                self.models[self.df.index[test_start]] = current_model

                # 移动到下一个训练起点
                current_train_start += self.step

                pbar.update(1)

        return self.predictions

    def evaluate(self, metric_func=None):
        """
        评估模型性能

        参数:
            metric_func: 可选的评估函数，接受y_true和y_pred作为参数
        """
        predictions = self.get_predictions()
        if predictions.empty:
            return None

        # 获取实际值
        actual = self.df[self.target_col].loc[predictions.index]

        if metric_func is not None:
            return metric_func(actual, predictions)

        # 默认评估指标
        if self._is_classifier(self.model):
            from sklearn.metrics import accuracy_score, classification_report

            if self.return_probability:
                # 对于概率预测，转换为类别
                pred_classes = pd.DataFrame(predictions).idxmax(axis=1)
                return classification_report(actual, pred_classes)
            return accuracy_score(actual, predictions)
        else:
            from sklearn.metrics import mean_squared_error, r2_score

            return {
                "mse": mean_squared_error(actual, predictions),
                "r2": r2_score(actual, predictions),
            }

    def _validate_inputs(self):
        """验证输入数据的有效性"""
        # 检查特征列是否存在
        missing_cols = [col for col in self.feature_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"特征列不存在: {missing_cols}")

        # 检查目标列是否存在
        if self.target_col not in self.df.columns:
            raise ValueError(f"目标列不存在: {self.target_col}")

        # 检查时间索引
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("数据框必须有时间索引")

        # 检查参数有效性
        if self.train_window <= 0:
            raise ValueError("train_window必须大于0")
        if self.step <= 0:
            raise ValueError("step必须大于0")
        if self.horizon <= 0:
            raise ValueError("horizon必须大于0")

    def get_model(self, timestamp):
        """获取特定时间点的模型"""
        return self.models.get(timestamp)

    def get_predictions(self):
        """获取所有预测结果"""
        return self.predictions.dropna()


def run_rolling_prediction(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    step: int = 5,
    train_window: int = 60,
    horizon: int = 20,
    model=None,
    model_params: Optional[dict] = None,
    return_probability: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """
    便捷函数用于执行滚动预测

    返回:
        对于非概率预测返回pd.Series
        对于概率预测返回pd.DataFrame，每列为一个类别的概率
    """
    rolling = Rolling(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        step=step,
        train_window=train_window,
        horizon=horizon,
        model=model,
        model_params=model_params,
        return_probability=return_probability,
    )

    predictions = rolling.run()
    return pd.DataFrame(predictions) if isinstance(predictions, dict) else predictions
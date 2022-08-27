'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-08-15 09:13:32
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-08-19 22:40:33
Description:
'''
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from sklearn import mixture
from tqdm.notebook import tqdm

from .trader import Trader
from .utils import create_empty_lists, double_loop, rolling_window


class Company(object):
    def __init__(
        self,
        stock_names: List[str],
        M: int,
        max_lag: int,
        l: int,
        time_window: int,
        activation_funcs: List[Callable],
        binary_operators: List[Callable],
        traders_num: int,
        Q: float = 0.5,
        unimodal: str = 'Gaussian',
        evaluation_protocols: str = 'ACC',
        seed: int = None,
    ) -> None:

        self.M = M
        self.max_lag = max_lag
        self.l = l
        self.time_window = time_window
        self.activation_funcs = activation_funcs
        self.binary_operators = binary_operators
        self.trader_num = traders_num
        self.Q = Q
        self.unimodal = unimodal
        self.evaluation_protocols = evaluation_protocols
        self.stock_num: int = len(stock_names)

        # 用于储存预测值
        self.prediction_by_trader: List[np.ndarray] = create_empty_lists(
            self.trader_num)

        if seed:
            np.random.seed(seed)

        self.traders: List[Trader] = [
            Trader(
                M=M,
                A=activation_funcs,
                O=binary_operators,
                stock_num=self.stock_num,
                time_window=time_window,
                max_lag=max_lag,
            ) for _ in range(traders_num)
        ]

        self.UNIMODAL_DICT: Dict = {
            'GaussianMixture': mixture.GaussianMixture,
            'BayesianGaussianMixture': mixture.BayesianGaussianMixture
        }

    def fit(self, train_data: np.ndarray) -> None:
        # 保存训练集数据
        self.train_data: np.ndarray = train_data
        total_size: int = len(train_data)
        for t in tqdm(range(self.max_lag + self.l, total_size), desc="TC模型训练"):

            sub_window_data: np.ndarray = train_data[t - self.max_lag -
                                                     self.l:t - self.l]
            self._get_trader_prediction(sub_window_data)

            if t >= self.time_window + self.max_lag + self.l:

                self.y_true: np.ndarray = train_data[t - self.time_window:t +
                                                     1]
                self.educate(self.y_true)
                self.prune_and_generate(train_data, self.y_true, t)

    def educate(self, y_true: np.ndarray) -> None:

        mean_squared_err: np.ndarray = self._get_mse(y_true)

        bad_traders: np.ndarray = self.get_bad_trader_flag(
            mean_squared_err, 1 - self.Q)

        [
            self.traders[trader_i].learn(stock_i, y_true[:, stock_i])
            for trader_i, stock_i in double_loop(bad_traders)
        ]

    def predict(self, data: np.ndarray) -> None:

        self.train_data = np.vstack((self.train_data, data))
        current_t: int = len(self.train_data)

        self._get_trader_prediction(self.train_data[current_t - self.max_lag -
                                                    self.l:current_t])
        y_true: np.ndarray = self.train_data[current_t - self.time_window -
                                             self.l:current_t]

        self.educate(y_true)
        self.prune_and_generate(self.train_data, y_true, current_t)

    def prune_and_generate(self, all_data: np.ndarray, y_true: np.ndarray,
                           end_idx: int) -> None:

        mean_squared_err: np.ndarray = self._get_mse(y_true)
        bad_traders: np.ndarray = self.get_bad_trader_flag(
            mean_squared_err, 1 - self.Q)

        unimodal_res = None

        if self.unimodal in self.UNIMODAL_DICT:
            # 当unimodal为高斯混合/贝叶斯时 将good trader的参数给bad trader
            good_trader_params: np.ndarray = self._get_stack_params(
                ~bad_traders)
            good_trader_factor_m: np.ndarray = self._get_stack_factor_num(
                ~bad_traders)

            # 因子参数进行高斯混合/贝叶斯

            unimodal_factor_res = self._unimodal_func(good_trader_params)
            unimodal_factor_m_res = self._unimodal_func(good_trader_factor_m)
            unimodal_res: Tuple = (unimodal_factor_res, unimodal_factor_m_res)

        self.roll_data = rolling_window(
            all_data[end_idx - self.time_window - self.max_lag -
                     self.l:end_idx],
            self.max_lag + self.l,
        )

        for trader_i, rows in enumerate(bad_traders):

            for stock_i, flag in enumerate(rows):

                if flag:
                    # 重置不合格的
                    # 这里使用的高斯/贝叶斯分布直接替换的bad trader
                    self.traders[trader_i].reset_params(stock_i,
                                                        unimodal=unimodal_res)
                    # 重新学习
                    for data in self.roll_data:

                        self.traders[trader_i].calc_factors(data[:-self.l],
                                                            stock_i=stock_i)

    def get_bad_trader_flag(self, mean_squared_err: np.ndarray,
                            q: float) -> np.ndarray:
        """获取bad_trader

        Parameters
        ----------
        data : np.ndarray
            all_data
        i : int
            当前位置
        """

        # 获取百分位数据
        q: float = np.quantile(mean_squared_err, q)
        # r_n小于q的部分为bad_trader
        return mean_squared_err > q

    def aggregate(self):
        # 这里跟论文原文的处理不同 我这里只将表现好的交易员取出作为预测
        weights: np.ndarray = np.zeros((self.trader_num, self.stock_num))
        predictions: np.ndarray = np.zeros((self.trader_num, self.stock_num))

        mse: np.ndarray = np.array(
            [trader.mean_squared_error for trader in self.traders])

        bad_traders: np.ndarray = self.get_bad_trader_flag(mse, self.Q)

        for trader_i, rows in enumerate(bad_traders):

            for stock_i, flag in enumerate(rows):

                if not flag:

                    weights[trader_i, stock_i] = (
                        1.0 /
                        self.traders[trader_i].mean_squared_error[stock_i])

                    predictions[trader_i, stock_i] = self.traders[
                        trader_i].get_current_predict()[stock_i]

        return np.true_divide((weights * predictions).sum(axis=0),
                              weights.sum(axis=0))

    def _get_trader_prediction(self, data: np.ndarray) -> None:
        """trader根据data进行预测"""
        [trader.calc_bulk_factors(data) for trader in self.traders]

    def _get_mse(self, y_true: np.ndarray) -> np.ndarray:
        # sourcery skip: remove-unnecessary-else

        if self.evaluation_protocols == 'MSE':
            # 原始代码 注意MSE的结构跟ACC不一样 需要调整
            return np.squeeze([
                trader.calc_mean_squared_error(y_true)
                for trader in self.traders
            ])
        else:
            # 这里仅用于aggregate 获取MSE
            [trader.evaluation(y_true, 'MSE') for trader in self.traders]
            # 这里用于剪枝及学习的评价

            return np.vstack([
                trader.evaluation(y_true, self.evaluation_protocols)
                for trader in self.traders
            ])

    def _get_stack_params(self, trader_flag_arr: np.ndarray) -> np.ndarray:
        """合并params"""

        params_ls: List = [
            self.traders[trader_i].get_params(stock_i)
            for trader_i, stock_i in double_loop(trader_flag_arr)
        ]

        return np.vstack(params_ls)

    def _get_stack_factor_num(self, trader_flag_arr: np.ndarray) -> np.ndarray:

        factor_num: List = [
            len(self.traders[trader_i].get_params(stock_i))
            for trader_i, stock_i in double_loop(trader_flag_arr)
        ]

        return np.array(factor_num).reshape(-1, 1)

    def _unimodal_func(self, X: np.ndarray):

        unimodal: Callable = self.UNIMODAL_DICT[self.unimodal]
        return unimodal(init_params='random_from_data').fit(X)
'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-08-15 09:13:32
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-08-27 16:09:24
Description:

Company aggregate函数待优化
'''

import functools
import multiprocessing
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple, Union

import empyrical as ep
import numpy as np
from scr.utils import calc_least_squares, calculate_best_chunk_size, rolling_window
from sklearn import mixture
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm

# 设置CPU可用核数
CPU_WORKER_NUM = int(multiprocessing.cpu_count() * 0.7)


class ParamsFactory(object):
    """参数工厂"""

    def __init__(self, A: List[Callable],
                 O: List[Callable],
                 stock_num: int,
                 M: int,
                 max_lag: int) -> None:
        """初始化

        Args:
            active_func (List[Callable]): 激活函数列表
            binary_oper (List[Callable]): 二元选择列表
            stock_num (int): 股票个数
            M (int): Alpha因子个数
            max_lag (int): 滞后期
        """
        self.A = A
        self.O = O
        self.stock_num = stock_num
        self.M = M
        self.max_lag = max_lag

        self.A_funcs2int_dic: Dict = self.trans_funcs2dict(A)
        self.O_funcs2in_dic: Dict = self.trans_funcs2dict(O)

        self.A_funcs2func_dic: Dict = self.reverse_dict(self.A_funcs2int_dic)
        self.O_funcs2func_dic: Dict = self.reverse_dict(self.O_funcs2in_dic)

    def create_funcs_params(self) -> None:
        """生成公式"""
        # 生成因子个数
        m = np.random.choice(np.arange(1, self.M + 1))
        # 选择对应的激活函数
        A = np.random.choice(self.A, m)
        O = np.random.choice(self.O, m)

        # 延迟参数
        size_rng: np.ndarray = np.arange(1, self.max_lag + 1)
        D = np.random.choice(size_rng, m)
        F = np.random.choice(size_rng, m)

        # 股票下标
        P = np.random.choice(self.stock_num, m)
        Q = np.random.choice(self.stock_num, m)

        # 列分别为A,O,D,F,P,Q,行为m m为生成的因子个数
        # A,O为callable
        self.funcs_params: np.ndarray = np.vstack([A, O, D, F, P, Q]).T
        # A,O为int
        self.int_params: np.ndarray = self.transform_params2int(
            self.funcs_params)

    def transform_params2func(self, int_params: np.ndarray) -> np.ndarray:
        """将A,O的数值转为可调用的函数

        Args:
            int_params (np.ndarray): A,O参数为数字

        Returns:
            np.ndarray: _description_
        """
        def _transform_int2func(arr: np.ndarray) -> np.ndarray:

            arr[0] = self.A_funcs2func_dic[arr[0]]
            arr[1] = self.O_funcs2func_dic[arr[1]]

            return arr

        int_params = np.copy(int_params)
        int_params = int_params.astype('object')

        return np.apply_along_axis(_transform_int2func, 1,
                                   self._parameter_bounds_check(int_params))

    def transform_params2int(self, funcs_params: np.ndarray) -> np.ndarray:
        """将A,O的可调用函数转为数值

        Args:
            funcs_params (np.ndarray):A,O参数为可调用的函数

        Returns:
            np.ndarray: _description_
        """
        def _transform_funcs2int(arr: np.ndarray) -> np.ndarray:

            arr[0] = self.A_funcs2int_dic[arr[0]]
            arr[1] = self.O_funcs2in_dic[arr[1]]

            return arr

        return np.apply_along_axis(_transform_funcs2int, 1,
                                   np.copy(funcs_params))

    def _parameter_bounds_check(self, params: np.ndarray):
        """参数边界检查"""
        # P,Q股票下标参数 A, O, D, F, P, Q
        theshold_stock_num: int = self.stock_num - 1

        # F,D严查参数
        theshold_lag: int = self.max_lag - 1

        # A
        params[:, 0] = np.clip(params[:, 0], 0, len(self.A) - 1)
        # O
        params[:, 1] = np.clip(params[:, 1], 0, len(self.O) - 1)
        # D
        params[:, 2] = np.clip(params[:, 2], 0, theshold_lag)
        # F
        params[:, 3] = np.clip(params[:, 3], 0, theshold_lag)
        # P
        params[:, 4] = np.clip(params[:, 4], 1, theshold_stock_num)
        # Q
        params[:, 5] = np.clip(params[:, 5], 1, theshold_stock_num)

        return params

    @staticmethod
    def trans_funcs2dict(funcs: List[Callable]) -> Dict:

        if not isinstance(funcs, (List, Tuple)):
            funcs = [funcs]

        return dict(zip(funcs, range(len(funcs))))

    @staticmethod
    def reverse_dict(dic: Dict) -> Dict:

        return {v: k for k, v in dic.items()}


class GenerationFitParams(object):

    def __init__(self, alpha_num: np.ndarray, params: np.ndarray, generate_method: str) -> None:

        GM_FUNC: Dict = {'GaussianMixture': mixture.GaussianMixture,
                         'BayesianGaussianMixture': mixture.BayesianGaussianMixture}[generate_method]

        if len(alpha_num.shape) < 2:

            alpha_num: np.ndarray = alpha_num.reshape(-1, 1)

        self.alhpa_num_gm = GM_FUNC(
            init_params='random_from_data').fit(alpha_num)
        self.params_gm = GM_FUNC(init_params='random_from_data').fit(params)

    def create_params(self, hight_m: int):

        m: int = np.rint(self.alhpa_num_gm.sample()[0][0][0]).astype(int)

        m: float = np.clip(m, 1, hight_m)

        return np.rint(self.params_gm.sample(m)[0]).astype(int)


class Trader(ParamsFactory):
    def __init__(self, A: List[Callable],
                 O: List[Callable],
                 stock_num: int,
                 M: int,
                 max_lag: int,
                 l: int) -> None:

        super().__init__(A, O, stock_num, M, max_lag)
        self.A = A
        self.O = O
        self.stock_num = stock_num
        self.M = M
        self.max_lag = max_lag
        self.l = l

        # 生成公式
        self.create_funcs_params()
        self.weights = self.get_weight()

    def get_prediction(self, X: np.ndarray) -> np.ndarray:
        """获取预测值"""
        # 列为M个因子的数据

        self.factors: np.ndarray = np.array(
            [self.formula(X, param) for param in self.funcs_params]).T

        # 预测值
        self.y_pred: np.ndarray = (self.factors * self.weights).sum(axis=1)

        return self.y_pred

    def set_params(self, int_params: np.ndarray) -> None:

        self.funcs_params = self.transform_params2func(int_params)
        self.weights = self.get_weight()

    def reset_params(self) -> None:

        self.create_funcs_params()
        self.weights = self.get_weight()

    def learn(self,
              y_true: np.ndarray,
              func: Callable = calc_least_squares) -> None:
        """更新权重

        Args:
            y_true (np.ndarray): 真实值
            func (Callable, optional): 权重生成的方法,默认使用最小二乘法. Defaults to calc_least_squares.
        """
        self.weights: np.ndarray = func(self.factors, y_true)
        # 更新预测值
        self.y_pred: np.ndarray = (self.factors * self.weights).sum(axis=1)

    def formula(self, X: np.ndarray, param: Union[np.ndarray, List,
                                                  Tuple]) -> Tuple[np.ndarray]:
        """使用公式获取数据

        Args:
            X (np.ndarrat): 数据集
            param (Union[np.ndarray,List,Tuple]): 容器储存的公式参数 (A,O,D,F,P,Q)

        Returns:
            Tuple[np.ndarray]: 0-r_{p_j},1-r_{q_j}
        """
        A, O, D, F, P, Q = param

        # 获取数据
        indices = np.arange(self.max_lag, len(X) - self.l)

        stock_p_d: np.ndarray = np.take(X[:, P], indices - D)

        stock_q_f: np.ndarray = np.take(X[:, Q], indices - F)

        return A(O(stock_p_d, stock_q_f))

    def get_weight(self) -> np.ndarray:

        return np.random.randn(self.funcs_params.shape[0])

    def calc_returns(self, y_true: np.ndarray) -> np.ndarray:

        return np.multiply(np.sign(self.y_pred), y_true)

    def calc_sortino_ratio(self, y_true: np.ndarray) -> np.ndarray:

        rets: np.ndarray = self.calc_returns(y_true)
        self.sortino: float = ep.sortino_ratio(rets)
        return self.sortino

    def calc_sharpe_ratio(self, y_true: np.ndarray) -> np.ndarray:

        rets: np.ndarray = self.calc_returns(y_true)
        self.sharp: float = ep.sharpe_ratio(rets)
        return self.sharp

    def calc_calmar_ratio(self, y_true: np.ndarray) -> np.ndarray:

        rets: np.ndarray = self.calc_returns(y_true)
        self.calmar: float = ep.calmar_ratio(rets)
        return self.calmar

    def calc_accuracy(self, y_true: np.ndarray) -> np.ndarray:
        """收益率不是对数收益率 ACC"""

        rets: np.ndarray = self.calc_returns(y_true)
        self.accuracy: float = np.expm1(np.log1p(rets).sum())
        return self.accuracy

    def calc_mean_squared_error(self, y_true: np.ndarray) -> np.ndarray:

        self.MSE: float = -mean_squared_error(y_true, self.y_pred)
        return self.MSE


class Company(object):

    def __init__(self, trader_num: int,
                 A: List[Callable],
                 O: List[Callable],
                 stock_num: int,
                 M: int,
                 max_lag: int,
                 l: int,
                 time_window: int,
                 Q: float,
                 generate_method: str,
                 evaluation_method: str,
                 aggregate_method: str,
                 seed: int = None) -> None:

        self.trader_num = trader_num
        self.active_func = A
        self.binary_oper = O
        self.stock_num = stock_num
        self.M = M
        self.max_lag = max_lag
        self.l = l
        self.time_window = time_window
        self.Q = Q
        self.evaluation_method = evaluation_method
        self.generate_method = generate_method

        AGGREGATE_DICT: Dict = {'ALL': None,
                                'Q': 0.5,
                                'LINEAR': calc_least_squares}
        GENERATE_FUNC_DICT: Dict = {'GaussianMixture': self.gaussian_mixture_generate,
                                    'BayesianGaussianMixture': self.gaussian_mixture_generate,
                                    'Gaussian': self.gaussian_generate}

        self.EVALUATION_FUNC_DICT: Dict = {'ACC': 'calc_accuracy', 'SR': 'calc_sharpe_ratio',
                                           'CR': 'calc_calmar_ratio', 'STR': 'calc_sortino_ratio', 'MSE': 'calc_mean_squared_error'}

        self.generate_func: Callable = GENERATE_FUNC_DICT[generate_method]
        self.aggregate_method: Union[float,
                                     Callable] = AGGREGATE_DICT[aggregate_method.upper()]

        if seed:
            np.random.seed(seed)

        self.traders: List[Trader] = [
            Trader(A=A,
                   O=O,
                   stock_num=stock_num,
                   M=M,
                   max_lag=max_lag,
                   l=l) for _ in range(trader_num)
        ]

    # def fit(self, X: np.ndarray, y: np.ndarray) -> None:

    #     roll_data: np.ndarray = rolling_window(X, self.time_window)
    #     roll_y_true: np.ndarray = rolling_window(y, self.time_window)

    #     self.predictions: List = []

    #     Q = None
    #     func = None
    #     if isinstance(self.aggregate_method, Callable):

    #         func = self.aggregate_method

    #     if isinstance(self.aggregate_method, (int, float)):

    #         Q = self.aggregate_method

    #     for train_data, y_true in tqdm(zip(roll_data, roll_y_true), total=len(roll_data), desc='TC模型'):

    #         y_true = y_true[self.max_lag+self.l:]
    #         [trader.get_prediction(train_data) for trader in self.traders]

    #         self.educate(y_true)
    #         self.prune_and_generate(y_true)

    #         self.predictions.append(self.calc_aggregate(
    #             y_true, Q=Q, func=func))

    #     self.predictions: np.ndarray = np.squeeze(self.predictions)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        roll_data: np.ndarray = rolling_window(X, self.time_window)
        roll_y_true: np.ndarray = rolling_window(y, self.time_window)

        size: int = len(roll_data)
        chunk_size: int = calculate_best_chunk_size(size, CPU_WORKER_NUM)

        Q = None
        func = None
        if isinstance(self.aggregate_method, Callable):

            func = self.aggregate_method

        if isinstance(self.aggregate_method, (int, float)):

            Q = self.aggregate_method

        use_multiprocessing2func = functools.partial(
            self._use_multiprocessing2func, Q=Q, func=func)

        with Pool(processes=CPU_WORKER_NUM) as pool:

            predictions: List[np.ndarray] = list(tqdm(pool.imap(use_multiprocessing2func,
                                                                zip(roll_data,
                                                                    roll_y_true),
                                                                chunksize=chunk_size),
                                                      desc='TC模型',
                                                      total=size))

        self.predictions: np.ndarray = np.squeeze(predictions)

    def _use_multiprocessing2func(self, arr: Tuple[np.ndarray, np.ndarray], Q: float, func: Callable) -> np.ndarray:

        train_data, y_true = arr
        y_true = y_true[self.max_lag + self.l:]  # 获取真正的y_true
        [trader.get_prediction(train_data) for trader in self.traders]

        self.educate(y_true)
        self.prune_and_generate(y_true)

        return self.calc_aggregate(y_true, Q=Q, func=func)

    # TODO:Aggregate可以拆分成一个类单独做

    @property
    def aggregate(self):

        return np.nanmean(self.predictions, axis=1)

    def calc_aggregate(self, y_true: np.ndarray = None, Q: float = None, func: Callable = None) -> np.ndarray:
        """
        (1)N个交易员预测的平均值;
        (2)一段时间内预测准确率前50%的交易员的预测平均值;
        (3)一段时间内evaluation_method前50%的交易员预测平均值;
        # TODO:(2未完成)
        """
        if Q:
            pred_values: np.ndarray = self._get_pred_values(y_true)
            good_trader_flag: np.ndarray = self.find_traget_data(
                pred_values, Q, ">")
            return np.array([trader.y_pred[-1] if flag else np.nan for flag, trader in zip(good_trader_flag, self.traders)])

        elif func:

            pred_values: np.ndarray = self._get_pred_values(y_true)
            good_trader_flag: np.ndarray = self.find_traget_data(
                pred_values, self.Q, ">")

            prediction_matrix: np.ndarray = np.array(
                [trader.y_pred for flag, trader in zip(good_trader_flag, self.traders) if flag]).T

            return func(prediction_matrix, y_true)

        else:
            return np.array([trader.y_pred[-1] for trader in self.traders])

    def educate(self, y_true: np.ndarray, func: Callable = calc_least_squares) -> None:

        pred_values: np.ndarray = self._get_pred_values(y_true)
        bad_traders_flag: np.ndarray = self.find_traget_data(
            pred_values, self.Q, "<=")

        [trader.learn(y_true, func=func) for flag, trader in zip(
            bad_traders_flag, self.traders) if flag]

    def prune_and_generate(self, y_true: np.ndarray) -> None:

        pred_values: np.ndarray = self._get_pred_values(y_true)
        good_trader_flag: np.ndarray = self.find_traget_data(
            pred_values, self.Q, ">")

        self.generate_func(good_trader_flag)

    def gaussian_mixture_generate(self, good_trader_flag: np.ndarray) -> None:
        """高斯混合分布"""
        alpha_num, all_params = self._get_trader_info(good_trader_flag)

        gm_fit = GenerationFitParams(
            alpha_num, all_params, generate_method=self.generate_method)

        [trader.set_params(gm_fit.create_params(self.M)) for flag, trader in zip(
            good_trader_flag, self.traders) if not flag]

    def gaussian_generate(self, good_trader_flag: np.ndarray) -> None:
        """高斯分布"""
        [trader.reset_params() for flag, trader in zip(
            good_trader_flag, self.traders) if not flag]

    @staticmethod
    def find_traget_data(arr: np.ndarray, Q: float, operator: str) -> np.ndarray:

        func: Dict = {'<': np.less, '>': np.greater,
                      '>=': np.greater_equal, '<=': np.less_equal}

        theshold: float = np.quantile(arr, Q)

        return func[operator](arr, theshold)

    def _get_trader_info(self, trader_flag: np.ndarray) -> Tuple[np.ndarray]:

        alpha_num: List = []
        params: List = []
        for flag, trader in zip(trader_flag, self.traders):

            if flag:
                int_params = trader.int_params
                alpha_num.append(int_params.shape[0])
                params.append(int_params)

        return np.array(alpha_num), np.vstack(params)

    def _get_pred_values(self, y_true: np.ndarray) -> np.ndarray:

        method: str = self.EVALUATION_FUNC_DICT[self.evaluation_method]
        return np.array([getattr(trader, method)(y_true) for trader in self.traders])

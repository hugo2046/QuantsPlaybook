'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-08-15 09:13:32
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-08-24 14:37:51
Description:
'''
import functools
from collections import namedtuple
from typing import Callable, Dict, List, Tuple, Union

import empyrical as ep
import numpy as np
from scr.utils import calc_ols_func, core_formula, create_empty_lists, create_formulae
from sklearn.metrics import mean_squared_error


class Trader:
    def __init__(
        self,
        M: int,
        A: List[Callable],
        O: List[Callable],
        stock_num: int,
        time_window: int,
        max_lag: int = 9,
        seed: int = None,
    ) -> None:
        """构造$\Theta$
        $\Theta=\sum^{M}_{j}w_{j}A_{j}(O_{j}(r_{P_{j}}[t-D_{j}],r_{Q_{j}}[t-F_{j}]))$

        Args:
            M (int): 每位交易员表达式最大项数
            A (List[Callable]): 激活函数列表
            O (List[Callable]): 二元操作符函数列表
            stock_num (int): 股票个数
            time_window (int): 用于保持factor数组长度
            max_lag (int, optional): 数据延迟最大取值. Defaults to 9.
            l (int, optional): 交易延迟量,即观察到数据后不可能立马进行交易,需要等待l时间. Defaults to 1.
            seed (int, optional): 随机数种子. Defaults to None.
        """
        # 储存因子 下表为股票 结果为因子值
        self.factors: List[np.ndarray] = create_empty_lists(stock_num)
        self.time_window = time_window
        self.stock_num = stock_num  # 股票个数
        self.max_lag = max_lag
        self.M = M
        self.A = A
        self.O = O
        if seed:
            np.random.seed(seed)

        # 生成公式create_formulae
        # 下标对交易员对股票的预测
        self.original_formulae: List[namedtuple] = [
            create_formulae(
                M,
                A,
                O,
                stock_num=stock_num,
                max_lag=max_lag,
            ) for _ in range(stock_num)
        ]

        # 操作函数字典用于后续转换
        self.activation_funcs_dict: Dict = self._create_funcs2dict(A)
        self.binary_operators_dict: Dict = self._create_funcs2dict(O)

        self.int2activation_funcs_dict: Dict = {
            v: k
            for k, v in self.activation_funcs_dict.items()
        }
        self.int2binary_operators_funcs_dict: Dict = {
            v: k
            for k, v in self.binary_operators_dict.items()
        }

        # 获取公式
        # List中的下表对应的是数据中的股票
        self.formulae: List[List[Callable]] = [
            formula_info.forumlae for formula_info in self.original_formulae
        ]

        # 获取参数
        self.params: List = [
            formula_info.params for formula_info in self.original_formulae
        ]

        # 初始权重生成
        # List中的下表对应的是数据中的股票
        self.weight: List[np.ndarray] = [
            np.random.randn(len(formula)) for formula in self.formulae
        ]

    def get_params(self, stock_i: int) -> Union[np.ndarray, List]:
        """获取对应股票的公式参数
        
        当stock_i为all时获取全部股票的公式参数
        """
        def _transform_funcs2int(arr: np.ndarray) -> np.ndarray:

            arr[0] = self.activation_funcs_dict[arr[0]]
            arr[1] = self.binary_operators_dict[arr[1]]

            return arr

        if isinstance(stock_i, str):
            stock_i = stock_i.lower()

        if stock_i == 'all':
            return [
                np.apply_along_axis(_transform_funcs2int, 1, np.copy(param))
                for param in self.params
            ]

        return np.apply_along_axis(_transform_funcs2int, 1,
                                   np.copy(self.params[stock_i]))

    # 重置参数
    def reset_params(self,
                     stock_i: int,
                     unimodal: Tuple[Callable] = None) -> None:
        """重置对应股票的参数"""

        if unimodal:

            self._gaussian_mixture(stock_i, unimodal)

        else:

            self._gaussian_distributions(stock_i)

    # 批量计算factors
    def calc_bulk_factors(self, data: np.ndarray) -> None:
        """批量获取factors"""
        for stock_i, stock_i_formula in enumerate(self.formulae):

            # 第N个股票的M个因子均储存到p中 下标为M个因子
            p: np.ndarray = np.array(
                [formula(data) for formula in stock_i_formula]).flatten()

            # 储存N个股票的M个因子 下标为股票

            self._append_factors(p, stock_i)

    # 指定对应股票 计算其factors
    def calc_factors(self, data: np.ndarray, stock_i: int = None) -> None:
        """获取指定股票的factors

        factors (List[np.ndarray]):下标为对应的股票数据
        """

        p: np.ndarray = np.array(
            [formula(data) for formula in self.formulae[stock_i]]).flatten()

        self._append_factors(p, stock_i)

    def get_y_pred(self) -> np.ndarray:
        """结果为预测"""

        pred: List = [
            factor @ w for factor, w in zip(self.factors, self.weight)
        ]

        return np.array(pred)

    # 使用least squares method进行学习
    def learn(self,
              stock_i: int,
              endog: np.ndarray,
              func: Callable = calc_ols_func) -> None:
        """原始weight为随机生成 在learn方法中使用func函数生成一组新的weight"""

        # 回归结果从下标1开始去除偏置项
        # 更新交易员对应股票的的weight
        self.weight[stock_i] = func(self.factors[stock_i],
                                    endog,
                                    add_constant=True)[1:]

    def get_current_predict(self) -> np.ndarray:

        y_pred: List = [
            factor[-1] @ w for factor, w in zip(self.factors, self.weight)
        ]
        return np.array(y_pred)

    # TODO:构建Evaluation Protocols模块
    def evaluation(self,
                   y_true: np.ndarray,
                   evaluation_protocols: str = 'ACC') -> np.ndarray:

        evaluation_protocols: str = evaluation_protocols.upper()

        func_dic = {
            'ACC': self.calc_accuracy,
            'MSE': self.calc_mean_squared_error,
            'SR': self.calc_sharpe_ratio,
            'CR': self.calc_calmar_ratio,
            'STR': self.calc_sortino_ratio
        }
        return func_dic[evaluation_protocols](y_true)

    def calc_returns(self, y_true: np.ndarray) -> np.ndarray:

        y_pred: List[np.ndarray] = self.get_y_pred()
        sign: np.ndarray = np.sign(y_pred).T

        return np.multiply(sign, y_true)

    def calc_sortino_ratio(self, y_true: np.ndarray) -> np.ndarray:

        rets: np.ndarray = self.calc_returns(y_true)
        self.sortino: np.ndarray = ep.sortino_ratio(rets)
        return self.sortino

    def calc_sharpe_ratio(self, y_true: np.ndarray) -> np.ndarray:

        rets: np.ndarray = self.calc_returns(y_true)
        self.sharp: np.ndarray = ep.sharpe_ratio(rets)
        return self.sharp

    def calc_calmar_ratio(self, y_true: np.ndarray) -> np.ndarray:

        rets: np.ndarray = self.calc_returns(y_true)
        self.calmar: np.ndarray = np.apply_along_axis(ep.calmar_ratio, 0, rets)
        return self.calmar

    def calc_accuracy(self, y_true: np.ndarray) -> np.ndarray:
        """收益率不是对数收益率 ACC"""

        rets: np.ndarray = self.calc_returns(y_true)
        self.accuracy: np.ndarray = np.expm1(np.log1p(rets).sum())
        return self.accuracy

    def calc_mean_squared_error(self, y_true: np.ndarray) -> np.ndarray:
        """MSE"""
        y_pred: List[np.ndarray] = self.get_y_pred()
        y_true: List[np.ndarray] = np.hsplit(y_true, self.stock_num)

        self.mean_squared_error: np.ndarray = np.array(
            [mean_squared_error(a, b) for a, b in zip(y_true, y_pred)])

        return self.mean_squared_error

    def _parameter_bounds_check(self, parameter: np.ndarray,
                                m: int) -> np.ndarray:
        """参数边界检查"""
        # A
        parameter[:, 0] = np.clip(parameter[:, 0], 0, len(self.A) - 1)
        # O
        parameter[:, 1] = np.clip(parameter[:, 1], 0, len(self.O) - 1)
        # P
        parameter[:, 2] = np.clip(parameter[:, 2], 0, self.stock_num - 1)
        # Q
        parameter[:, 3] = np.clip(parameter[:, 3], 0, self.stock_num - 1)
        # F
        parameter[:, 4] = np.clip(parameter[:, 4], 1, self.max_lag - 1)
        # D
        parameter[:, 5] = np.clip(parameter[:, 5], 1, self.max_lag - 1)

        return parameter

    def _gaussian_mixture(self, stock_i: int,
                          unimodal: Tuple[Callable]) -> None:
        """高斯/贝叶斯混合分布
        在此方法下会将good trader参数拟合后生成新的参数
        Args:
            stock_i (int): _description_
            unimodal (Tuple[Callable]): _description_
        """
        def _transform_int2params(int_param: np.ndarray) -> np.ndarray:
            """将int参数转为func"""
            def _trans_int2func(arr: np.ndarray) -> np.ndarray:
                arr[0] = self.int2activation_funcs_dict[arr[0]]
                arr[1] = self.int2binary_operators_funcs_dict[arr[1]]

                return arr

            return np.apply_along_axis(_trans_int2func, 1,
                                       np.copy(int_param).astype('object'))

        factors_unimodal, m_unimodal = unimodal
        # 重置Alpha因子个数 最大限不能超过M
        m: float = np.rint(m_unimodal.sample()[0][0][0]).astype(int)
        m: float = np.clip(m, 1, self.M)
        # 重置参数
        params: np.ndarray = np.rint(factors_unimodal.sample(m)[0]).astype(int)
        # 修剪 params的区间范围
        params: np.ndarray = self._parameter_bounds_check(params, m)
        params: List = _transform_int2params(params).tolist()

        self.params[stock_i] = params
        self.factors[stock_i] = []
        self.weight[stock_i] = np.random.randn(m)

        self.formulae[stock_i] = [
            functools.partial(core_formula,
                              active_func=row[0],
                              binary_oper=row[1],
                              P=row[2],
                              Q=row[3],
                              F=row[4],
                              D=row[5],
                              max_lag=self.max_lag) for row in params
        ]

    def _gaussian_distributions(self, stock_i: int) -> None:
        """重置指定目标股票的公式
        
        使用高斯分布重置bad trader
        """

        # 生成公式
        formula: namedtuple = create_formulae(
            self.M,
            self.A,
            self.O,
            stock_num=self.stock_num,
            max_lag=self.max_lag,
        )

        # 更新公式
        self.formulae[stock_i] = formula.forumlae
        # 更新参数
        self.params[stock_i] = formula.params
        # 重置对应股票的factors
        self.factors[stock_i] = []

        self.weight[stock_i] = np.random.randn(len(formula.forumlae))

    def _append_factors(self, data: np.ndarray, stock_num: int):
        """添加factors数据到指定的stock容器中

        Parameters
        ----------
        data : np.ndarray
            需要添加的数据
        stock_num : int
            股票下标
        """
        factor = self.factors[stock_num]

        if len(factor) == 0:

            self.factors[stock_num] = data

        elif len(factor) > self.time_window:

            factor = np.vstack([factor, data])

            self.factors[stock_num] = factor[1:]

        else:

            self.factors[stock_num] = np.vstack([factor, data])

    @staticmethod
    def _create_funcs2dict(self) -> Dict:
        """将函数名转为数值,用于后续转换
        k-函数名 v-对应的数值
        """
        return dict(zip(self, np.arange(len(self))))
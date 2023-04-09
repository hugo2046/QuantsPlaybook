from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


from qlib.data.base import Expression, ExpressionOps


from .cyq import ChipFactor, calc_dist_chips
from .utils import rolling_frame


class FiveFeatureOperator(ExpressionOps):
    """构造筹码因子的五个特征的操作符

    Parameters
    ----------
    feature_a: close
    feature_b: high
    feature_c: low
    feature_d: vol
    feature_e: turnover
    N:N日区间
    method:分布的方法 triang or uniform or turnover_coeff
    factor:筹码因子的名称 CYQK_C,CKDW,PRP,ASR

    Returns
    ----------
    Feature:
        five features' operation output
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
        method: str,
        factor_name: str,
    ):
        self.feature_a = feature_a
        self.feature_b = feature_b
        self.feature_c = feature_c
        self.feature_d = feature_d
        self.feature_e = feature_e
        self.N = N
        self.method = method
        self.factor_name = factor_name

    def __str__(self):
        return f"{type(self).__name__}({self.feature_a},{self.feature_b},{self.feature_c},{self.feature_d},{self.feature_e},{self.N})"

    def check_br(self, feature: str):

        if isinstance(feature, (Expression,)):
            return feature.get_longest_back_rolling()
        else:
            return 0

    def get_longest_back_rolling(self):

        br_ls: List = [
            self.check_br(feature)
            for feature in {
                self.feature_a,
                self.feature_b,
                self.feature_c,
                self.feature_d,
                self.feature_e,
            }
        ]

        return np.max(br_ls)

    # 改写get_extended_window_size使其可以应对多个值的情况
    def get_extended_window_size(self):
        if isinstance(self.feature_a, (Expression,)):
            ll, lr = self.feature_a.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_b, (Expression,)):
            rl, rr = self.feature_b.get_extended_window_size()
        else:
            rl, rr = 0, 0

        if isinstance(self.feature_c, (Expression,)):
            cl, cr = self.feature_c.get_extended_window_size()
        else:
            cl, cr = 0, 0

        if isinstance(self.feature_d, (Expression,)):
            dl, dr = self.feature_d.get_extended_window_size()
        else:
            dl, dr = 0, 0

        if isinstance(self.feature_e, (Expression,)):
            el, er = self.feature_e.get_extended_window_size()
        else:
            el, er = 0, 0

        return max(ll, rl, cl, dl, el), max(lr, rr, cr, dr, er)


class CyqBasic(FiveFeatureOperator):
    """CyqBasic

    Parameters
    ----------


    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
        method: str,
        name: str,
    ):
        super(CyqBasic, self).__init__(
            feature_a, feature_b, feature_c, feature_d, feature_e, N, method, name
        )

    def _load_internal(self, instrument, start_index, end_index, *args):
        """
        To avoid error raised by bool type input, we transform the data into float32.
        """
        factor_name: str = f"get_{self.factor_name.lower()}"

        def _calc_factor(arr: pd.DataFrame) -> float:

            cumpdf: pd.Series = calc_dist_chips(arr, method=self.method)

            doc_factor: ChipFactor = ChipFactor(arr[-1, 0], cumpdf)

            return getattr(doc_factor, factor_name)()

        data: pd.DataFrame = pd.concat(
            (
                feature.load(instrument, start_index, end_index, *args)
                for feature in (
                    self.feature_a,
                    self.feature_b,
                    self.feature_c,
                    self.feature_d,
                    self.feature_e,
                )
            ),
            axis=1,
        )
        # 必须保证位置顺序
        data: pd.DataFrame = data[["$close", "$high", "$low", "$vol", "$turnover_rate"]]

        # TODO:  More precision types should be configurable
        data:pd.DataFrame = data.astype(np.float32)
        idx:pd.DatetimeIndex = data.index # 获取时间索引
        data:pd.DataFrame = data.dropna()

        if data.empty:
            return pd.Series([np.nan] * len(idx), index=idx)
        
        idx:pd.DatetimeIndex = data.index # 重新获取时间索引
        
        if len(data) < self.N:
            return pd.Series(_calc_factor(data.values), index=[idx[-1]])

        dfs: np.ndarray = rolling_frame(data, self.N)
        idx: pd.Index = idx[self.N - 1 :]
        return pd.Series([_calc_factor(df) for df in dfs], index=idx).sort_index()


#################### CYQK_C ####################
class CYQK_C_T(CyqBasic):
    """CYQK_C_T

    CYQK_C 因子使用三角分布计算
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(CYQK_C_T, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "triang",
            "CYQK_C",
        )


class CYQK_C_U(CyqBasic):
    """CYQK_C_U

    CYQK_C 因子使用均匀分布计算
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(CYQK_C_U, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "uniform",
            "CYQK_C",
        )


class CYQK_C_TN(CyqBasic):
    """CYQK_C_TN

    CYQK_C_TN 因子使用历史换手率半衰期
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(CYQK_C_TN, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "turn_coeff",
            "CYQK_C",
        )


#################### ASR ####################
class ASR_T(CyqBasic):
    """ASR_T

    AR_T 因子使用三角分布计算
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(ASR_T, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "triang",
            "ASR",
        )


class ASR_U(CyqBasic):
    """ASR_U

    ASR_U 因子使用均匀分布计算
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(ASR_U, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "uniform",
            "ASR",
        )


class ASR_TN(CyqBasic):
    """ASR_TN

    ASR_TN 因子使用历史换手率半衰期
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(ASR_TN, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "turn_coeff",
            "ASR",
        )


#################### CKDW ####################
class CKDW_T(CyqBasic):
    """CKDW_T

    AR_T 因子使用三角分布计算
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(CKDW_T, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "triang",
            "CKDW",
        )


class CKDW_U(CyqBasic):
    """CKDW_U

    CKDW_U 因子使用均匀分布计算
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(CKDW_U, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "uniform",
            "CKDW",
        )


class CKDW_TN(CyqBasic):
    """CKDW_TN

    CKDW_TN 因子使用历史换手率半衰期
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(CKDW_TN, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "turn_coeff",
            "CKDW",
        )


#################### PRP ####################
class PRP_T(CyqBasic):
    """PRP_T

    PRP_T 因子使用三角分布计算
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(PRP_T, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "triang",
            "PRP",
        )


class PRP_U(CyqBasic):
    """PRP_U

    PRP_U 因子使用均匀分布计算
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(PRP_U, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "uniform",
            "PRP",
        )


class PRP_TN(CyqBasic):
    """CKDW_TN

    CKDW_TN 因子使用历史换手率半衰期
    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        feature_c: str,
        feature_d: str,
        feature_e: str,
        N: int,
    ):
        super(PRP_TN, self).__init__(
            feature_a,
            feature_b,
            feature_c,
            feature_d,
            feature_e,
            N,
            "turn_coeff",
            "PRP",
        )

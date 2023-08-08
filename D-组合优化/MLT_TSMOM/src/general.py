from typing import Dict, List, Tuple, Union

import empyrical as ep
import numpy as np
import pandas as pd
import torch

N: int = 252


# def rolling_windows_torch(arr: torch.Tensor, window: int) -> torch.Tensor:
#     if window > arr.shape[0]:
#         raise ValueError(
#             "Specified `window` length of {0} exceeds length of"
#             " `arr`, {1}.".format(window, arr.shape[0])
#         )
#     if arr.ndim == 1:
#         arr: torch.Tensor = arr.reshape(-1, 1)

#     shape: Tuple = (arr.shape[0] - window + 1, window) + arr.shape[1:]
#     strides: Tuple = (arr.stride(0),) + arr.stride()
#     windows: torch.Tensor = torch.as_strided(arr, size=shape, stride=strides).squeeze()

#     if windows.ndim == 1:
#         windows: torch.Tensor = windows.unsqueeze(0)
#     return windows

###############################################################################################
#                                 损失函数
###############################################################################################
import empyrical as ep


def calc_sharpe_ratio(returns: torch.Tensor, N: int = 252) -> torch.float32:
    # 计算returns的均值和标准差
    mean_returns: torch.float32 = torch.nanmean(returns)
    std_returns: torch.float32 = torch.std(returns)
    if torch.all(std_returns == 0):
        return 0
    # 计算相关系数的中间结果
    # out: torch.float32 = torch.empty(std_returns.shape)
    ann_factor: torch.float32 = torch.sqrt(torch.tensor(252.0))
    return torch.multiply(torch.div(mean_returns, std_returns), ann_factor)


def calc_corrcoef(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """计算三维张量的相关系数

    Parameters
    ----------
    A : torch.Tensor
        1维为时序长度
        2维为特征数
        3维为标的数
    B : torch.Tensor
        1维为时序长度
        2维为特征数
        3维为标的数

    Returns
    -------
    torch.Tensor
        标的相关系数的均值
    """
    # 计算三维张量的相关系数
    # Calculate the mean of each row
    mean_A: torch.Tensor = A.mean(dim=0)
    mean_B: torch.Tensor = B.mean(dim=0)

    # Subtract the mean from each row
    A_centered: torch.Tensor = A - mean_A
    B_centered: torch.Tensor = B - mean_B

    # Calculate the covariance and standard deviations along the first dimension (400 samples)
    cov_matrix: torch.Tensor = torch.einsum("ijk,ilk->jlk", A_centered, B_centered) / (
        A.size(0) - 1
    )
    std_A: torch.Tensor = A_centered.std(dim=0)
    std_B: torch.Tensor = B_centered.std(dim=0)

    # Calculate the correlation coefficients for each pair of columns in A and B
    corrcoef: torch.Tensor = cov_matrix / (std_A * std_B)

    return corrcoef.diagonal().mean(dim=0)


def share_loss(
    weight: torch.Tensor,
    returns: torch.Tensor,
    target_vol: float,
    transaction_cost: float,
) -> torch.Tensor:
    # 论文给我的感觉像是双重差分
    # 但是感觉双重差分不太对,差分才像变动变化情况
    delta_weight: torch.Tensor = torch.diff(weight, dim=0)
    rr: torch.Tensor = (weight * returns)[1:] - torch.abs(
        delta_weight
    ) * transaction_cost
    # TODO:并未考虑目标波动率
    avg_rr: torch.Tensor = rr.mean(dim=1) * target_vol

    return -calc_sharpe_ratio(avg_rr)


def corrcoef_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    corrcoef: torch.Tensor = calc_corrcoef(y_pred, y_true)
    return -corrcoef.sum()


###############################################################################################
#                                 特征计算
###############################################################################################


def calc_volatility(cprice: np.ndarray, n: int = N, *args, **kwarys) -> np.float32:
    if "method" in kwarys:
        if "oprice" not in kwarys:
            raise ValueError("oprice is not in kwarys")
        method = kwarys["method"]
        cprice: np.ndarray = kwarys["oprice"] if method == "OTO" else cprice
    ln_cc: np.ndarray = np.log(cprice[1:] / cprice[:-1])
    return ln_cc.std(ddof=1) * np.sqrt(n)


def calc_parkinson_volatility(
    hprice: np.ndarray, lprice: np.ndarray, n: int = N, *args, **kwargs
) -> np.float32:
    a: float = 1 / (4 * np.log(2))
    b: np.ndarray = np.mean(np.square(np.log(hprice / lprice)))
    return np.sqrt(a * b) * np.sqrt(n)


def calc_garmanklass_volatility(
    hprice: np.ndarray,
    lprice: np.ndarray,
    cprice: np.ndarray,
    oprice: np.ndarray,
    n: int = N,
    *args,
    **kwargs,
) -> np.float32:
    ln_hl: np.ndarray = np.log(hprice / lprice)
    ln_co: np.ndarray = np.log(cprice / oprice)

    rs: np.ndarray = 0.5 * np.square(ln_hl) - (2 * np.log(2) - 1) * np.square(ln_co)

    return np.sqrt(rs.mean()) * np.sqrt(n)


def calc_rogers_satchell_volatility(
    hprice: np.ndarray,
    lprice: np.ndarray,
    cprice: np.ndarray,
    oprice: np.ndarray,
    n: int = N,
    *args,
    **kwargs,
) -> np.float32:
    ln_ho: np.ndarray = np.log(hprice / oprice)
    ln_co: np.ndarray = np.log(cprice / oprice)
    ln_lo: np.ndarray = np.log(lprice / oprice)

    rs: np.ndarray = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)

    return np.sqrt(rs.mean()) * np.sqrt(n)


def calc_yangzhang_volatility(
    hprice: np.ndarray,
    lprice: np.ndarray,
    cprice: np.ndarray,
    oprice: np.ndarray,
    n: int = N,
    *args,
    **kwargs,
) -> np.float32:
    ln_ho: np.ndarray = np.log(hprice / oprice)
    ln_co: np.ndarray = np.log(cprice / oprice)
    ln_lo: np.ndarray = np.log(lprice / oprice)

    N: int = len(hprice)
    k: np.float16 = 0.34 / (1.34 + (N + 1) / (N - 1))
    std_oc: np.ndarray = np.log(oprice[1:] / cprice[:-1]).var(ddof=1)
    std_co: np.ndarray = np.log(cprice[1:] / oprice[:-1]).var(ddof=1) * k

    rs: np.ndarray = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)
    vrs: np.float16 = np.mean(rs)

    return np.sqrt(std_oc + std_co + (1 - k) * vrs) * np.sqrt(n)


###############################################################################################
#                                 滚动窗口
###############################################################################################


def rolling_windows(
    df: Union[pd.DataFrame, pd.Series, np.ndarray], window: int
) -> List[np.ndarray]:
    if window > df.shape[0]:
        raise ValueError(
            "Specified `window` length of {0} exceeds length of"
            " `a`, {1}.".format(window, df.shape[0])
        )
    if isinstance(df, (pd.Series, pd.DataFrame)):
        df: np.ndarray = df.values
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    shape: int = (df.shape[0] - window + 1, window) + df.shape[1:]
    strides: int = (df.strides[0],) + df.strides
    windows: List[np.ndarray] = np.squeeze(
        np.lib.stride_tricks.as_strided(df, shape=shape, strides=strides)
    )
    # In cases where window == len(a), we actually want to "unsqueeze" to 2d.
    #     I.e., we still want a "windowed" structure with 1 window.
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


###############################################################################################
#                                 主要
###############################################################################################


def get_estimator(
    df: pd.DataFrame,
    window: int,
    estimator: str,
    method: str = None,
    usedf: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """计算各种波动率

    Parameters
    ----------
    df : pd.DataFrame
        index-datetime columns-high,low,close,open
    window : int
        滚动窗口期
    estimator : str
        波动率名称
            1.Raw-简单波动率
            2.Parkinson-帕金森波动率
            3.GarmanKlass-加曼克拉斯波动率
            4.RogersSatchell-罗杰斯萨切尔波动率
            5.YangZhang-杨张波动率
    method : str, optional
        近对Raw有用,可选CTC,OTO,None, by default None
            CTC-使用收盘价计算波动率
            OTO-使用开盘价计算波动率
    usedf : bool, optional
        True时返回df类型, by default False

    Returns
    -------
    Union[np.ndarray,pd.DataFrame]
    """
    ESTIMATORS: Dict = {
        "GarmanKlass": calc_garmanklass_volatility,
        "Parkinson": calc_parkinson_volatility,
        "Raw": calc_volatility,
        "RogersSatchell": calc_rogers_satchell_volatility,
        "YangZhang": calc_yangzhang_volatility,
    }

    df: pd.DataFrame = df[["high", "low", "close", "open"]].astype(np.float32)

    dfs: List[np.ndarray] = rolling_windows(df, window=window)
    func = ESTIMATORS[estimator]
    arr: np.ndarray = np.array(
        [
            func(
                hprice=df[:, 0],
                lprice=df[:, 1],
                cprice=df[:, 2],
                oprice=df[:, 3],
                method=method,
            )
            for df in dfs
        ]
    )
    if usedf:
        return pd.DataFrame(
            arr, index=df.index[window - 1 :], columns=[f"{estimator}_{window}"]
        )
    else:
        return arr


def generate_features(price: pd.DataFrame, method: str = None) -> pd.DataFrame:
    """生成论文所需特征即预测目标

    Parameters
    ----------
    price : pd.DataFrame
        index-date columns-high,low,close,open
    method : str, optional
        仅控制Raw波动率的计算方法, by default None
        默认为CTC
        当method为OTO时,使用开盘价计算Raw波动率
        当method为CTC时,使用收盘价计算Raw波动率
    Returns
    -------
    pd.DataFrame
        _description_
    """
    if method is None:
        method: str = "CTC"

    field: str = {"OTO": "open", "CTC": "close"}[method]
    hist_periods: set = {5, 21, 63, 126, 252}

    def generate_vol(estimator: str) -> pd.DataFrame:
        return pd.concat(
            (
                get_estimator(price, i, estimator, method=method, usedf=True)
                for i in hist_periods
            ),
            axis=1,
        )

    def generate_ln_return(periods: Tuple) -> pd.DataFrame:
        return pd.concat(
            (
                np.log(price[field] / price[field].shift(i)).to_frame(
                    f"ln_return_{str(i)}"
                )
                for i in periods
            ),
            axis=1,
        )

    Parkinson: pd.DataFrame = get_estimator(price, 21, "Parkinson", usedf=True)
    GarmanKlass: pd.DataFrame = get_estimator(price, 21, "GarmanKlass", usedf=True)
    RogersSatchell: pd.DataFrame = get_estimator(
        price, 21, "RogersSatchell", usedf=True
    )
    YangZhang: pd.DataFrame = get_estimator(price, 21, "YangZhang", usedf=True)
    Raw: pd.DataFrame = generate_vol("Raw")
    ln_return: pd.DataFrame = generate_ln_return((1, 5, 21, 126, 252))

    rolling_vol: pd.DataFrame = Raw.rolling(21).std(ddof=0)
    rolling_vol: pd.DataFrame = rolling_vol.add_suffix("_avg")

    vol_df: pd.DataFrame = pd.concat(
        (Raw, Parkinson, GarmanKlass, RogersSatchell, YangZhang), axis=1
    )
    return pd.concat((ln_return, rolling_vol, vol_df), axis=1)


def get_backtest_metrics(rr: np.ndarray, periods: str = "daily") -> pd.Series:
    metrics: Dict = {
        "Annualized Return": ep.annual_return(rr, period=periods),
        "Cumulative Return": ep.cum_returns_final(rr),
        "Annualized Sharpe Ratio": ep.sharpe_ratio(rr, period=periods),
        "Annualized Sortino Ratio": ep.sortino_ratio(rr, period=periods),
        "Max Drawdown": ep.max_drawdown(rr),
        "Annualized volatility": ep.annual_volatility(rr, period=periods),
    }

    return pd.Series(metrics).to_frame("metrics")


def calc_strategy_returns(weight: torch.Tensor, next_ret: torch.Tensor) -> np.ndarray:
    weights_np, next_ret_np = weight.cpu().numpy(), next_ret.cpu().numpy()
    rr: np.ndarray = (weights_np * next_ret_np).mean(axis=1)

    return rr


def get_strategy_returns(
    weights: torch.Tensor, next_ret: torch.Tensor, index: pd.DatetimeIndex
) -> pd.DataFrame:
    rr_ser: pd.Series = pd.Series(
        data=calc_strategy_returns(weights, next_ret), index=index
    )
    rr_ser.name = "MTL_TSMOM"
    benchmark: pd.Series = pd.Series(
        data=next_ret.cpu().numpy().mean(axis=1), index=index
    )
    benchmark.name = "Benchmark"

    return pd.concat((rr_ser, benchmark), axis=1)

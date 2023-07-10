from functools import lru_cache
from typing import Dict, Union

import numpy as np
import pandas as pd


def get_coins_team(
    baseline_df: pd.DataFrame, factor_df: pd.DataFrame, opr: str = "lt"
) -> pd.DataFrame:
    """比较baseline与baseline截面均值的关系
    关系根据opr参数确定,生成的布尔值矩阵再乘以factor_df

    Parameters
    ----------
    baseline_df : pd.DataFrame
        比较基准
    factor_df : pd.DataFrame
        因子值
    opr : str, optional
        gt(>);lt(<), by default 'lt'

    Returns
    -------
    pd.DataFrame
        index-date columns-code value-factor
    """
    cross_avg: pd.Series = baseline_df.mean(axis=1)
    # 研报中多为 因子<截面均值 则为硬币型股票 因子值*-1 
    # 即：sign(因子 - 截面均值) * 因子
    # 意外收获是波动率反转类 仅标记波动率小于截面均值的 其他标记在0时效果会很好
    # getattr(baseline_df, opr)(cross_avg, axis=0)
    reversed: int = {"lt": 1, "gt": -1}[opr]
    coins: pd.DataFrame = (
        baseline_df.sub(cross_avg, axis=0).mul(reversed).apply(lambda x: np.sign(x))
    )

    return factor_df * coins


def check_data_cols(df: pd.DataFrame) -> None:
    """检查传入的数据是否具备计算因子所需的列

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex level0-date level1-code columns-close open turnover_rate turnover_rate_f

    Raises
    ------
    ValueError
        如果不存在则报错
    """
    for col in ["close", "open", "turnover_rate", "turnover_rate_f"]:
        if col not in df.columns:
            raise ValueError(f"{col}不在df中!")


class SprotBettingsFactorBase(object):

    created_by = "DataFrame"

    def __init__(
        self,
        data: pd.DataFrame,
        index_name: str = "datetime",
        columns_name="instrument",
    ) -> pd.DataFrame:
        check_data_cols(data)
        self.data = data
        self.columns_name = columns_name
        self.index_name = index_name

    @lru_cache
    def _calc_overnight_distance(self) -> pd.DataFrame:
        """隔夜涨跌幅的市场平均水平是最平静的，因此我们计算每只个股的隔夜涨跌幅与市场平均水平的差值，然后取绝对值
        表示这只个股与“最平静”之间的距离，并将其记为“隔夜距离”因子

        Returns
        -------
        pd.DataFrame
            index-date columns-code value-factor
        """
        overnight_ret: pd.DataFrame = self._calc_overnight_ret()

        return overnight_ret.sub(overnight_ret.mean(axis=1), axis=0).abs()

    @lru_cache
    def _calc_overnight_ret(self) -> pd.DataFrame:
        """计算隔夜收益

        Returns
        -------
        pd.DataFrame
            index-date columns-code value-factor
        """
        close_df: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values="close",
        )

        open_df: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values="open",
        )
        # 使用t日开盘价除以t-1日的收盘价再减 1
        return open_df / close_df.shift(1) - 1

    @lru_cache
    def _calc_intreday_ret(self) -> pd.DataFrame:
        """计算日间收益率

        Returns
        -------
        pd.DataFrame
            index-date columns-code value-factor
        """
        price: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values="close",
        )
        return price.pct_change(fill_method=None)

    @lru_cache
    def _calc_intraday_ret(self) -> pd.DataFrame:
        """计算日内收益

        Returns
        -------
        pd.DataFrame
            index-date columns-code value-factor
        """
        close_df: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values="close",
        )

        open_df: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values="open",
        )

        return close_df.div(open_df) - 1

    def _get_turnover_rate(self, field: str, offset: int = None) -> pd.DataFrame:
        """获取换手率

        Parameters
        ----------
        field : str
            turnover_rate,turnover_rate_f
        offset : int, optional
            如果无则不偏移, by default None

        Returns
        -------
        pd.DataFrame
            index-date columns-code value-factor
        """
        turnover: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values=field,
        )
        return turnover if offset is None else turnover.shift(offset)

    def _get_returns(self, type_of_return: str) -> pd.DataFrame:
        """获取收益类型"""
        returns_func_dict: Dict = {
            "interday": self._calc_intreday_ret,
            "intraday": self._calc_intraday_ret,
            "overnight": self._calc_overnight_ret,
        }
        return returns_func_dict[type_of_return]()

    def create_volatility_reverse(
        self, type_of_return: str, window: int = 20, opr: str = "lt"
    ) -> pd.DataFrame:
        """
        type_of_return:str
            interday,intraday,overnight
        windiw:int
            计算窗口
        opr:str
            根据研报默认为lt,gt(>);lt(<)
        -------------
        计算逻辑:
            step1:每月月底计算最近20天的日间收益率的均值和标准差,作为当月的"日间收益率"和"日间波动率";
            step2:比较每只股票的日间波动率与市场截面均值的大小关系,将[日间波动率]"小于"市场均值的股票,视为"硬币"型股票
            由于未来其发生动量效应的概率更大,因此我们将其当[月日间收益率]乘以-1
        """
        pct_chg: pd.DataFrame = self._get_returns(type_of_return)
        avg_ret: pd.DataFrame = pct_chg.rolling(window).mean(
            engine="numba", engine_kwargs={"parallel": True}
        )
        std_df: pd.DataFrame = pct_chg.rolling(window).std(
            engine="numba", engine_kwargs={"parallel": True}
        )
        factor_df: pd.DataFrame = get_coins_team(std_df, avg_ret, opr)
        return factor_df

    def create_turnover_reverse(
        self,
        field: str,
        type_of_return: str,
        window: int = 20,
        offset: int = None,
        opr: str = "lt",
    ) -> pd.DataFrame:
        """
        field:str
            turnover_rate,turnover_rate_f
        type_of_return:str
            interday,intraday,overnight
        windiw:int
            计算窗口
        offset:int
            如果无则不偏移
        opr:str
            根据研报默认为lt,gt(>);lt(<)
        -------------
        计算逻辑:
            step1: t日换手率与t-1日换手率的差值,作为t日换手率的变化量;
            step2: [换手率变化量]低于市场均值的,为"硬币"股票,t日的日间收益率,将"硬币"型股票的日间收益率乘以-1;
            step3: 计算最近20天的"翻转收益率"的均值作为因子值
        """
        if field is None:
            raise ValueError("field must be given")
        pct_chg: pd.DataFrame = self._get_returns(type_of_return)
        turnover: pd.DataFrame = self._get_turnover_rate(field, offset)
        diff_turnover: pd.DataFrame = turnover - turnover.shift(1)
        # engine_kwargs={'parallel': True}
        factor_df: pd.DataFrame = (
            get_coins_team(diff_turnover, pct_chg, opr)
            .rolling(window)
            .mean(engine="numba", engine_kwargs={"parallel": True})
        )
        return factor_df

    def get_factor(
        self,
        type_of_factor: str = None,
        type_of_return: str = None,
        window: int = 20,
        usedf: bool = False,
        method: str = None,
        *args,
        **kwargs,
    ) -> Union[pd.DataFrame, pd.Series]:
        """获取因子

        Parameters
        ----------
        type_of_factor : str
            因子类型 interday,intraday,overnight
            当method不为None时,无效
        type_of_return : str
            收益类型 volatility,turnover
        window : int, optional
            计算窗口, by default 20
        usedf : bool, optional
            True返回为pd.DataFrame;False返回为pd.Series, by default False
        method : str, optional
            修正方法, by default None
        Returns
        -------
        pd.DataFrame
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        def _trans2ser(
            df: pd.DataFrame, type_of_return: str, type_of_factor: str
        ) -> pd.Series:
            """转换df为ser并设置name"""
            if usedf:
                return df.sort_index()
            ser: pd.Series = df.stack()
            ser.name = f"{type_of_return}_{type_of_factor}_reverse"
            return ser.sort_index()

        def _check_kwargs(**kwargs) -> None:
            """检查kwargs"""
            offset = kwargs.get("offset", None)
            field = kwargs.get("field", None)
            if field is None:
                raise ValueError("field must be given")
            return field, offset

        if len(self.data) < window:
            raise ValueError("window must be less than data length")

        if method is None:
            if type_of_factor is None:
                raise ValueError("type_of_factor must be given if method is None")
            if type_of_factor == "volatility":
                factor_df: pd.DataFrame = self.create_volatility_reverse(
                    type_of_return, window
                )
                return _trans2ser(factor_df, type_of_return, type_of_factor)

            else:
                field, offset = _check_kwargs(**kwargs)
                factor_df: pd.DataFrame = self.create_turnover_reverse(
                    field, type_of_return, window, offset
                )
                return _trans2ser(factor_df, type_of_return, field)

        else:
            field, offset = _check_kwargs(**kwargs)
            factor_df: pd.DataFrame = self.create_turnover_reverse(
                field, type_of_return, window, offset
            ) + self.create_volatility_reverse(type_of_return, window)
            factor_df *= 0.5
            if field == "turnover_rate_f":
                type_of_return: str = f"{type_of_return}_f"
            return _trans2ser(factor_df, "revise", type_of_return)


class SportBettingsFactor(SprotBettingsFactorBase):
    def __init__(
        self,
        data: pd.DataFrame,
        index_name: str = "datetime",
        columns_name="instrument",
    ) -> None:
        super().__init__(data, index_name, columns_name)
        self.data = data
        self.columns_name = columns_name
        self.index_name = index_name

    def interday_volatility_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """日间反转-波动翻转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        return self.get_factor("volatility", "interday", window, usedf)

    def intraday_volatility_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """日内反转-波动翻转

        Parameters
        ----------
        window : int, optional
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor("volatility", "intraday", window, usedf)

    def overnight_volatility_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """隔夜反转-波动翻转

        Parameters
        ----------
        window : int, optional
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor("volatility", "overnight", window, usedf)

    def interday_turnover_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """日间反转-换手率翻转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor(
            "turnover", "interday", window, usedf, field="turnover_rate"
        )

    def intraday_turnover_reverse(
        self, window: int, usedf: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """日内反转-换手翻转

        Parameters
        ----------
        window : int
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.DataFrame,pd.Series]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        return self.get_factor(
            "turnover", "intraday", window, usedf, field="turnover_rate"
        )

    def overnight_turnover_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """隔夜反转-换手翻转

        Parameters
        ----------
        window : int
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.DataFrame,pd.Series]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor(
            "turnover", "overnight", window, usedf, field="turnover_rate"
        )

    def interday_turnover_f_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """日间反转-换手率翻转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor(
            "turnover", "interday", window, usedf, field="turnover_rate_f"
        )

    def intraday_turnover_f_reverse(
        self, window: int, usedf: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """日内反转-换手翻转

        Parameters
        ----------
        window : int
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.DataFrame,pd.Series]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        return self.get_factor(
            "turnover", "intraday", window, usedf, field="turnover_rate_f", offset=1
        )

    def overnight_turnover_f_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """隔夜反转-换手翻转

        Parameters
        ----------
        window : int
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.DataFrame,pd.Series]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor(
            "turnover", "overnight", window, usedf, field="turnover_rate_f", offset=1
        )

    def revise_interday_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """修正日间反转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        return self.get_factor(
            type_of_return="interday",
            window=window,
            usedf=usedf,
            method="revise",
            field="turnover_rate",
        )

    def revise_intraday_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """修正日内反转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor(
            type_of_return="intraday",
            window=window,
            usedf=usedf,
            method="revise",
            field="turnover_rate",
        )

    def revise_overnight_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """修正隔夜反转

        Parameters
        ----------
        window : int, optional
            滚动窗口期, by default 20
        method : int, optional
                by default 1
            1 - turnover_rate
            2 - turnover_rate_f 自由流通换手率
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor(
            type_of_return="overnight",
            window=window,
            usedf=usedf,
            method="revise",
            field="turnover_rate",
        )

    def revise_interday_f_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """修正日间反转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        return self.get_factor(
            type_of_return="interday",
            window=window,
            usedf=usedf,
            method="revise",
            field="turnover_rate_f",
        )

    def revise_intraday_f_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """修正日内反转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor(
            type_of_return="intraday",
            window=window,
            usedf=usedf,
            method="revise",
            field="turnover_rate_f",
        )

    def revise_overnight_f_reverse(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """修正隔夜反转

        Parameters
        ----------
        window : int, optional
            滚动窗口期, by default 20
        method : int, optional
                by default 1
            1 - turnover_rate
            2 - turnover_rate_f 自由流通换手率
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """

        return self.get_factor(
            type_of_return="overnight",
            window=window,
            usedf=usedf,
            method="revise",
            field="turnover_rate_f",
        )

    def coin_team(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        df: pd.DataFrame = (
            self.revise_interday_reverse(window, usedf)
            + self.revise_intraday_reverse(window, usedf)
            + self.revise_overnight_reverse(window, usedf)
        )
        df.name = "coin_team"
        return df

    def coin_team_f(
        self, window: int = 20, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        df: pd.DataFrame = (
            self.revise_interday_f_reverse(window, usedf)
            + self.revise_intraday_f_reverse(window, usedf)
            + self.revise_overnight_f_reverse(window, usedf)
        )

        df.name = "coin_team_f"
        return df

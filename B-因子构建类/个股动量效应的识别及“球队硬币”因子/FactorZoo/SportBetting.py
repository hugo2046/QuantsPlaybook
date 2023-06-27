import pandas as pd
from typing import Union, Dict, List


def get_coins_team(
    baseline_df: pd.DataFrame, factor_df: pd.DataFrame, opr: str = "gt"
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    baseline_df : pd.DataFrame
        _description_
    factor_df : pd.DataFrame
        _description_
    opr : str, optional
        gt(>);lt(<), by default 'gt'

    Returns
    -------
    pd.DataFrame
        _description_
    """
    cross_avg: pd.Series = baseline_df.mean(axis=1)
    coins: pd.DataFrame = getattr(baseline_df, opr)(cross_avg, axis=0) * -1
    return factor_df * coins


def check_data_cols(df: pd.DataFrame) -> bool:
    for col in ["close", "open", "turnover_rate", "turnover_rate_f"]:
        if col not in df.columns:
            raise ValueError(f"{col}不在df中!")


class SportBettingFactor(object):
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
        self.created_by = "DataFrame"

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

    @staticmethod
    def create_volatility_reverse(
        pct_chg: pd.DataFrame, window: int = 20, opr: str = "lt"
    ) -> pd.DataFrame:
        avg_ret: pd.DataFrame = pct_chg.rolling(window, min_periods=1).mean()
        std_df: pd.DataFrame = pct_chg.rolling(window, min_periods=1).std()
        factor_df: pd.DataFrame = (
            get_coins_team(std_df, avg_ret, opr).rolling(window).mean()
        )
        return factor_df

    @staticmethod
    def create_turnover_reverse(
        turnover: pd.DataFrame, pct_chg: pd.DataFrame, window: int = 20, opr: str = "lt"
    ) -> pd.DataFrame:
        diff_turnover: pd.DataFrame = turnover - turnover.shift(1)
        factor_df: pd.DataFrame = (
            get_coins_team(diff_turnover, pct_chg, opr).rolling(window).mean()
        )
        return factor_df

    def interday_volatility_reverse(
        self, window: int = 20, usedf: bool = False,**kwargs
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
        if len(self.data) < window:
            raise ValueError("window must be less than data length")

        price: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values="close",
        )
        pct_chg: pd.DataFrame = price.pct_change()
        factor_df: pd.DataFrame = self.create_volatility_reverse(pct_chg, window)
        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "interday_volatility_reverse"
        return ser

    def interday_turnover_reverse(
        self, window: int = 20, method: int = 1, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """日间反转-换手率翻转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        method: int, optional 1
            1 - turnover_rate
            2 - turnover_rate_f 自由流通换手率
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        if len(self.data) < window:
            raise ValueError("window must be less than data length")

        price: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values="close",
        )

        pct_chg: pd.DataFrame = price.pct_change()
        fields_dict: Dict = {1: "turnover_rate", 2: "turnover_rate_f"}
        turnover: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values=fields_dict[method],
        )

        factor_df: pd.DataFrame = self.create_turnover_reverse(
            turnover, pct_chg, window
        )

        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "interday_turnover_reverse"
        return ser

    def revise_interday_reverse(
        self, window: int = 20, method: int = 1, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """修正日间反转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        method: int, optional 1
            1 - turnover_rate
            2 - turnover_rate_f 自由流通换手率
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        if len(self.data) < window:
            raise ValueError("window must be less than data length")

        factor_df: pd.DataFrame = (
            self.interday_volatility_reverse(window, True)
            + self.interday_turnover(window, method, True)
        ) * 0.5
        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "revise_interday_reverse"
        return ser

    def intraday_volatility_reverse(
        self, window: int = 20, usedf: bool = False,**kwargs
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
        if len(self.data) < window:
            raise ValueError("window must be less than data length")

        intrady_ret: pd.DataFrame = self._calc_intraday_ret()
        factor_df: pd.DataFrame = self.create_volatility_reverse(intrady_ret, window)
        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "intraday_volatility_reverse"
        return ser

    def intraday_turnover_reverse(
        self, window: int, method: str, usedf: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """日内反转-换手翻转

        Parameters
        ----------
        window : int
            滚动窗口期, by default 20
        method : str,by default 1
            1 - turnover_rate
            2 - turnover_rate_f 自由流通换手率
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.DataFrame,pd.Series]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        if len(self.data) < window:
            raise ValueError("window must be less than data length")
        intrady_ret: pd.DataFrame = self._calc_intraday_ret()
        fields_dict: Dict = {1: "turnover_rate", 2: "turnover_rate_f"}
        turnover: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values=fields_dict[method],
        )

        # 换手率变化量低于截面均值为反转
        factor_df: pd.DataFrame = self.create_turnover_reverse(
            turnover, intrady_ret, window
        )
        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "intraday_turnover_reverse"
        return ser

    def revise_intraday_reverse(
        self, window: int = 20, method: int = 1, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """修正日内反转

        Parameters
        ----------
        window : int, optional 20
            滚动窗口期, by default 20
        method: int, optional 1
            1 - turnover_rate
            2 - turnover_rate_f 自由流通换手率
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.Series,pd.DataFrame]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        if len(self.data) < window:
            raise ValueError("window must be less than data length")

        factor_df: pd.DataFrame = (
            self.intraday_volatility_reverse(window, True)
            + self.intraday_turnover(window, method, True)
        ) * 0.5
        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "revise_intraday_reverse"
        return ser

    def overnight_volatility_reverse(
        self, window: int = 20, usedf: bool = False,**kwargs
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
        if len(self.data) < window:
            raise ValueError("window must be less than data length")
        overnight_ret: pd.DataFrame = self._calc_overnight_distance()

        factor_df: pd.DataFrame = self.create_volatility_reverse(overnight_ret, window)
        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "overnight_volatility_reverse"
        return ser

    def overnight_turnover_reverse(
        self, window: int = 20, method: int = 1, usedf: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """隔夜反转-换手翻转

        Parameters
        ----------
        window : int
            滚动窗口期, by default 20
        method : str,by default 1
            1 - turnover_rate
            2 - turnover_rate_f 自由流通换手率
        usedf : bool, optional
            为True时返回df否则为ser, by default False

        Returns
        -------
        Union[pd.DataFrame,pd.Series]
            df - index-date columns-code value-factor
            ser - MultiIndex level0-date level1-code value-factor
        """
        if len(self.data) < window:
            raise ValueError("window must be less than data length")
        overnight_ret: pd.DataFrame = self._calc_overnight_distance()
        fields_dict: Dict = {1: "turnover_rate", 2: "turnover_rate_f"}
        turnover: pd.DataFrame = pd.pivot_table(
            self.data.reset_index(),
            index=self.index_name,
            columns=self.columns_name,
            values=fields_dict[method],
        )

        # 换手率变化量低于截面均值为反转
        factor_df: pd.DataFrame = self.create_turnover_reverse(
            turnover.shift(1), overnight_ret, window
        )
        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "overnight_turnover_reverse"
        return ser

    def revise_overnight_reverse(
        self, window: int = 20, method: int = 1, usedf: bool = False
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
            _description_, by default False

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if len(self.data) < window:
            raise ValueError("window must be less than data length")

        factor_df: pd.DataFrame = (
            self.overnight_volatility_reverse(window, True)
            + self.overnight_turnover_reverse(window, method, True)
        ) * 0.5
        if usedf:
            return factor_df

        ser: pd.Series = factor_df.stack()
        ser.name = "revise_overnight_reverse"
        return ser

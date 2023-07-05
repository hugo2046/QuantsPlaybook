"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-04-04 10:49:17
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-07-04 10:41:12
Description: 
"""

from collections import namedtuple
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from alphalens.utils import quantize_factor


def clean_multiindex_columns(factor_data: pd.DataFrame) -> pd.DataFrame:
    """预处理因子数据,将多层索引转换为单层索引

    Args:
        factor_data (pd.DataFrame): MultiIndex level0:datetime level1:instrument MultiColumns level0:feature level1:label

    Returns:
        pd.DataFrame: MultiIndex level0:date level1:assert columns->factor next_ret
    """
    clean_factor: pd.DataFrame = factor_data.copy()
    if isinstance(clean_factor.columns, pd.MultiIndex):
        clean_factor.columns = clean_factor.columns.droplevel(0)

    clean_factor.index.names = ["datetime", "instrument"]

    return clean_factor


def get_factor_group_returns(
    factor: pd.DataFrame,
    groupby: Union[pd.Series, Dict] = None,
    binning_by_group: bool = False,
    quantiles: int = 5,
    bins: int = None,
    groupby_labels: Dict = None,
    max_loss: float = 0.35,
    zero_aware: bool = False,
) -> namedtuple:
    """获取单因子分组收益

    Parameters
    ----------
    factor : pd.DataFrame
        index - MultiIndex level0:datetime level1:instrument columns->factor_name|next_ret
    groupby : Union[pd.Series, Dict], optional
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data, by default None
    binning_by_group : bool, optional
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio, by default False
    quantiles : int, optional
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None, by default 5
    bins : int, optional
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None, by default None
    groupby_labels : Dict, optional
        A dictionary keyed by group code with values corresponding
        to the display name for each group, by default None
    max_loss : float, optional
         Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression., by default 0.35
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively., by default False

    Returns
    -------
    namedtuple
        factor_quantile 因子分组状况
        factor_return 因子收益率 columns MultiIndex level0:factor_name level1:factor_quantile index-date value-factor_return
    """
    res: namedtuple = namedtuple("Res", "factor_quantile,factor_return")
    # 获取因子名称
    sel_cols: List = [
        [col, "next_ret"] for col in factor.columns.tolist() if col != "next_ret"
    ]
    idx = factor.index

    return_dict: Dict = {}  # 储存因子收益率
    factor_quantile_dict: Dict = {}  # 储存因子分组

    for col in sel_cols:
        clean_factor: pd.DataFrame = get_clean_factor(
            factor[col],
            groupby=groupby,
            groupby_labels=groupby_labels,
            binning_by_group=binning_by_group,
            quantiles=quantiles,
            bins=bins,
            max_loss=max_loss,
            zero_aware=zero_aware,
        )

        clean_factor: pd.DataFrame = clean_factor.reindex(idx)
        return_dict[col[0]] = pd.pivot_table(
            clean_factor.reset_index(),
            index="datetime",
            columns="factor_quantile",
            values="next_ret",
        )
        factor_quantile_dict[f"{col[0]}_group"] = clean_factor['factor_quantile']

    return_df: pd.DataFrame = _clean_index_name(return_dict)
    quantile_df: pd.DataFrame = pd.concat(factor_quantile_dict, axis=1)
    return res(quantile_df, return_df)


def _clean_index_name(dfs: Dict) -> pd.DataFrame:
    df: pd.DataFrame = pd.concat(dfs, axis=1)
    df.index.names = ["date"]
    df.columns.names = ["factor_name", "group"]
    return df





def get_factor_describe(ser: pd.Series) -> pd.Series:
    """计算因子的一些统计指标

    Parameters
    ----------
    ser : pd.Series
        MultiIndex level0:date level1:assert columns->factor next_ret

    Returns
    -------
    pd.Series
    """
    dic: Dict = {"total": len(ser), "miss_value": ser.isna().sum()}
    dic["zero_value"] = (ser == 0).sum()
    dic["mean"] = ser.mean()
    dic["std"] = ser.std()
    dic["min"] = ser.min()
    dic["max"] = ser.max()
    dic["skew"] = ser.skew()
    dic["kurt"] = ser.kurt()
    dic["25%"] = ser.quantile(0.25)
    dic["75%"] = ser.quantile(0.75)
    dic["median"] = ser.median()
    dic["miss_value_ratio[%]"] = (dic["miss_value"] / dic["total"]) * 100
    dic["zero_value_ratio[%]"] = (dic["zero_value"] / dic["total"]) * 100
    return pd.Series(dic)


class MaxLossExceededError(Exception):
    pass


def get_clean_factor(
    factor: pd.DataFrame,
    groupby=None,
    binning_by_group=False,
    quantiles=5,
    bins=None,
    groupby_labels=None,
    max_loss=0.35,
    zero_aware=False,
):
    
    initial_amount: int = len(factor)

    factor_copy: pd.DataFrame = factor.copy()
    factor_copy.index = factor_copy.index.rename(["datetime", "instrument"])
    factor_copy: pd.DataFrame = factor_copy[np.isfinite(factor_copy)]

    def _get_factor_name(factor: pd.DataFrame) -> str:
        factor_name: List = [col for col in factor.columns if col != "next_ret"]
        if len(factor_name) > 1:
            raise ValueError("存在多个因子名称,请检查!")
        return factor_name[0]

    factor_name: str = _get_factor_name(factor_copy)

    if groupby is not None:
        if isinstance(groupby, dict):
            if diff := set(factor_copy.index.get_level_values("asset")) - set(
                groupby.keys()
            ):
                raise KeyError(f"Assets {list(diff)} not in group mapping")

            ss = pd.Series(groupby)
            groupby = pd.Series(
                index=factor_copy.index,
                data=ss[factor_copy.index.get_level_values("asset")].values,
            )

        if groupby_labels is not None:
            if diff := set(groupby.values) - set(groupby_labels.keys()):
                raise KeyError(f"groups {list(diff)} not in passed group names")

            sn: pd.Series = pd.Series(groupby_labels)
            groupby: pd.Series = pd.Series(
                index=groupby.index, data=sn[groupby.values].values
            )

        factor_copy["group"] = groupby.astype("category")

    factor_copy: pd.DataFrame = factor_copy.dropna()

    fwdret_amount: int = len(factor_copy.index)

    no_raise: bool = max_loss != 0
    quantile_data: pd.DataFrame = quantize_factor(
        factor_copy[factor_name],
        quantiles,
        bins,
        binning_by_group,
        no_raise,
        zero_aware,
    )

    factor_copy["factor_quantile"] = quantile_data

    factor_copy: pd.DataFrame = factor_copy.dropna()

    binning_amount: int = len(factor_copy)

    tot_loss: float = (initial_amount - binning_amount) / initial_amount
    fwdret_loss: float = (initial_amount - fwdret_amount) / initial_amount
    bin_loss: float = tot_loss - fwdret_loss

    print(
        "【%s】:Dropped %.1f%% entries from factor data: %.1f%% in forward "
        "returns computation and %.1f%% in binning phase "
        "(set max_loss=0 to see potentially suppressed Exceptions)."
        % (factor_name.title(),tot_loss * 100, fwdret_loss * 100, bin_loss * 100)
    )

    if tot_loss > max_loss:
        message = "【%s】:max_loss (%.1f%%) exceeded %.1f%%, consider increasing it." % (
            factor_name.title(),
            max_loss * 100,
            tot_loss * 100,
        )
        raise MaxLossExceededError(message)
    else:
        print("【%s】:max_loss is %.1f%%, not exceeded: OK!" % (factor_name.title(),max_loss * 100))

    return factor_copy


def rethrow(exception, additional_message):
    """
    Re-raise the last exception that was active in the current scope
    without losing the stacktrace but adding an additional message.
    This is hacky because it has to be compatible with both python 2/3
    """
    e = exception
    m = additional_message
    e.args = (e.args[0] + m,) + e.args[1:] if e.args else (m,)
    raise e


def non_unique_bin_edges_error(func):
    """
    Give user a more informative error in case it is not possible
    to properly calculate quantiles on the input dataframe (factor)
    """
    message = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical
    values and they span more than one quantile. The quantiles are choosen
    to have the same number of records each, but the same value cannot span
    multiple quantiles. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get unequal
        number of records per quantile
    3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
        buckets to be evenly spaced according to the values themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - for factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value
    Please see utils.get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.

"""

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if "Bin edges must be unique" in str(e):
                rethrow(e, message)
            raise

    return dec


@non_unique_bin_edges_error
def quantize_factor(
    factor_ser: pd.Series,
    quantiles=5,
    bins=None,
    by_group=False,
    no_raise=False,
    zero_aware=False,
):
    """
    Computes period wise factor quantiles.

    Parameters
    ----------
    factor_ser : pd.Series- MultiIndex
        A MultiIndex Series indexed by datetime (level 0) and instrument (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the instrument belongs to.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group : bool, optional
        If True, compute quantile buckets separately for each group.
    no_raise: bool, optional
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.

    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and asset.
    """
    if not (
        (quantiles is not None and bins is None)
        or (quantiles is None and bins is not None)
    ):
        raise ValueError("Either quantiles or bins should be provided")

    if zero_aware and not isinstance(quantiles, int) and not isinstance(bins, int):
        msg = "zero_aware should only be True when quantiles or bins is an" " integer"
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = (
                    pd.qcut(x[x >= 0], _quantiles // 2, labels=False)
                    + _quantiles // 2
                    + 1
                )
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2, labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2, labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2, labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index, dtype=np.float64)
            raise e

    grouper = [factor_ser.index.get_level_values("datetime")]
    if by_group:
        grouper.append("group")

    factor_quantile = factor_ser.groupby(grouper, group_keys=False).apply(
        quantile_calc, quantiles, bins, zero_aware, no_raise
    )
    factor_quantile.name = "factor_quantile"

    return factor_quantile.dropna()

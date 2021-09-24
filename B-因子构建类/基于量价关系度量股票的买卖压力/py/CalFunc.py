import pandas as pd
import numpy as np
import time



def rolling_5vwap(xdf):
    return xdf.groupby(level='code').apply(
        lambda x: rolling_apply(x, lambda x_df: cal_vwap(x_df), 5))


def rolling_30vwap(xdf):
    return xdf.groupby(level='code').apply(
        lambda x: rolling_apply(x, lambda x_df: cal_vwap(x_df), 30))


def rolling_5apb(xdf):
    return xdf.groupby(level='code').apply(
        lambda x: rolling_apply(x, lambda x_df: cal_APB(x_df, 3), 5))


def rolling_30apb(xdf):
    return xdf.groupby(level='code').apply(
        lambda x: rolling_apply(x, lambda x_df: cal_APB(x_df, 15), 30))


# 计算vwap
def cal_vwap(df: pd.DataFrame) -> pd.Series:
    idx = df.index.get_level_values(1)[-1]
    return pd.DataFrame({'vwap': np.average(df['close'], weights=df['volume'])},
                        index=[idx])


# 计算APB
def cal_APB(df: pd.DataFrame, threshold: int) -> pd.Series:

    idx = df.index.get_level_values(1)[-1]
    trade_day = df['paused'].sum()
    # 过滤小于threshold数量的股票
    if trade_day < threshold:
        return pd.DataFrame(
            {
                'apb':
                    np.mean(df['vwap']) /
                    np.average(df['vwap'], weights=df['volume'])
            },
            index=[idx])
    else:
        return pd.DataFrame({'apb': np.nan}, index=[idx])


# 定义rolling_apply理论上应该比for循环快
# pandas.rolling.apply不支持多列
def rolling_apply(df, func, win_size) -> pd.Series:

    iidx = np.arange(len(df))
    shape = (iidx.size - win_size + 1, win_size)
    strides = (iidx.strides[0], iidx.strides[0])
    res = np.lib.stride_tricks.as_strided(
        iidx, shape=shape, strides=strides, writeable=True)

    return pd.concat((func(df.iloc[r]) for r in res), axis=0)  # concat可能会有点慢
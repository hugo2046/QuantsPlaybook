import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd


def plot_signals(
    signal: pd.Series = None,
    entries: pd.Series = None,
    exits: pd.Series = None,
    figsize: Tuple[int] = (16, 0.85),
    title: str = "",
    ylabel: str = "",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    绘制信号图。
    参数：
    - signal: pd.Series，包含信号的序列。
    - figsize: Tuple[int]，图形尺寸，默认为 (16, 0.85)。
    - title: str，图形标题，默认为空字符串。
    返回：
    - plt.axes，图形的坐标轴对象。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if signal is not None and (entries is not None) and (exits is not None):
        raise ValueError("signal, entries, and exits cannot be plotted together.")

    if signal is not None:
        signal: pd.Series = signal.astype(int)
        signal.where(signal == 1, 0).plot.area(
            ax=ax, stacked=True, color="red", alpha=0.25
        )
        signal.where(signal == -1, 0).abs().plot.area(
            ax=ax, stacked=True, color="green", alpha=0.25
        )

    if (entries is not None) and (exits is not None):
        exits: pd.Series = exits.astype(int)
        entries: pd.Series = entries.astype(int)

        entries.where(entries == 1, 0).plot.area(
            ax=ax, stacked=True, color="red", alpha=0.25
        )
        exits.where(exits == 1, 0).plot.area(
            ax=ax, stacked=True, color="green", alpha=0.25
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


def plot_signal_conflicts(
    entries: pd.Series,
    exits: pd.Series,
    figsize: Tuple[int, int] = (16, 0.85),
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    检查买入和卖出信号是否有冲突，并绘制相应的图表。

    参数:
    entries (pd.Series): 买入信号
    exits (pd.Series): 卖出信号
    figsize (tuple): 图表的尺寸，默认为 (16, 0.85)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # 将 entries 和 exits 转换为整数并相加
    data: pd.Series = entries.astype(int) + exits.astype(int)
    if data.max()>1:
        # 绘制大于等于2的部分（冲突部分）
        data.where(data >= 2).plot.area(ax=ax, color="red", stacked=True)

        # 绘制小于2的部分（非冲突部分）
        data.where(data < 2).plot.area(ax=ax, color="darkgray", stacked=True)
    else:
        data.plot.area(ax=ax, color="darkgray", stacked=True)
    return ax

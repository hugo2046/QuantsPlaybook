'''
Author: Hugo
Date: 2025-02-06 09:43:07
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-07 10:55:27
Description: 画图
'''
from typing import Optional

import empyrical as ep
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


def plot_cumulative_returns(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = 'Cumulative Returns',
    figsize: tuple = (12, 4),
    period: str = 'daily',
    ax:plt.Axes=None
) -> plt.Axes:
    """
    绘制累计回报曲线和风险指标
    
    参数:
        returns: pd.Series, 策略收益率序列（日收益率）
        benchmark_returns: pd.Series, 基准收益率序列（可选）
        title: str, 图表标题
        figsize: tuple, 图表大小
        trading_days: int, 年交易日数
    """
    # 计算风险指标
    
    cum_returns:pd.Series = ep.cum_returns(returns)
    annual_return:float = ep.annual_return(returns, period=period)
    annual_vol:float = ep.annual_volatility(returns, period=period)
    sharpe:float = ep.sharpe_ratio(returns, period=period)
    max_drawdown:float = ep.max_drawdown(returns)
    
    # 创建图表
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制主标题和风险指标
    plt.suptitle(title, fontsize=14, y=1.05)
    risk_metrics = (
        f'Cumulative Return: {cum_returns.iloc[-1]:.2%} | '
        f'Annual Return: {annual_return:.2%} | '
        f'Annual Volatility: {annual_vol:.2%} | '
        f'Sharpe Ratio: {sharpe:.2f} | '
        f'Max Drawdown: {max_drawdown:.2%}'
    )
    plt.title(risk_metrics, fontsize=10, pad=20)
    
    # 绘制累计收益曲线
    cum_returns.plot(label='Strategy', color='red',alpha=0.8)
    
    # 如果有基准收益率，也绘制基准的累计收益曲线
    if benchmark_returns is not None:
        benchmark_cum_returns:pd.Series = ep.cum_returns(benchmark_returns)
        benchmark_cum_returns.plot(label='Benchmark', color='gray', linewidth=1, linestyle='--')
    
    # 设置y轴为百分比格式
    def percentage_formatter(x, p):
        return f'{x:.1%}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # 设置图表格式
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black',ls='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    
    # 调整布局
    plt.tight_layout()
    return ax
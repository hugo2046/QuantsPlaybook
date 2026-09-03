"""
RRG + 扩散指标算法层（主体纯函数，compute_rs 含数据兜底）。

按西部证券 2026-05-25《指数化配置系列研究（6）》研报复现。
- 公式参考：spec docs/superpowers/specs/2026-05-25-rrg-diffusion-reproduction-design.md
- 数据契约：所有输入为宽表 DataFrame（index=date, columns=code, value=对应值）。
- 调用者职责：价格严格为正；price 自身 index 上的停牌 NaN 应在传入前 ffill。
- 算法层兜底：compute_rs 把 price reindex 到 benchmark.index；若 price.index
  缺 benchmark 已有的交易日，会 ``warnings.warn`` 并对**整张** price ffill
  兜底（注意：触发兜底时 price 原有的 NaN 也会被一并 ffill；若调用方需要
  保留停牌 NaN 语义，应在传入前先 reindex+ffill 自己处理）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _smooth_ratio_shift(
    x: pd.DataFrame, lookback: int, smooth: int
) -> pd.DataFrame:
    """共享算子：``100 * x / x.shift(lookback)`` 再做 smooth-window MA。

    被 ``compute_rs_ratio`` / ``compute_rs_momentum`` / ``compute_rrg`` 共用，
    保证 RRG 核心公式只在此一处落地，避免多份实现漂移。
    """
    return (100.0 * x / x.shift(lookback)).rolling(window=smooth).mean()


def _require_min_len(
    price: pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    min_len: int,
    detail: str,
) -> None:
    """共享守卫：校验 price/benchmark 行数 >= ``min_len``，否则抛 ValueError。

    把 warm-up 长度下界这一处「曾经写错过」的载荷逻辑收口到一处，避免
    ``compute_rs_ratio`` / ``compute_rs_momentum`` / ``compute_rrg`` 三处
    各自复制公式与报错文案而漂移（参见项目 CLAUDE.md「NaN warm-up」一节）。

    :param min_len: 调用方按各自公式算出的最小行数。
    :param detail: 嵌入报错括号内的公式说明，如
        ``"lookback=220 + smooth_window=20"``。
    """
    if price.shape[0] < min_len or benchmark.shape[0] < min_len:
        raise ValueError(
            f"数据长度不足：需 >= {min_len} 行 ({detail})，"
            f"实得 price={price.shape[0]}, benchmark={benchmark.shape[0]}"
        )


def equal_weight_benchmark(price: pd.DataFrame) -> pd.Series:
    """等权组合净值基准：各列日收益等权平均后累乘成 NAV（首日=1.0）。

    研报 §2.1（line 273）以「中信一级行业等权组合」为 RRG 基准。本函数把行业收盘价
    宽表聚成等权组合净值，供 ``compute_rs`` / ``compute_rrg`` 作 ``benchmark`` 入参。

    :param price: 行业收盘价宽表，index=date × columns=industry_code（>0）。
        某列某日 NaN（未上市/停牌）由 ``mean(skipna)`` 自动从当日截面剔除。
    :returns: 等权组合 NAV ``pd.Series``，index=price.index，首日=1.0。
    :note:
        - RRG 的 RS-Ratio 仅依赖基准的**收益比**（``B_{t-T}/B_t``），对基准绝对水平
          尺度不变，故 NAV 首日=1.0 不影响 RRG 结果。
        - 「等权**组合**」=各行业日收益等权复合（本函数），**不是**行业指数点位的
          截面算术均值（后者收益被高点位行业主导，非等权组合）。
    """
    rets = price.pct_change().mean(axis=1)          # 截面等权日收益（skipna 跳 NaN 列）
    return (1.0 + rets.fillna(0.0)).cumprod()


def compute_rs(
    price: pd.DataFrame, benchmark: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """
    计算相对强度 RS = (price / benchmark) * 100。

    :param price: 行业/指数收盘价宽表，index=date, columns=industry_code。
        必须严格为正（>0）；含 0 或负值会产生 inf/NaN 并通过下游 rolling 传播。
    :param benchmark: 基准价格，Series 或单列 DataFrame，index=date。
        同样必须严格为正。
    :returns: RS DataFrame，**index = benchmark.index**，columns = price.columns。

    :note: price 会被 reindex 到 benchmark.index——以宽基指数日历为基准。
        若 ``benchmark.index`` 含 ``price.index`` 没有的交易日（典型如行业
        指数 ETL 漏跑某几天），会 ``warnings.warn`` 并对整张 price 进行
        ``ffill`` 兜底，避免散落 NaN 被 ``shift(lookback)`` + ``rolling``
        双层放大约 ``2 * smooth_window`` 倍。无缺日时仅 reindex，零额外开销。
    """
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[:, 0]

    price_aligned = price.reindex(benchmark.index)
    missing = benchmark.index.difference(price.index)
    if len(missing) > 0:
        sample = [d.strftime("%Y-%m-%d") for d in missing[:5]]
        suffix = "..." if len(missing) > 5 else ""
        warnings.warn(
            f"price 在 benchmark.index 上缺 {len(missing)} 个交易日，"
            f"已自动 ffill 兜底；建议核查数据源完整性。"
            f"前 5 个缺失日: {sample}{suffix}",
            UserWarning,
            stacklevel=2,
        )
        price_aligned = price_aligned.ffill()
    return price_aligned.div(benchmark, axis=0) * 100.0


def compute_rs_ratio(
    price: pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    lookback: int = 220,
    smooth_window: int = 20,
) -> pd.DataFrame:
    """
    JdK RS-Ratio：相对强度的 lookback 期比率，再做 smooth_window MA 平滑。

    :param price: 行业/指数收盘价宽表，index=date, columns=industry_code。
    :param benchmark: 基准价格，Series 或单列 DataFrame，index=date。
    :param lookback: RS 比率回看天数（研报默认 220）。
    :param smooth_window: 平滑窗口（研报默认 20）。
    :returns: RS_Ratio DataFrame，**index = benchmark.index**，columns = price.columns。
        起始 lookback + smooth_window - 1 行 NaN。
    :raises ValueError: 当 price 或 benchmark 长度不足以产生至少一行非 NaN，
        即 < lookback + smooth_window（默认 240）。
    """
    min_len = lookback + smooth_window
    _require_min_len(
        price, benchmark, min_len,
        f"lookback={lookback} + smooth_window={smooth_window}",
    )
    rs = compute_rs(price, benchmark)
    return _smooth_ratio_shift(rs, lookback, smooth_window)


def compute_rs_momentum(
    price: pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    lookback_ratio: int = 220,
    lookback_mom: int = 60,
    smooth_window: int = 20,
) -> pd.DataFrame:
    """
    JdK RS-Momentum：RS_Ratio 的 lookback_mom 期比率，再做 smooth_window MA 平滑。

    :param price: 行业/指数收盘价宽表，index=date, columns=industry_code。
    :param benchmark: 基准价格，Series 或单列 DataFrame，index=date。
    :param lookback_ratio: RS 比率回看天数（研报默认 220）。
    :param lookback_mom: RS_Ratio 比率回看天数（研报默认 60）。
    :param smooth_window: 平滑窗口（研报默认 20）。
    :returns: RS_Momentum DataFrame，**index = benchmark.index**，columns = price.columns。
        起始 lookback_ratio + lookback_mom + 2*(smooth_window - 1) 行 NaN（默认 318）。
    :raises ValueError: 当 price 或 benchmark 长度不足以产生至少一行非 NaN 的 RS_Mom，
        即 < lookback_ratio + lookback_mom + 2*(smooth_window - 1) + 1。
    """
    min_len = lookback_ratio + lookback_mom + 2 * (smooth_window - 1) + 1
    _require_min_len(
        price, benchmark, min_len,
        f"lookback_ratio={lookback_ratio} + lookback_mom={lookback_mom} "
        f"+ 2*(smooth_window-1)+1，smooth_window={smooth_window}",
    )
    rs_ratio = compute_rs_ratio(
        price, benchmark, lookback=lookback_ratio, smooth_window=smooth_window
    )
    return _smooth_ratio_shift(rs_ratio, lookback_mom, smooth_window)


def compute_rrg(
    price: pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    lookback_ratio: int = 220,
    lookback_mom: int = 60,
    smooth_window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    一次性计算 (RS_Ratio, RS_Momentum)，复用中间 RS_Ratio，避免重复计算。

    :param price: 行业/指数收盘价宽表，index=date, columns=industry_code。
    :param benchmark: 基准价格，Series 或单列 DataFrame，index=date。
    :param lookback_ratio: RS 比率回看天数（研报默认 220）。
    :param lookback_mom: RS_Ratio 比率回看天数（研报默认 60）。
    :param smooth_window: 平滑窗口（研报默认 20）。
    :returns: (rs_ratio, rs_mom)，两者 **index = benchmark.index**，columns = price.columns。
        起始 rs_mom 头 lookback_ratio + lookback_mom + 2*(smooth_window - 1) 行 NaN（默认 318）。
    :raises ValueError:
        - benchmark 是 DataFrame 但 shape[1] == 0（无可取列）
        - price 或 benchmark 长度 < lookback_ratio + lookback_mom + 2*(smooth_window - 1) + 1
        - price.index 与 benchmark.index 重合行数 < 上述阈值（即使等长，错位日历也会静默全 NaN）
    """
    if isinstance(benchmark, pd.DataFrame):
        if benchmark.shape[1] == 0:
            raise ValueError("benchmark DataFrame 必须至少有 1 列")
        benchmark = benchmark.iloc[:, 0]

    min_len = lookback_ratio + lookback_mom + 2 * (smooth_window - 1) + 1
    _require_min_len(
        price, benchmark, min_len,
        f"lookback_ratio={lookback_ratio} + lookback_mom={lookback_mom} "
        f"+ 2*(smooth_window-1)+1，smooth_window={smooth_window}",
    )

    overlap = price.index.intersection(benchmark.index)
    if overlap.shape[0] < min_len:
        raise ValueError(
            f"price 与 benchmark 日历重合不足：重合 {overlap.shape[0]} 行 < {min_len} 行；"
            f"请确保两者覆盖同一交易日历"
        )

    rs = compute_rs(price, benchmark)
    rs_ratio = _smooth_ratio_shift(rs, lookback_ratio, smooth_window)
    rs_mom = _smooth_ratio_shift(rs_ratio, lookback_mom, smooth_window)
    return rs_ratio, rs_mom


def _resolve_industries(membership: pd.DataFrame) -> list:
    """从 membership 矩阵抽取去重、排序的行业名列表（剔除 NaN）。

    :note: 排序使用 key=str 以兼容行业名混含 str 与数值的边界情况。
    """
    names = pd.unique(membership.values.ravel())
    return sorted([n for n in names if pd.notna(n)], key=str)


def _up_flags(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """上涨判定矩阵：``close > lookback 日前``，停牌/缺值位置置 NaN（不计入分母）。

    两个 diffusion 变体（count / value-weighted）共用此口径——单一定义避免
    warm-up / invalid 判定漂移；``shift`` 只算一次。
    """
    prev = close.shift(lookback)
    is_up = (close > prev).astype(float)
    return is_up.where(~(close.isna() | prev.isna()))


def diffusion_count_ratio(
    close: pd.DataFrame,
    membership: pd.DataFrame,
    lookback: int = 220,
    smooth_window: int = 20,
) -> pd.DataFrame:
    """
    数量占比版扩散指标：行业内"上涨成分股数 / 有效成分股数"，再做 MA 平滑。

    :param close: index=date, columns=stock_code, value=收盘价。
    :param membership: index=date, columns=stock_code, value=行业标签 或 NaN。
        本函数对标签取值不敏感（按值分组即可）；但若要与 RRG 指标列对齐，标签
        须用 industry_code（见 etf_rrg spec §3.2，非 industry_name）。时变格式，
        自动支持成分股调入调出。
    :param lookback: 上涨判定回看天数（研报默认 220）。
    :param smooth_window: 平滑窗口（研报默认 20）。
    :returns: index=date, columns=行业标签（同 membership 取值）, value ∈ [0, 1]。
        某行业某日全员无效判定 → NaN（避免 0 误导）。
    """
    membership_aligned = membership.reindex(index=close.index, columns=close.columns)

    is_up = _up_flags(close, lookback)

    industries = _resolve_industries(membership_aligned)
    cols = {}
    for ind in industries:
        in_ind = membership_aligned == ind
        is_up_in_ind = is_up.where(in_ind)
        ups = is_up_in_ind.sum(axis=1, min_count=1)
        valid_count = is_up_in_ind.notna().sum(axis=1)
        cols[ind] = ups / valid_count.where(valid_count > 0)

    diffusion = pd.DataFrame(cols, index=close.index)
    return diffusion.rolling(window=smooth_window).mean()


def diffusion_value_weighted(
    close: pd.DataFrame,
    membership: pd.DataFrame,
    free_float_mv: pd.DataFrame,
    lookback: int = 220,
    smooth_window: int = 20,
) -> pd.DataFrame:
    """
    自由流通市值加权扩散指标：行业内"上涨成分股自由流通市值之和 / 有效成分股市值之和"，
    再做 MA 平滑。

    :param close: index=date, columns=stock_code, value=收盘价。
    :param membership: index=date, columns=stock_code, value=行业标签 或 NaN。
        本函数对标签取值不敏感（按值分组即可）；但若要与 RRG 指标列对齐，标签
        须用 industry_code（见 etf_rrg spec §3.2，非 industry_name）。时变格式，
        自动支持成分股调入调出。
    :param free_float_mv: index=date, columns=stock_code, value=自由流通市值 (元)。
    :param lookback: 上涨判定回看天数（研报默认 220）。
    :param smooth_window: 平滑窗口（研报默认 20）。
    :returns: index=date, columns=行业标签（同 membership 取值）, value ∈ [0, 1]。
        某行业某日全员无效判定 → NaN（避免 0 误导）。
        某成分股 free_float_mv 缺失 → 该日该股票从分子分母双方剔除
        （视作权重未知样本，比率基于剩余成分股；如需 NaN 显式标记，
        调用者应在传入前自行处理 mv 缺失）。
    """
    membership_aligned = membership.reindex(index=close.index, columns=close.columns)
    mv_aligned = free_float_mv.reindex(index=close.index, columns=close.columns)

    is_up = _up_flags(close, lookback)

    industries = _resolve_industries(membership_aligned)
    cols = {}
    for ind in industries:
        in_ind = membership_aligned == ind
        # 仅在 is_up 有效且属于本行业的位置取市值
        mv_in_ind = mv_aligned.where(in_ind & is_up.notna())
        mv_up = mv_in_ind.where(is_up == 1.0)
        numerator = mv_up.sum(axis=1, min_count=1)
        denominator = mv_in_ind.sum(axis=1, min_count=1)
        cols[ind] = numerator / denominator.where(denominator > 0)

    diffusion = pd.DataFrame(cols, index=close.index)
    return diffusion.rolling(window=smooth_window).mean()


def classify_quadrant(
    rs_ratio: pd.DataFrame, rs_mom: pd.DataFrame
) -> pd.DataFrame:
    """
    将 (RS_Ratio, RS_Mom) 按研报象限定义分类为 1/2/3/4。

    - 1 = RS_Ratio > 100 & RS_Mom > 100（领先 leading）
    - 2 = RS_Ratio < 100 & RS_Mom > 100（改善 improving）
    - 3 = RS_Ratio < 100 & RS_Mom < 100（滞后 lagging）
    - 4 = RS_Ratio > 100 & RS_Mom < 100（疲软 weakening）

    任一值为 NaN 或恰为 100 → 返回 NaN（不强行归边）。

    :param rs_ratio: RS_Ratio 宽表，index=date, columns=industry_code。
    :param rs_mom: RS_Momentum 宽表，必须与 rs_ratio 列完全一致（行索引可不同，会做 axis=0 对齐）。
    :returns: 同形状的象限编号 DataFrame（float，1/2/3/4 或 NaN）。
    :raises ValueError: rs_ratio.columns 与 rs_mom.columns 不一致时（避免静默生成幻列）。
    """
    if not rs_ratio.columns.equals(rs_mom.columns):
        raise ValueError(
            f"rs_ratio 与 rs_mom 列不一致：rs_ratio={list(rs_ratio.columns)}, "
            f"rs_mom={list(rs_mom.columns)}"
        )
    rs_ratio_aligned, rs_mom_aligned = rs_ratio.align(rs_mom, axis=0)
    result = pd.DataFrame(
        np.nan, index=rs_ratio_aligned.index, columns=rs_ratio_aligned.columns
    )
    result[(rs_ratio_aligned > 100) & (rs_mom_aligned > 100)] = 1.0
    result[(rs_ratio_aligned < 100) & (rs_mom_aligned > 100)] = 2.0
    result[(rs_ratio_aligned < 100) & (rs_mom_aligned < 100)] = 3.0
    result[(rs_ratio_aligned > 100) & (rs_mom_aligned < 100)] = 4.0
    return result

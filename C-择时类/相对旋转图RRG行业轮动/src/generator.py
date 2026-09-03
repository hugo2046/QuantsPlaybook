"""ETF-RRG 信号编排层：串「取数 → 算指标 → 出信号」，对外单一入口 generate。

信号类型 dict 派发——扩展点 = 加一个 strategy 函数 + 在 _SIGNALS 注册一行。
按 signal 的需求位惰性取数：纯 quadrant 不触发重的成分股拉取。
全程日频；月末采样交给调用者。
"""
from __future__ import annotations

import warnings
from typing import Iterable

import pandas as pd

from src import strategy
from src.data_provider import fetch_diffusion_dataset, fetch_rrg_dataset
from src.factor_algo import compute_rrg, diffusion_value_weighted
from src.utils import to_equal_weight

# signal 名 → (strategy 函数, 需要 rrg 指标?, 需要 diffusion 指标?)
_SIGNALS = {
    "quadrant": (strategy.select_by_quadrant, True, False),
    "diffusion": (strategy.select_by_diffusion, False, True),
    "diffusion_rrg": (strategy.select_diffusion_with_rrg, True, True),
}

__all__ = ["RRGSignalGenerator"]


class RRGSignalGenerator:
    """RRG/扩散行业信号生成器（轻量编排，按需惰性取数 + 缓存）。

    :param benchmark_code: RRG 基准——``"ind_avg"``（默认）用**行业等权组合净值**
        （研报 §2.1 口径），或传宽基指数代码（如中证 1000 ``"000852.SH"``）。
    :param start_date: 起始日期（``'YYYY-MM-DD'``）。
    :param end_date: 结束日期（``'YYYY-MM-DD'``）。
    :param lookback_ratio: RS_Ratio 回看天数（研报默认 220）。
    :param lookback_mom: RS_Momentum 回看天数（研报默认 60）。
    :param smooth_window: 平滑窗口（研报默认 20）。
    :param benchmark_exclude: 仅 ``"ind_avg"`` 基准生效——构造等权基准时剔除的行业代码集。
        默认哨兵 ``"auto"`` 按 ``classification`` 解析：``sw`` → ``("801230.SI",)``、
        ``zx`` → ``("CI005029.CI", "CI005030.CI")``（综合 + 综合金融，研报原文口径）。
        传显式 tuple 或 ``None`` 则原样使用。
    :param classification: 行业分类体系，``"sw"``（申万一级，默认）/ ``"zx"``（中信一级）。
        ``code2name`` 自动跟随（它本就来自取数返回的 industry_name）。
        信号链（``RRGSignalGenerator``）与暴露链（``ETFExposureBuilder``）必须使用同一
        ``classification``，否则 ``optimizer`` 的行业打分与暴露矩阵列无法对齐，
        会静默退化为空仓。
    """

    def __init__(
        self,
        benchmark_code: str = "ind_avg",
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31",
        *,
        lookback_ratio: int = 220,
        lookback_mom: int = 60,
        smooth_window: int = 20,
        benchmark_exclude: tuple[str, ...] | str | None = "auto",
        classification: str = "sw",
    ):
        self.benchmark_code = benchmark_code
        self.benchmark_exclude = benchmark_exclude
        self.classification = classification
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_ratio = lookback_ratio
        self.lookback_mom = lookback_mom
        self.smooth_window = smooth_window
        self._rs_ratio: pd.DataFrame | None = None
        self._rs_mom: pd.DataFrame | None = None
        self._diffusion: pd.DataFrame | None = None
        self._code2name: dict[str, str] | None = None

    @property
    def code2name(self) -> dict[str, str]:
        """行业代码 → 中文简称映射（首次访问惰性触发 RRG 取数）。

        供 ``plotting.plot_signal_heatmap`` 等把信号面板的行业代码列转中文标签。
        """
        self._ensure_rrg()
        return self._code2name

    def get_rrg(self, replace_col: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """取原始 RS_Ratio / RS_Momentum 指标，供 RRG 轨迹图绘制。

        与信号无关的"裸指标"出口（首次访问惰性触发 RRG 取数 + 计算）。可直接
        解包传入双后端绘图函数：``plot_rrg_echart(*gen.get_rrg())`` /
        ``plot_rrg_chart(*gen.get_rrg())``。

        :returns: ``(rs_ratio, rs_mom)`` 两个 DataFrame，index=date、
            columns=industry_code，中心 100。
        :note:
            index 与 ``__init__`` 的 ``[start_date, end_date]`` 窗口一致——
            ``fetch_rrg_dataset`` 不预取 warm-up，故前
            ``lookback_ratio + lookback_mom + 2*(smooth_window-1)`` 行（默认 318）
            为 NaN（绘图时自然略过）。如需窗口起点即有有效轨迹，
            把 ``start_date`` 提前约 318 个交易日。
        """
        self._ensure_rrg()
        if replace_col:
            # 轨迹图不关心行业代码，直接用中文名替换列标签；columns 对齐后再改名，避免 code2name 漏掉某些列
            rs_ratio = self._rs_ratio.rename(columns=self._code2name)
            rs_mom = self._rs_mom.rename(columns=self._code2name)
            return rs_ratio, rs_mom
        return self._rs_ratio, self._rs_mom

    def _ensure_rrg(self) -> None:
        """惰性计算并缓存 RS_Ratio / RS_Momentum 指标 + 行业代码→名映射。"""
        if self._rs_ratio is None:
            benchmark, industry_close, self._code2name = fetch_rrg_dataset(
                benchmark_code=self.benchmark_code,
                start_date=self.start_date,
                end_date=self.end_date,
                benchmark_exclude=self.benchmark_exclude,
                classification=self.classification,
            )
            self._rs_ratio, self._rs_mom = compute_rrg(
                industry_close,
                benchmark,
                lookback_ratio=self.lookback_ratio,
                lookback_mom=self.lookback_mom,
                smooth_window=self.smooth_window,
            )

    def _ensure_diffusion(self) -> None:
        """惰性计算并缓存扩散指标。"""
        if self._diffusion is None:
            close, membership, free_float_mv = fetch_diffusion_dataset(
                start_date=self.start_date,
                end_date=self.end_date,
                lookback=self.lookback_ratio,
                smooth_window=self.smooth_window,
                classification=self.classification,
            )
            self._diffusion = diffusion_value_weighted(
                close,
                membership,
                free_float_mv,
                lookback=self.lookback_ratio,
                smooth_window=self.smooth_window,
            )

    def generate(
        self,
        signal: str = "diffusion_rrg",
        *,
        exclude: Iterable[str] | None = None,
        **params,
    ) -> pd.DataFrame:
        """
        生成日频 boolean 信号 mask。

        :param signal: ``'quadrant'`` / ``'diffusion'`` / ``'diffusion_rrg'``。
        :param exclude: 在排序/选取**之前**从候选行业池剔除的 ``industry_code`` 列表
            （如研报"剔除综合、综合金融"——申万体系传 ``["801230.SI"]``（无综合金融），
            中信体系传 ``["CI005029.CI", "CI005030.CI"]``）。在排序前剔除
            才能让 ``top_n`` 自动回填下一名可交易行业；排序后剔除只会徒留空位。
            不存在的代码静默忽略；不修改内部缓存，故同一实例可在剔/不剔之间反复切换。
        :param params: 透传给对应 strategy 函数（如 top_n / quadrants / keep_quadrants）。
        :returns: index=date × columns=industry_code 的 boolean DataFrame，
            行索引裁剪到 ``[start_date, end_date]`` 请求窗口（剔除内部 warm-up 预取行）。
        :note:
            RRG 类信号（``quadrant`` / ``diffusion_rrg``）因 ``fetch_rrg_dataset`` 不预取
            warm-up，窗口内前 ``lookback_ratio + lookback_mom + 2*(smooth_window-1)`` 行
            （默认 318，约 15 个月）RS 指标为 NaN → 当日全 False。如需 start_date 起即有
            RRG 信号，请把 start_date 提前约 318 个交易日。``diffusion`` 信号无此问题
            （数据层已预取 warm-up）。
        :raises ValueError: signal 不在 _SIGNALS。
        """
        if signal not in _SIGNALS:
            raise ValueError(f"未知 signal {signal!r}；可选：{list(_SIGNALS)}")
        fn, need_rrg, need_diff = _SIGNALS[signal]
        drop = list(exclude) if exclude else []

        def _trim(df: pd.DataFrame) -> pd.DataFrame:
            # 排序前剔除候选行业；errors="ignore" 容忍不存在代码；drop 返回新对象，缓存不变
            return df.drop(columns=drop, errors="ignore") if drop else df

        args: list[pd.DataFrame] = []
        if need_diff:
            self._ensure_diffusion()
            args.append(_trim(self._diffusion))
        if need_rrg:
            self._ensure_rrg()
            args += [_trim(self._rs_ratio), _trim(self._rs_mom)]

        # 列对齐早暴露（spec §10）：扩散+RRG 两套指标列无交集会静默全 False
        if need_diff and need_rrg:
            if self._diffusion.columns.intersection(self._rs_ratio.columns).empty:
                warnings.warn(
                    "diffusion 与 RRG 指标列无交集，信号将全为 False；"
                    "请检查 membership 是否使用 industry_code（spec §3.2）。",
                    UserWarning,
                    stacklevel=2,
                )

        mask = fn(*args, **params)
        # 裁剪到请求窗口：fetch_diffusion_dataset 为 warm-up 前移了起点，diffusion 比 rs
        # 多出若干 warm-up 行；不裁剪会让不同 signal 的行索引不一致（diffusion_rrg 取并集
        # → 2021 行，quadrant 只到 start_date），调用者按某一 mask 采样另一 mask 时会 KeyError。
        return mask.loc[self.start_date : self.end_date]

    def to_weight(self, mask: pd.DataFrame) -> pd.DataFrame:
        """boolean mask → 等权权重（每行和=1）。"""
        return to_equal_weight(mask)

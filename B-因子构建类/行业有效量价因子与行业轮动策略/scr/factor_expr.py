"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-02-15 14:59:13
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-02-20 15:20:20
Description: 
"""
from itertools import permutations
from typing import Dict, List, Tuple

from qlib.contrib.data.handler import (
    _DEFAULT_INFER_PROCESSORS,
    _DEFAULT_LEARN_PROCESSORS,
    DataHandlerLP,
    check_transform_proc,
)


class VolumePriceFactor192(DataHandlerLP):
    def __init__(
        self,
        instruments="pool",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processor=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.get("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
        )

    def get_label_config(self):

        return (["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"])

    def get_feature_config(self) -> Tuple[List, List]:

        conf: Dict = {"all": {}}

        return self.parse_config_to_fields(conf)

    @staticmethod
    def parse_config_to_fields(config: Dict) -> Tuple[List, List]:

        fields: List = []
        names: List = []

        if "all" in config:
            windows: List = [5, 10, 20, 60, 120, 180]
            periods: Tuple = tuple(permutations(windows, 2))
            second_mom_periods: Tuple = tuple(permutations(windows, 3))
            factornames: List = [
                "SencondMomentum",
                "MomentumTermSpread",
                "AmountVolatility",
                "VolumeVolatility",
                "NetPosition",
                "PositionChange",
                "VolumePriceRankCorr",
                "VolumePriceCorr",
                "FirstOrderDivergence",
                "VolumeAmplitudeCoMovement",
            ]

            for k in factornames:

                if k == "SencondMomentum":
                    config[k] = {"windows": second_mom_periods}
                elif k in ["MomentumTermSpread", "PositionChange"]:
                    config[k] = {"windows": periods}
                elif k in [
                    "AmountVolatility",
                    "VolumeVolatility",
                    "NetPosition",
                    "VolumePriceRankCorr",
                    "VolumePriceCorr",
                    "FirstOrderDivergence",
                    "VolumeAmplitudeCoMovement",
                ]:
                    config[k] = {"windows": windows}

        if "SencondMomentum" in config:

            second_mom_periods: Tuple = config["SencondMomentum"].get(
                "windows", ((5, 5, 10),)
            )

            # 二阶动量 Second Momentum
            fields += [
                f"EMA(($close-Mean($close,{d1}))/Mean($close,{d1})-Ref(($close-Mean($close,{d1}))/Mean($close,{d1}),{d2}),{d3})"
                for d1, d2, d3 in second_mom_periods
            ]

            names += [
                f"SencondMomentum_{d1}_{d2}_{d3}" for d1, d2, d3 in second_mom_periods
            ]

        if "MomentumTermSpread" in config:
            periods: Tuple = config["MomentumTermSpread"].get("windows", ((10, 5),))
            # 动量期限差 Momentum Term Spread
            fields += [
                f"($close-Ref($close,{d1}))/Ref($close,{d1})-($close-Ref($close,{d2}))/Ref($close,{d2})"
                for d1, d2 in periods
                if d1 > d2
            ]

            names += [f"MomentumTermSpread_{d1}_{d2}" for d1, d2 in periods if d1 > d2]

        if "AmountVolatility" in config:

            windows: Tuple = config["AmountVolatility"].get("windows", ((5),))

            # 成交金额波动 Turnover Volatility
            fields += [f"-1*Std($amount,{d1})" for d1 in windows]

            names += [f"AmountVolatility_{d1}" for d1 in windows]

        if "VolumeVolatility" in config:

            windows: Tuple = config["VolumeVolatility"].get("windows", ((5),))

            # 成交量波动 Volume Volatility
            fields += [f"-1*Std($volume,{d1})" for d1 in windows]

            names += [f"VolumeVolatility_{d1}" for d1 in windows]

        if "NetPosition" in config:

            windows: Tuple = config["NetPosition"].get("windows", ((10),))

            # 多空对比总量 Net Position
            fields += [
                f"Sum(($close-$low)/($high-$close+1e-12),{d1})" for d1 in windows
            ]

            names += [f"NetPosition_{d1}" for d1 in windows]

        if "PositionChange" in config:

            periods: Tuple = config["PositionChange"].get("windows", ((10, 5),))

            # 多空对比变化 PositionChange
            fields += [
                f"EMA($volume*($close-$low-$high+$close)/($high-$low+1e-12),{d1})-EMA($volume*($close-$low-$high+$close)/($high-$low+1e-12),{d2})"
                for d1, d2 in periods
                if d1 > d2
            ]

            names += [f"PositionChange_{d1}_{d2}" for d1, d2 in periods if d1 > d2]

        if "VolumePriceRankCorr" in config:

            windows: Tuple = config["VolumePriceRankCorr"].get("windows", ((10),))
            # 量价背离协方差 Volume-Price Rank Correlation
            fields += [
                f"-1*Cov(Rank($close,{d1}),Rank($volume,{d1}),{d1})" for d1 in windows
            ]
            names += [f"VolumePriceRankCorr_{d1}" for d1 in windows]

        if "VolumePriceCorr" in config:

            windows: Tuple = config["VolumePriceCorr"].get("windows", ((20),))
            # 量价相关系数 Volume-Price Correlation
            fields += [f"-1*Corr($close,$volume,{d1})" for d1 in windows]
            names += [f"VolumePriceCorr_{d1}" for d1 in windows]

        if "FirstOrderDivergence" in config:

            windows: Tuple = config["FirstOrderDivergence"].get("windows", ((10),))
            # 一阶量价背离 First-Order Divergence
            fields += [
                f"-1*Corr(Rank($volume/Ref($volume,1)-1,{d1}),Rank($close/$open-1,{d1}),{d1})"
                for d1 in windows
            ]

            names += [f"FirstOrderDivergence_{d1}" for d1 in windows]

        if "VolumeAmplitudeCoMovement" in config:

            windows: Tuple = config["VolumeAmplitudeCoMovement"].get("windows", ((10),))
            # 量幅同向 Volume-Amplitude Co-movement
            fields += [
                f"-1*Corr(Rank($volume/Ref($volume,1)-1,{d1}),Rank($high/$low-1,{d1}),{d1})"
                for d1 in windows
            ]

            names += [f"VolumeAmplitudeCoMovement_{d1}" for d1 in windows]

        return fields, names


class VolumePriceFactor10(VolumePriceFactor192):
    def get_feature_config(self) -> Tuple[List, List]:

        conf: Dict = {
            "SencondMomentum": {},
            "MomentumTermSpread": {},
            "AmountVolatility": {},
            "VolumeVolatility": {},
            "NetPosition": {},
            "PositionChange": {},
            "VolumePriceRankCorr": {},
            "VolumePriceCorr": {},
            "FirstOrderDivergence": {},
            "VolumeAmplitudeCoMovement": {},
        }

        return super().parse_config_to_fields(conf)

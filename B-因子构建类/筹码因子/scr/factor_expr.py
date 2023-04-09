from typing import List, Tuple

from qlib.contrib.data.handler import (
    DataHandlerLP,
    check_transform_proc,
)

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},

]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "CSRankNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]

class ChipsFactorBaisc(DataHandlerLP):
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

        pass


class TurnCoeffChips(ChipsFactorBaisc):
    def get_feature_config(self) -> Tuple[List, List]:

        fields: List = [
            "ARC($turnover_rate,$close,60)",
            "VRC($turnover_rate,$close,60)",
            "SRC($turnover_rate,$close,60)",
            "KRC($turnover_rate,$close,60)",
        ]
        names: List = ["ARC", "VRC", "SRC", "KRC"]

        return fields, names


class Chips(ChipsFactorBaisc):
    def get_feature_config(self) -> Tuple[List, List]:

        names: List = [
            "CYQK_C_T",
            "CYQK_C_U",
            "CYQK_C_TN",
            "ASR_T",
            "ASR_U",
            "ASR_TN",
            "CKDW_T",
            "CKDW_U",
            "CKDW_TN",
            "PRP_T",
            "PRP_U",
            "PRP_TN",
        ]

        fields: List = [
            f"{name}($close,$high,$low,$vol,$turnover_rate,80)" for name in names
        ]

        return fields, names


# class TurnCoeffChips(ChipsFactorBaisc):
#     def __init__(
#         self,
#         instruments="pool",
#         start_time=None,
#         end_time=None,
#         freq="day",
#         infer_processors=_DEFAULT_INFER_PROCESSORS,
#         learn_processors=_DEFAULT_LEARN_PROCESSORS,
#         fit_start_time=None,
#         fit_end_time=None,
#         filter_pipe=None,
#         inst_processor=None,
#         **kwargs,
#     ):
#         infer_processors = check_transform_proc(
#             infer_processors, fit_start_time, fit_end_time
#         )
#         learn_processors = check_transform_proc(
#             learn_processors, fit_start_time, fit_end_time
#         )

#         data_loader = {
#             "class": "QlibDataLoader",
#             "kwargs": {
#                 "config": {
#                     "feature": self.get_feature_config(),
#                     "label": kwargs.get("label", self.get_label_config()),
#                 },
#                 "filter_pipe": filter_pipe,
#                 "freq": freq,
#                 "inst_processor": inst_processor,
#             },
#         }

#         super().__init__(
#             instruments=instruments,
#             start_time=start_time,
#             end_time=end_time,
#             data_loader=data_loader,
#             learn_processors=learn_processors,
#             infer_processors=infer_processors,
#         )

#     def get_label_config(self):

#         return (["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"])

#     def get_feature_config(self) -> Tuple[List, List]:

#         fields: List = [
#             "ARC($turnover_rate,$close,60)",
#             "VRC($turnover_rate,$close,60)",
#             "SRC($turnover_rate,$close,60)",
#             "KRC($turnover_rate,$close,60)",
#         ]
#         names: List = ["ARC", "VRC", "SRC", "KRC"]

#         return fields, names

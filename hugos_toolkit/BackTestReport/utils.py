from typing import Dict


def get_value_from_traderanalyzerdict(dic: Dict, *args) -> float:
    """获取嵌套字典中的指定key"""
    if len(args) == 1:
        return dic.get(args[0], 0)
    for k in args:

        if res := dic.get(k, None):
            return get_value_from_traderanalyzerdict(res, *args[1:])

        return 0

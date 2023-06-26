'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-26 10:37:39
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-26 10:43:07
Description: 
'''
from typing import List,Tuple,Union
import numpy as np


def check_sign(left:Union[float,np.ndarray],right:Union[float,np.ndarray])->Union[float,np.ndarray]:
    """比较两个数的符号是否相同

    Parameters
    ----------
    left : Union[float,np.ndarray]
        数字或数组
    right : Union[float,np.ndarray]
        数字或数组

    Returns
    -------
    Union[float,np.ndarray]
        结果为布尔值或布尔数组
    """
    return (left>=0)==(right>=0)

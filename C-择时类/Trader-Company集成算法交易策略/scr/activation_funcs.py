'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-12 21:10:12
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-12 22:16:28
Description: 激活函数
'''
from typing import Union

import numpy as np


def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x


def tanh(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.tanh(x)


def sign(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.where(x > 0.0, 1, 0)


def ReLU(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return sign(x) * x


def Exp(x:Union[float,np.ndarray])->Union[float, np.ndarray]:
    
    return np.exp(x)
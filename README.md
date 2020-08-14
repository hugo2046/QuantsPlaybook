# Quantitative-analysis

## 利用python对国内各大券商的金工研报进行复现

数据依赖:[jqdata](https://www.joinquant.com/) 和 [tushare](https://tushare.pro/)

每个文件夹中有对于的券商研报及相关的论文,py文件中为ipynb的复现文档

## 目录

**指数择时类**

1. RSRS择时指标（光大证券）
2. 低延迟趋势线与交易择时（广发证券-主要使用广发开发的LLT均线，本质为一种低阶滤波器)
3. 基于相对强弱下单向波动差值应用（国信证券）
4. 扩散指标（东北证券)
5. 指数高阶矩择时(广发证券)
6. CSVC框架及熊牛指标（华泰证券)
    - a. CSVC为一个防过拟框架
    - b. 熊牛指标
7. 羊群效应(国泰君安)

**因子**

1. 基于量价关系度量股票的买卖压力(东方证券)
2. [来自优秀基金经理的超额收益(东方证券)](https://www.joinquant.com/view/community/detail/51d97afb8d619ffb5219d2e166414d70)

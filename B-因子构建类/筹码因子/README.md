# 说明

参考:

> 《广发证券_多因子Alpha系列报告之（二十七）——基于筹码分布的选股策略》

这篇并没太多创新,主要是汇总了关于筹码因子计算的相关算法,本文依赖qlib框架,相关算子在scr/cyq_ops.py和turnover_coefficient_ops.py中

1. 使用前景理论的换手率半衰期加权的筹码构建,这种方式是构建处置效应因子(CGO)的底层算法
2. 国内看盘软件上显示的筹码分布算法（源于陈浩-筹码分布,国外称之为CYQ）此算法又有两种类型
   1. 三角分布
   2. 平均分布

以上相关代码在scr/distribution_of_chips.py中。

*考虑到效率问题,并未使用scipy.stats.triang和uniform而是使用numba对其进行了重构。*

**数据获取**

这里打包了相关数据,cn_data为csv原始数据,qlib_data为处理后qlib可以直接使用的数据,factor_data是经*DatasetH*处理后的数据

链接：https://pan.baidu.com/s/14xbh7IJg7j7NRDA2qZcZvw?pwd=bpta 
提取码：bpta
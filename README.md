<!--
 * @Author: your name
 * @Date: 2022-04-17 00:54:11
 * @LastEditTime: 2022-05-23 09:31:56
 * @LastEditors: hugo2046 shen.lan123@gmail.com
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \undefinedd:\WrokSpace\Quantitative-analysis\README.md
-->
# Quantitative-analysis

## 利用python对国内各大券商的金工研报进行复现

数据依赖:[jqdata](https://www.joinquant.com/) 和 [tushare](https://tushare.pro/)

每个文件夹中有对应的券商研报及相关的论文,py文件中为ipynb的复现文档

## 目录

**指数择时类**

1. RSRS择时指标
    - [原始](https://www.joinquant.com/view/community/detail/1f0faa953856129e5826979ff9b68095)
    - [修正](https://www.joinquant.com/view/community/detail/32b60d05f16c7d719d7fb836687504d6)
    - [本土改造](https://www.joinquant.com/view/community/detail/e855e5b3cf6a3f9219583c2281e4d048)
2. [低延迟趋势线与交易择时（LLT均线，本质为一种低阶滤波器)](https://www.joinquant.com/view/community/detail/f011921f2398c593eee3542a6069f61c)
3. [基于相对强弱下单向波动差值应用](https://www.joinquant.com/view/community/detail/ddf35e24e9dbad456d3e6beaf0841262)
4. [扩散指标](https://www.joinquant.com/view/community/detail/aa69406f4427ea472b1c640fc2e8c448)
5. [指数高阶矩择时](https://www.joinquant.com/view/community/detail/e585df64077e4073ece0bcaa6b054bfa)
6. [CSVC框架及熊牛指标](https://www.joinquant.com/view/community/detail/6a77f468b6f996fcd995a8d0ad8c939c)
    - a. CSVC为一个防过拟框架
    - b. [熊牛指标](https://www.joinquant.com/view/community/detail/d0b0406c2ad2086662de715c92d518cd)
7. [羊群效应](https://www.joinquant.com/view/community/detail/3b4c68880062b3b660165bba7571d5a4)
8. [小波分析择时](https://www.joinquant.com/view/community/detail/eab0008b70882d0b1966bb6425db3469)
9. [时变夏普](https://www.joinquant.com/view/community/detail/634a7a14e79f87d44c980094c5e8d5d1)
10. [北向资金交易能力一定强吗](https://www.joinquant.com/view/community/detail/c11731e00f6de8e489ed64cec1621c33)
11. [择时视角下的波动率因子](https://www.joinquant.com/view/community/detail/986f2732c0b0287bc8f161829f32b689)
12. [趋与势的量化定义研究](https://www.joinquant.com/view/community/detail/9d12d9691b4201f95e4d0b99ada7676d)
13. [基于点位效率理论的个股趋势预测研究](https://www.joinquant.com/view/community/detail/f5d05b8233169adbbf44fb7522b2bf53)
14. [技术指标形态识别](https://www.joinquant.com/view/community/detail/1636a1cadab86dc65c65355fe431380c)
    - 复现《Foundations of Technical Analysis》
    - Technical Pattern Recognition文件：申万行业日度跟踪(Technical Pattern Recognition)
15. [C-VIX中国版VIX编制手册](https://www.joinquant.com/view/community/detail/787f5bf7ba5add2d5bc68e154046c10e)


**因子**

1. [基于量价关系度量股票的买卖压力](https://www.joinquant.com/view/community/detail/efc4f507b2ef8703d2c20283b1301980)
2. [来自优秀基金经理的超额收益](https://www.joinquant.com/view/community/detail/51d97afb8d619ffb5219d2e166414d70)
3. [市场微观结构研究系列（1）：A股反转之力的微观来源](https://www.joinquant.com/view/community/detail/521e854c0accab11c0bac2a9d8dac484)
4. [多因子指数增强的思路](https://www.joinquant.com/view/community/detail/8c60c343407d41b09def615c52c8693d)
5. [特质波动率因子](https://www.joinquant.com/view/community/detail/6e4ddf0a1cf3bb17367b463cefe3b5e4?type=1)
6. [处置效应因子](https://www.joinquant.com/view/community/detail/1c3aa95d7485065d977f9ba17cc014fd)
7. [技术因子-上下影线因子](https://www.joinquant.com/view/community/detail/92d2ccab2d412dbfa7df366369e6373b)
8. [聪明钱因子模型](https://www.joinquant.com/view/community/detail/fa281cadcbbca005854c7c45c3c9bd58)
9. [A股市场中如何构造动量因子？](https://www.joinquant.com/view/community/detail/d709c7c9abbee23149d3d4d07e128357)
10. [振幅因子的隐藏结构](https://www.joinquant.com/view/community/detail/a35fe484e3164893d4e48fafd3e08fd2)
11. [高质量动量因子选股](https://www.joinquant.com/view/community/detail/f72c599da7d4ca155b25bff4b281e2e6)
12. [APM因子改进模型](https://www.joinquant.com/view/community/detail/992fe40cc06c0bde50aa4aaf93fa042c)
13. [高频价量相关性，意想不到的选股因子](https://www.joinquant.com/view/community/detail/539e74507dbf571f2be21d8fa4ebb8e6)
14. ["因时制宜"系列研究之二：基于企业生命周期的因子有效性分析](https://www.joinquant.com/view/community/detail/6740756eee3287ae66cbb239a9c53479)
    1.  composition_factor算法来源于:《20190104-华泰证券-因子合成方法实证分析》
    2.  [IPCA](https://github.com/bkelly-lab/ipca)来源于[《Instrumented Principal Component Analysis》](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919)
15. [因子择时](https://www.joinquant.com/view/community/detail/a873b8ba2b510a228eac411eafb93bea)

**量化价值**

1. [罗伯·瑞克超额现金流选股法则](https://www.joinquant.com/view/community/detail/30543ad72454c7648b03bae542af55c9)
2. [华泰FFScore](https://www.joinquant.com/view/community/detail/c4bb321a8124ed575a66a88caf100b9f)

**组合优化**

1. [DE进化算法下的组合优化](https://www.joinquant.com/view/community/detail/2044ade4baf51132d257f2d3c0e56597)


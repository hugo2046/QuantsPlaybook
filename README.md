<!--
 * @Author: your name
 * @Date: 2022-04-17 00:54:11
 * @LastEditTime: 2022-07-15 09:09:48
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


    参考:
    > 《择时-20170501-光大证券-择时系列报告之一：基于阻力支撑相对强度（RSRS）的市场择时》
    >
    > 《20191117-光大证券-技术指标系列报告之六：RSRS择时~回顾与改进》

    <br>最新文章如下,光大金工团队转入中信,将RSRS更名为QRS:</br>
    > 《20210121-量化择时系列（1）：金融工程视角下的技术择时艺术》

2. [低延迟趋势线与交易择时（LLT均线，本质为一种低阶滤波器)](https://www.joinquant.com/view/community/detail/f011921f2398c593eee3542a6069f61c)
   <br>参考:</br>
   > 《20170303-广发证券-低延迟趋势线与交易择时》

3. [基于相对强弱下单向波动差值应用](https://www.joinquant.com/view/community/detail/ddf35e24e9dbad456d3e6beaf0841262)
   <br>参考:</br>
   > 《20151022-国信证券-市场波动率研究：基于相对强弱下单向波动差值应用》

4. [扩散指标](https://www.joinquant.com/view/community/detail/aa69406f4427ea472b1c640fc2e8c448)
   <br>参考:</br>
   > 《择时-20190924-东北证券-金融工程研究报告：扩散指标择时研究之一，基本用法》

5. [指数高阶矩择时](https://www.joinquant.com/view/community/detail/e585df64077e4073ece0bcaa6b054bfa)
   <br>参考:</br>
   >《20150520-广发证券-交易性择时策略研究之八：指数高阶矩择时策略》

6. [CSVC框架及熊牛指标](https://www.joinquant.com/view/community/detail/6a77f468b6f996fcd995a8d0ad8c939c)
    - a. CSVC为一个防过拟框架
    - b. [熊牛指标](https://www.joinquant.com/view/community/detail/d0b0406c2ad2086662de715c92d518cd)
    <br>参考:</br>
    >《The Probability of Backtest Overfitting》
    >
    >《20190617-华泰证券-华泰人工智能系列之二十二：基于CSCV框架的回测过拟合概率》
    >
    >《20200407-华泰证券-华泰金工量化择时系列：牛熊指标在择时轮动中的应用探讨》
    >
    >《择时-20190927-华泰证券-华泰金工量化择时系列：波动率与换手率构造牛熊指标》

7. [基于CCK模型的股票市场羊群效应研究](https://www.joinquant.com/view/community/detail/3b4c68880062b3b660165bba7571d5a4)
   <br>参考:</br>
   >《20181128-国泰君安-数量化专题之一百二十二：基于CCK模型的股票市场羊群效应研究》

8. [小波分析择时](https://www.joinquant.com/view/community/detail/eab0008b70882d0b1966bb6425db3469)
    <br>参考</br>
    >《20100621-国信证券-基于小波分析和支持向量机的指数预测模型》
    >
    >《20120220-平安证券-量化择时选股系列报告二：水致清则鱼自现_小波分析与支持向量机择时研究》

9. [时变夏普](https://www.joinquant.com/view/community/detail/634a7a14e79f87d44c980094c5e8d5d1)
    <br>参考:</br>
    >《20101028-国海证券-新量化择时指标之二：时变夏普比率把握长中短趋势》
    >
    >《20120726-国信证券-时变夏普率的择时策略》
    >
    >《sharpe2-1997》
    >
    >《The Applicability of Time-varying Sharpe Ratio to Chinese》
    >
    >《tvsharpe》
    >
    >《varcov jf94-1994》

10. [北向资金交易能力一定强吗](https://www.joinquant.com/view/community/detail/c11731e00f6de8e489ed64cec1621c33)
    <br>参考:</br>
    >《20200624-安信证券-金融工程主题报告：北向资金交易能力一定强吗》

11. [择时视角下的波动率因子](https://www.joinquant.com/view/community/detail/986f2732c0b0287bc8f161829f32b689)

12. [趋与势的量化定义研究](https://www.joinquant.com/view/community/detail/9d12d9691b4201f95e4d0b99ada7676d)
     <br>参考:</br>
     >《数量化专题之六十四_趋与势的量化定义研究_2015-08-10_国泰君安》

13. [基于点位效率理论的个股趋势预测研究](https://www.joinquant.com/view/community/detail/f5d05b8233169adbbf44fb7522b2bf53)
     <br>参考:</br>
     >《20210917-兴业证券-花开股市，相似几何系列二：基于点位效率理论的个股趋势预测研究》
     >
     >《20211007-兴业证券-花开股市、相似几何系列三：基于点位效率理论的量化择时体系搭建》

14. [技术指标形态识别](https://www.joinquant.com/view/community/detail/1636a1cadab86dc65c65355fe431380c)
     <br>参考:</br>
     > 《Foundations of Technical Analysis》
     > 《20210831_中泰证券_破解“看图”之谜：技术分析算法、框架与实战》
     - Technical Pattern Recognition文件：申万行业日度跟踪(Technical Pattern Recognition)

15. [C-VIX中国版VIX编制手册](https://www.joinquant.com/view/community/detail/787f5bf7ba5add2d5bc68e154046c10e)
     参考:
     >《20140331-国信证券-衍生品应用与产品设计系列之vix介绍及gsvx编制》
     >
     >《20180707_东北证券_金融工程_市场波动风险度量：vix与skew指数构建与应用》
     >
     >《20191210-东海证券-VIX及SKEW指数的构建、分析与预测》
     >
     >《20200317_浙商证券_金融工程_衍生品系列（一）：c-vix：中国版vix编制手册》
     >
     >《vixwhite》

     本文主要参考《20180707_东北证券_金融工程_市场波动风险度量：vix与skew指数构建与应用》

16. [特征分布建模择时](https://www.joinquant.com/view/community/detail/17e97079ece6d76c85fb5f3aa62acdc0)
     <br>参考:</br>
     >《2022-06-17_华创证券_金融工程_特征分布建模择时系列之一：物极必反，龙虎榜机构模型》

17. Trader-Company集成算法交易策略

     参考:

     > 《Trader-Company Method A Metaheuristic for Interpretable Stock Price Prediction》
     >
     > 《20220517_浙商证券_金融工程_一种自适应寻找市场alpha的方法：“trader-company”集成算法交易策略》

18. [特征分布建模择时系列之二：特征成交量](https://www.joinquant.com/view/community/detail/25eef9ac283dd0592359c0eb25e18247)

     参考:

     > 《20220805*华创证券*宏观研究_特征分布建模择时系列之二：物极必反，巧妙做空，特征成交量，模型终完备》

**因子**

1. [基于量价关系度量股票的买卖压力](https://www.joinquant.com/view/community/detail/efc4f507b2ef8703d2c20283b1301980)
   <br>参考:</br>
   
   >《20191029-东方证券- 因子选股系列研究六十：基于量价关系度量股票的买卖压力》
2. [来自优秀基金经理的超额收益](https://www.joinquant.com/view/community/detail/51d97afb8d619ffb5219d2e166414d70)
   <br>参考:</br>
   >《20190115-东方证券-因子选股系列之五十：A股行业内选股分析总结》
   >
   >《20191127-东方证券-《因子选股系列研究之六十二》：来自优秀基金经理的超额收益》
   >
   >《20200528-东方证券-金融工程专题报告：东方A股因子风险模型（DFQ~2020）》
   >
   >《20200707-海通证券-选股因子系列研究（六十八）：基金重仓超配因子及其对指数增强组合的影响》
3. [市场微观结构研究系列（1）：A股反转之力的微观来源](https://www.joinquant.com/view/community/detail/521e854c0accab11c0bac2a9d8dac484)
   <br>参考:</br>
   >《20191223-开源证券-市场微观结构研究系列（1）：A股反转之力的微观来源》
4. [多因子指数增强的思路](https://www.joinquant.com/view/community/detail/8c60c343407d41b09def615c52c8693d)
   <br>参考:</br>
   >《【华泰金工】指数增强方法汇总及实例20180531》
   >
   >《20180705-天风证券-金工专题报告：基于自适应风险控制的指数增强策略》
5. [特质波动率因子](https://www.joinquant.com/view/community/detail/6e4ddf0a1cf3bb17367b463cefe3b5e4?type=1)
   <br>参考:</br>
   >《20200528-东吴证券-“波动率选股因子”系列研究（一）：寻找特质波动率中的纯真信息，剔除跨期截面相关性的纯真波动率因子》
6. [处置效应因子](https://www.joinquant.com/view/community/detail/1c3aa95d7485065d977f9ba17cc014fd)
   <br>参考:</br>
   >《20170707-广发证券-行为金融因子研究之一：资本利得突出量CGO与风险偏好》
   >
   >《20190531-国信证券-行为金融学系列之二：处置效应与新增信息参与定价的反应迟滞》
7. [技术因子-上下影线因子](https://www.joinquant.com/view/community/detail/92d2ccab2d412dbfa7df366369e6373b)
   <br>参考:</br>
   >《20200619-东吴证券-“技术分析拥抱选股因子”系列研究（二）：上下影线，蜡烛好还是威廉好》
8. [聪明钱因子模型](https://www.joinquant.com/view/community/detail/fa281cadcbbca005854c7c45c3c9bd58)
   <br>参考:</br>
   >《20200209-开源证券-市场微观结构研究系列（3）：聪明钱因子模型的2.0版本》
9.  [A股市场中如何构造动量因子？](https://www.joinquant.com/view/community/detail/d709c7c9abbee23149d3d4d07e128357)
    <br>参考:</br>
    >《20200721-开源证券-开源量化评论（3）：A股市场中如何构造动量因子？》
10. [振幅因子的隐藏结构](https://www.joinquant.com/view/community/detail/a35fe484e3164893d4e48fafd3e08fd2)
    <br>参考:</br>
    >《20200516-开源证券-市场微观结构研究系列（7）：振幅因子的隐藏结构》
11. [高质量动量因子选股](https://www.joinquant.com/view/community/detail/f72c599da7d4ca155b25bff4b281e2e6)
12. [APM因子改进模型](https://www.joinquant.com/view/community/detail/992fe40cc06c0bde50aa4aaf93fa042c)
    <br>参考:</br>
    >《20200307-开源证券-市场微观结构研究系列（5）：APM因子模型的进阶版》
13. [高频价量相关性，意想不到的选股因子](https://www.joinquant.com/view/community/detail/539e74507dbf571f2be21d8fa4ebb8e6)
    <br>参考:</br>
    >《20200223_东吴证券_“技术分析拥抱选股因子”系列研究（一）：高频价量相关性，意想不到的选股因子》
14. ["因时制宜"系列研究之二：基于企业生命周期的因子有效性分析](https://www.joinquant.com/view/community/detail/6740756eee3287ae66cbb239a9c53479)
    1.  composition_factor算法来源于:《20190104-华泰证券-因子合成方法实证分析》
    2.  [IPCA](https://github.com/bkelly-lab/ipca)来源于[《Instrumented Principal Component Analysis》](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919)
15. [因子择时](https://www.joinquant.com/view/community/detail/a873b8ba2b510a228eac411eafb93bea)

**量化价值**

1. [罗伯·瑞克超额现金流选股法则](https://www.joinquant.com/view/community/detail/30543ad72454c7648b03bae542af55c9)
   <br>参考:</br>
   >《20151019-申万宏源-申万大师系列.价值投资篇之十三：罗伯.瑞克超额现金流选股法则》
2. [华泰FFScore](https://www.joinquant.com/view/community/detail/c4bb321a8124ed575a66a88caf100b9f)
    <br>参考:</br>
    >《20170209-华泰证券-华泰价值选股之FFScore模型：比乔斯基选股模型A股实证研究》

**组合优化**

1. [DE进化算法下的组合优化](https://www.joinquant.com/view/community/detail/2044ade4baf51132d257f2d3c0e56597)
   <br>参考:</br>
   >《20191018-浙商证券-人工智能系列（二）：人工智能再出发，次优理论下的组合配置与策略构建》
   >《20191101-浙商证券-FOF组合系列（一）：回撤最小目标下的偏债FOF组合构建以，一家公募产品为例》


## 请我喝杯咖啡吧

![image](https://raw.githubusercontent.com/hugo2046/Quantitative-analysis/master/coffee.png)


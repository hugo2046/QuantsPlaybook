<!--

 * @Author: your name
 * @Date: 2022-04-17 00:54:11
 * @LastEditTime: 2022-12-09 16:08:37
 * @LastEditors: hugo2046 shen.lan123@gmail.com
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \undefinedd:\WrokSpace\Quantitative-analysis\README.md
-->
# Quantitative-analysis

## 利用python对国内各大券商的金工研报进行复现

数据依赖:[jqdata](https://www.joinquant.com/) 和 [tushare](https://tushare.pro/)

每个文件夹中有对应的券商研报及相关的论文,py文件中为ipynb的复现文档

## 目录
<table>
    <tboday>
        <tr>
            <th>类别</th>
            <th>名称</th>
            <th>参考</th>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td>RSRS择时指标</td>
            <td>
            <a href="https://www.joinquant.com/view/community/detail/1f0faa953856129e5826979ff9b68095">原始版本</a>
                        <br><a href="https://www.joinquant.com/view/community/detail/32b60d05f16c7d719d7fb836687504d6">修正版本</a></br>
                                    <a href="https://www.joinquant.com/view/community/detail/e855e5b3cf6a3f9219583c2281e4d048">本土改造版本</a>
            <li>《择时-20170501-光大证券-择时系列报告之一：基于阻力支撑相对强度（RSRS）的市场择时》</li>
                <li>《20191117-光大证券-技术指标系列报告之六：RSRS择时~回顾与改进》</li>
                <li>最新文章如下,光大金工团队转入中信,将RSRS更名为QRS:</li>
                <li>《20210121-量化择时系列（1）：金融工程视角下的技术择时艺术》</li>
            </td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/f011921f2398c593eee3542a6069f61c">低延迟趋势线与交易择时</a></td>
            <td>《20170303-广发证券-低延迟趋势线与交易择时》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/ddf35e24e9dbad456d3e6beaf0841262">基于相对强弱下单向波动差值应用</a></td>
            <td>《20151022-国信证券-市场波动率研究：基于相对强弱下单向波动差值应用》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/aa69406f4427ea472b1c640fc2e8c448">扩散指标</a></td>
            <td>《择时-20190924-东北证券-金融工程研究报告：扩散指标择时研究之一，基本用法》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/e585df64077e4073ece0bcaa6b054bfa">指数高阶矩择时</a></td>
            <td>《20150520-广发证券-交易性择时策略研究之八：指数高阶矩择时策略》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/6a77f468b6f996fcd995a8d0ad8c939c">CSVC框架及熊牛指标</a></td>
            <td><strong>CSVC防过拟框架</strong>
                <br><strong><a href="https://www.joinquant.com/view/community/detail/d0b0406c2ad2086662de715c92d518cd">熊牛线指标构建</a></strong></br>
                相关论文
                <li>《The Probability of Backtest Overfitting》</li>
                相关研报
                <li>《20190617-华泰证券-华泰人工智能系列之二十二：基于CSCV框架的回测过拟合概率》</li>
                <li>《20200407-华泰证券-华泰金工量化择时系列：牛熊指标在择时轮动中的应用探讨》</li>
                <li>《择时-20190927-华泰证券-华泰金工量化择时系列：波动率与换手率构造牛熊指标》</li>
            </td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/3b4c68880062b3b660165bba7571d5a4">基于CCK模型的股票市场羊群效应研究</a></td>
            <td>《20181128-国泰君安-数量化专题之一百二十二：基于CCK模型的股票市场羊群效应研究》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/eab0008b70882d0b1966bb6425db3469">小波分析择时</a></td>
            <td>
                <br>《20100621-国信证券-基于小波分析和支持向量机的指数预测模型》</br>
                《20120220-平安证券-量化择时选股系列报告二：水致清则鱼自现_小波分析与支持向量机择时研究》
            </td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/634a7a14e79f87d44c980094c5e8d5d1">时变夏普</a></td>
            <td>相关研报
                <li>《20101028-国海证券-新量化择时指标之二：时变夏普比率把握长中短趋势》</li>
                <li>《20120726-国信证券-时变夏普率的择时策略》</li>
                相关论文
                <li>《sharpe2-1997》</li>
                相关研报
                <li>《The Applicability of Time-varying Sharpe Ratio to Chinese》</li>
                <li>《tvsharpe》</li>
                <li>《varcov jf94-1994》</li>
            </td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/c11731e00f6de8e489ed64cec1621c33">北向资金交易能力一定强吗</a></td>
            <td>《20200624-安信证券-金融工程主题报告：北向资金交易能力一定强吗》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/986f2732c0b0287bc8f161829f32b689">择时视角下的波动率因子</a></td>
            <td>无</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/9d12d9691b4201f95e4d0b99ada7676d">趋与势的量化定义研究</a></td>
            <td>《数量化专题之六十四_趋与势的量化定义研究_2015-08-10_国泰君安》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/f5d05b8233169adbbf44fb7522b2bf53">基于点位效率理论的个股趋势预测研究</a></td>
            <td>
                <ur>
                <li>《20210917-兴业证券-花开股市，相似几何系列二：基于点位效率理论的个股趋势预测研究》</li>
                <li>《20211007-兴业证券-花开股市、相似几何系列三：基于点位效率理论的量化择时体系搭建》</li>
                </ur>
            </td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/1636a1cadab86dc65c65355fe431380c">技术指标形态识别</a></td>
            <td>
                <ur>
                相关论文
                <br>《Foundations of Technical Analysis》</br>
                相关研报
                <br>《20210831_中泰证券_破解“看图”之谜：技术分析算法、框架与实战》</br>
                </ur>
            </td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/787f5bf7ba5add2d5bc68e154046c10e">C-VIX中国版VIX编制手册</a></td>
            <td>
                <ur>
                <li>《20140331-国信证券-衍生品应用与产品设计系列之vix介绍及gsvx编制》</li>
                <li>《20180707_东北证券_金融工程_市场波动风险度量：vix与skew指数构建与应用》</li>
                <li>《20191210-东海证券-VIX及SKEW指数的构建、分析与预测》</li>
                <li>《20200317_浙商证券_金融工程_衍生品系列（一）：c-vix：中国版vix编制手册》</li>
                </ur>
            </td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/17e97079ece6d76c85fb5f3aa62acdc0">特征分布建模择时</a></td>
            <td>《2022-06-17_华创证券_金融工程_特征分布建模择时系列之一：物极必反，龙虎榜机构模型》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/25eef9ac283dd0592359c0eb25e18247">特征分布建模择时系列之二：特征成交量</a></td>
            <td>《20220805华创证券宏观研究_特征分布建模择时系列之二：物极必反，巧妙做空，特征成交量，模型终完备》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td>Trader-Company集成算法交易策略</td>
            <td>
                <ur>
                相关论文
                <br>《Trader-Company Method A Metaheuristic for Interpretable Stock Price Prediction》</br>
                相关研报
                <br>《20220517_浙商证券_金融工程_一种自适应寻找市场alpha的方法：“trader-company”集成算法交易策略》</br>
                </ur>
            </td>
        </tr>
		<tr>
            <td><strong>择时</strong></td>
            <td>成交量的奥秘：另类价量共振指标的择时</td>
            <td>《2019-02-22_华创证券_金融工程_成交量的奥秘：另类价量共振指标的择时》</td>
        </tr>
        <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://www.joinquant.com/view/community/detail/42bd804bcf72f0a086b442f6769fd3ad">均线交叉结合通道突破择时研究</a></td>
            <td>《20180410-申万宏源-均线交叉结合通道突破择时研究》</td>
        </tr>
		<tr>
         <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://nbviewer.org/github/hugo2046/QuantsPlaybook/blob/dev/C-%E6%8B%A9%E6%97%B6%E7%B1%BB/%E6%8A%95%E8%B5%84%E8%80%85%E6%83%85%E7%BB%AA%E6%8C%87%E6%95%B0%E6%8B%A9%E6%97%B6%E6%A8%A1%E5%9E%8B/%E6%8A%95%E8%B5%84%E8%80%85%E6%83%85%E7%BB%AA%E6%8C%87%E6%95%B0%E6%8B%A9%E6%97%B6%E6%A8%A1%E5%9E%8B.ipynb">投资者情绪指数择时模型</a></td>
            <td>《20140804_国信证券_量化择时系列报告之二：国信投资者情绪指数择时模型》</td>
        </tr>
		<tr>
            <tr>
         <tr>
            <td><strong>择时</strong></td>
            <td><a href="https://nbviewer.org/github/hugo2046/QuantsPlaybook/blob/ea5bf8d7c20587db4a64b34af6c4d89def99747e/C-%E6%8B%A9%E6%97%B6%E7%B1%BB/%E8%A1%8C%E4%B8%9A%E6%8C%87%E6%95%B0%E9%A1%B6%E9%83%A8%E5%92%8C%E5%BA%95%E9%83%A8%E4%BF%A1%E5%8F%B7/%E8%A1%8C%E4%B8%9A%E6%8C%87%E6%95%B0%E9%A1%B6%E9%83%A8%E5%92%8C%E5%BA%95%E9%83%A8%E4%BF%A1%E5%8F%B7.ipynb">行业指数顶部和底部信号</a></td>
            <td>《华福证券-市场情绪指标专题（五）：行业指数顶部和底部信号，净新高占比（（NH~NL）%）-230302》</td>
        </tr>
		<tr>
            <td><strong>择时</strong></td>
            <td><a href="https://github.com/hugo2046/QuantsPlaybook/blob/398d0cc5732d9d1c0f26768b8fad8c2e6617d250/C-%E6%8B%A9%E6%97%B6%E7%B1%BB/ICU%E5%9D%87%E7%BA%BF/ICU_MA.ipynb">ICU均线</a></td>
            <td>《20230412_中泰证券_“均线”才是绝对收益利器-ICU均线下的择时策略》</td>
        </tr>
		<tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/efc4f507b2ef8703d2c20283b1301980">基于量价关系度量股票的买卖压力</a></td>
        <td>
            《20191029-东方证券- 因子选股系列研究六十：基于量价关系度量股票的买卖压力》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/51d97afb8d619ffb5219d2e166414d70">来自优秀基金经理的超额收益</a></td>
        <td>
            <li>《20190115-东方证券-因子选股系列之五十：A股行业内选股分析总结》</li>
            <li>《20191127-东方证券-《因子选股系列研究之六十二》：来自优秀基金经理的超额收益》</li>
            <li>《20200528-东方证券-金融工程专题报告：东方A股因子风险模型（DFQ~2020）》</li>
            <li>《20200707-海通证券-选股因子系列研究（六十八）：基金重仓超配因子及其对指数增强组合的影响》</li>
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/521e854c0accab11c0bac2a9d8dac484">市场微观结构研究系列（1）：A股反转之力的微观来源</a></td>
        <td>
            《20191223-开源证券-市场微观结构研究系列（1）：A股反转之力的微观来源》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/8c60c343407d41b09def615c52c8693d">多因子指数增强的思路</a></td>
        <td>
            <li>《【华泰金工】指数增强方法汇总及实例20180531》</li>
            <li>《20180705-天风证券-金工专题报告：基于自适应风险控制的指数增强策略》</li>
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/6e4ddf0a1cf3bb17367b463cefe3b5e4?type=1">特质波动率因子</a></td>
        <td>
            20200528-东吴证券-“波动率选股因子”系列研究（一）：寻找特质波动率中的纯真信息，剔除跨期截面相关性的纯真波动率因子》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/1c3aa95d7485065d977f9ba17cc014fd">处置效应因子</a></td>
        <td>
            <li>《20170707-广发证券-行为金融因子研究之一：资本利得突出量CGO与风险偏好》</li>
            <li>《20190531-国信证券-行为金融学系列之二：处置效应与新增信息参与定价的反应迟滞》</li>
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/92d2ccab2d412dbfa7df366369e6373b">技术因子-上下影线因子</a></td>
        <td>
            《20200619-东吴证券-“技术分析拥抱选股因子”系列研究（二）：上下影线，蜡烛好还是威廉好》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/fa281cadcbbca005854c7c45c3c9bd58">聪明钱因子模型</a></td>
        <td>
            《20200209-开源证券-市场微观结构研究系列（3）：聪明钱因子模型的2.0版本》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/d709c7c9abbee23149d3d4d07e128357">A股市场中如何构造动量因子?</a></td>
        <td>
            《20200721-开源证券-开源量化评论（3）：A股市场中如何构造动量因子？》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/a35fe484e3164893d4e48fafd3e08fd2">振幅因子的隐藏结构</a></td>
        <td>
            《20200516-开源证券-市场微观结构研究系列（7）：振幅因子的隐藏结构》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/f72c599da7d4ca155b25bff4b281e2e6">高质量动量因子选股</a></td>
        <td>
            图书《构建量化动量选股系统的实用指南》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/992fe40cc06c0bde50aa4aaf93fa042c">APM因子改进模型</a></td>
        <td>
            《20200307-开源证券-市场微观结构研究系列（5）：APM因子模型的进阶版》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/539e74507dbf571f2be21d8fa4ebb8e6">高频价量相关性，意想不到的选股因子</a></td>
        <td>
            《20200223_东吴证券_“技术分析拥抱选股因子”系列研究（一）：高频价量相关性，意想不到的选股因子》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/6740756eee3287ae66cbb239a9c53479">"因时制宜"系列研究之二：基于企业生命周期的因子有效性分析</a></td>
        <td>
            <strong>composition_factor算法来源于</strong>
            <li>《20190104-华泰证券-因子合成方法实证分析》</li>
            <strong><a href="https://github.com/bkelly-lab/ipca">IPCA</a></strong>源于
            <li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919">《Instrumented Principal Component Analysis》</a></li>
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/a873b8ba2b510a228eac411eafb93bea">因子择时</a></td>
        <td>
            来自于:光大证券路演
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/39135">分析师推荐概率增强金股组合策略</a></td>
        <td>
            《20220822_浙商证券_投资策略_金融工程深度：金股数据库及金股组合增强策略（一）》
        </td>
    </tr>
<tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://nbviewer.org/github/hugo2046/QuantsPlaybook/blob/ecb97803a7c1e40bca6555fa41ff093439a81a55/B-%E5%9B%A0%E5%AD%90%E6%9E%84%E5%BB%BA%E7%B1%BB/%E8%A1%8C%E4%B8%9A%E6%9C%89%E6%95%88%E9%87%8F%E4%BB%B7%E5%9B%A0%E5%AD%90%E4%B8%8E%E8%A1%8C%E4%B8%9A%E8%BD%AE%E5%8A%A8%E7%AD%96%E7%95%A5/%E8%A1%8C%E4%B8%9A%E6%9C%89%E6%95%88%E9%87%8F%E4%BB%B7%E5%9B%A0%E5%AD%90%E4%B8%8E%E8%A1%8C%E4%B8%9A%E8%BD%AE%E5%8A%A8%E7%AD%96%E7%95%A5ETF.ipynb">行业有效量价因子与行业轮动策略</a></td>
        <td>
            《【华西证券】金融工程研究报告：行业有效量价因子与行业轮动策略》
        </td>
    </tr>
    <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://github.com/hugo2046/QuantsPlaybook/blob/master/B-%E5%9B%A0%E5%AD%90%E6%9E%84%E5%BB%BA%E7%B1%BB/%E7%AD%B9%E7%A0%81%E5%9B%A0%E5%AD%90/%E7%AD%B9%E7%A0%81%E5%88%86%E5%B8%83%E5%9B%A0%E5%AD%90.ipynb">筹码分布因子</a></td>
        <td>
            《广发证券_多因子Alpha系列报告之（二十七）——基于筹码分布的选股策略》
        </td>
    </tr>
     <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://github.com/hugo2046/QuantsPlaybook/blob/master/B-%E5%9B%A0%E5%AD%90%E6%9E%84%E5%BB%BA%E7%B1%BB/%E5%87%B8%E6%98%BE%E7%90%86%E8%AE%BASTR%E5%9B%A0%E5%AD%90/%E5%87%B8%E6%98%BE%E5%BA%A6%E5%9B%A0%E5%AD%90.ipynb">凸显度因子</a></td>
        <td>
            <li>《20221213_方大证券_显著效应、极端收益扭曲决策权重和“草木皆兵”因子》</li>
            <li>《20221214-招商证券-“青出于蓝”系列研究之四：行为金融新视角，“凸显性收益”因子STR》</li>
            <li>《20230323_广发证券_行为金融研究系列之七_凸显理论之 A 股“价”“量”应用》</li>
            <li>《Salience theory and stock prices Empirical evidence》</li>
            <li>《SalientStocksFMA2017》</li>
        </td>
    </tr>
     <tr>
        <td><strong>因子构建</strong></td>
        <td><a href="https://github.com/hugo2046/QuantsPlaybook/blob/master/B-%E5%9B%A0%E5%AD%90%E6%9E%84%E5%BB%BA%E7%B1%BB/%E4%B8%AA%E8%82%A1%E5%8A%A8%E9%87%8F%E6%95%88%E5%BA%94%E7%9A%84%E8%AF%86%E5%88%AB%E5%8F%8A%E7%90%83%E9%98%9F%E7%A1%AC%E5%B8%81%E5%9B%A0%E5%AD%90/%E7%90%83%E9%98%9F%E7%A1%AC%E5%B8%81%E5%9B%A0%E5%AD%90.ipynb">球队硬币因子</a></td>
        <td>
            <li>《20220611-方正证券-多因子选股系列研究之四：个股动量效应识别及“球队硬币”因子构建》</li>
            <li>《Moskowitz T J. Asset pricing and sports betting[J]. Journal of Finance, Forthcoming, 2021.》</li>
        </td>
    </tr>
    <tr>
        <td><strong>量化价值</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/30543ad72454c7648b03bae542af55c9">罗伯·瑞克超额现金流选股法则</a></td>
        <td>
            《20151019-申万宏源-申万大师系列.价值投资篇之十三：罗伯.瑞克超额现金流选股法则》
        </td>
    </tr>
    <tr>
        <td><strong>量化价值</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/c4bb321a8124ed575a66a88caf100b9f">华泰FFScore</a></td>
        <td>
            《20170209-华泰证券-华泰价值选股之FFScore模型：比乔斯基选股模型A股实证研究》
        </td>
    </tr>
    <tr>
        <td><strong>组合优化</strong></td>
        <td><a href="https://www.joinquant.com/view/community/detail/2044ade4baf51132d257f2d3c0e56597">DE进化算法下的组合优化</a></td>
        <td>
            <li>《20191101-浙商证券-FOF组合系列（一）：回撤最小目标下的偏债FOF组合构建以，一家公募产品为例》</li>
            <li>《20191018-浙商证券-人工智能系列（二）：人工智能再出发，次优理论下的组合配置与策略构建》</li>
        </td>
    </tr>
<tr>
        <td><strong>组合优化</strong></td>
        <td><a href="https://github.com/hugo2046/QuantsPlaybook/blob/master/D-%E7%BB%84%E5%90%88%E4%BC%98%E5%8C%96/MLT_TSMOM/mlt_tsmom.ipynb">多任务时序动量策略</a></td>
        <td>
            <li><a href="https://arxiv.org/abs/2306.13661">《Constructing Time-Series Momentum Portfolios with Deep Multi-Task Learning》</a></li>
        </td>
    </tr>
</tboday>
</table>





## 请我喝杯咖啡吧

![image](https://raw.githubusercontent.com/hugo2046/Quantitative-analysis/master/coffee.png)


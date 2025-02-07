<!--
 * @Author: Hugo
 * @Date: 2024-10-25 13:14:49
 * @LastEditors: shen.lan123@gmail.com
 * @LastEditTime: 2025-02-06 13:55:49
 * @Description: 
-->
# SignalMaker

择时信号构造

## HHT择时信号

### HT信号

```python
from SignalMaker.hht_signal import get_ht_signal

# hs300_df中必须有close为DataFrame类型
get_ht_signal(hs300_df,60,30)
```

### HHT信号

```python
from SignalMaker.hht_signal import get_hht_signal

# hs300_df中必须有close为DataFrame类型
## 采用EMD算法
get_ht_signal(hs300_df,60, 2, 9, "EMD")

## 采用VMD算法
get_hht_signal(hs300_data, 60, 2, 9, "VMD")
```

# NoiseArea择时信号

研究文档:[另类ETF交易策略：日内动量](https://github.com/hugo2046/QuantsPlaybook/blob/master/C-%E6%8B%A9%E6%97%B6%E7%B1%BB/%E5%8F%A6%E7%B1%BBETF%E4%BA%A4%E6%98%93%E7%AD%96%E7%95%A5%EF%BC%9A%E6%97%A5%E5%86%85%E5%8A%A8%E9%87%8F/etf_mom_strategy.ipynb)

```python
from SignalMaker.noise_area import NoiseArea

# 使用指数分钟数据生成信号
## index_price需要有OHLCV
index_signal: pd.DataFrame = NoiseArea(index_price).fit(14)
# 返回为DataFrame:ubound,signal,lbound
```

# QRS择时信号

研究文档：[QRS](https://github.com/hugo2046/QuantsPlaybook/blob/master/C-%E6%8B%A9%E6%97%B6%E7%B1%BB/QRS%E6%8B%A9%E6%97%B6%E4%BF%A1%E5%8F%B7/QRS.ipynb)

```python
from SignalMaker.qrs import QRSCreator

# low_df,high_df结构为DataFrame,index-date,columns-code,values
qrs:QRSCreator = QRSCreator(low_df, high_df)
signal_df:pd.DataFrame = qrs.fit(18,600)

```

# 鳄鱼线择时信号

研究文档:[基于鳄鱼线的指数择时及轮动策略](https://github.com/hugo2046/QuantsPlaybook/blob/master/C-%E6%8B%A9%E6%97%B6%E7%B1%BB/%E5%9F%BA%E4%BA%8E%E9%B3%84%E9%B1%BC%E7%BA%BF%E7%9A%84%E6%8C%87%E6%95%B0%E6%8B%A9%E6%97%B6%E5%8F%8A%E8%BD%AE%E5%8A%A8%E7%AD%96%E7%95%A5/zs_timing_strategy.ipynb)

```python
from SignalMaker.alligator_indicator_timing import get_alligator_signal,get_ao_indicator_signal,get_macd_signal

# 鳄鱼线
## index-date columns-code values-close
alligator_signal: pd.DataFrame = get_alligator_signal(close_df)

# 动量震荡指标 (AO)
## index-date columns-code values-high,low
ao_signal: pd.DataFrame = get_ao_indicator_signal(high_df, low_df)

# MACD信号
macd_signal:pd.DataFrame = get_macd_signal(close_df)
```


# analyze.py 使用指南

**版本**: 1.1.0
**更新日期**: 2025-12-25
**作者**: Hugo

## 更新日志

- **v1.1.0** (2025-12-25): 新增灵活的数据过滤控制功能
  - 添加 `filter_suspended` 和 `filter_limit` 参数
  - 支持自定义停牌和涨跌停股票的过滤行为
  - 更新文档和使用示例

- **v1.0.0** (2025-12-24): 初始版本
  - 完整的因子分析功能
  - factor_group_analysis() 便捷函数
  - FactorAnalyzer 分析器类

## 目录

1. [快速开始](#快速开始)
2. [接口对比](#接口对比)
3. [数据过滤控制](#数据过滤控制) ⭐ **新增**
4. [使用场景](#使用场景)
5. [完整示例](#完整示例)
6. [最佳实践](#最佳实践)
7. [常见问题](#常见问题)
8. [API参考](#api参考)

---

## 快速开始

### 5分钟上手示例

```python
from src.analyze import factor_group_analysis

# 1. 准备因子数据（MultiIndex [date, instrument]）
# 假设已经有 factor_data，包含 'scc' 列
factor_data = ...  # 你的因子数据

# 2. 一键分析
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',      # 因子列名
    group_count=10         # 分10组
)

# 3. 查看结果
print("分组收益：")
print(pivot_df.head())
```

**输出结果**：
```
分组收益：
factor_quantile    1    2    3   ...   10
date
2020-01-21      0.02 0.01 0.00 ... -0.01
2020-01-22      0.01 0.02 0.01 ...  0.00
...
```

---

## 接口对比

### 方式1：便捷函数 `factor_group_analysis()`

**适合**：快速分析、一次性计算

**特点**：
- ✅ **简单**：一行代码完成分析
- ✅ **自动推断**：自动从数据推断股票池和日期范围
- ✅ **零配置**：无需手动创建对象

**示例**：
```python
from src.analyze import factor_group_analysis

# 单次分析
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    group_count=10
)
```

### 方式2：分析器类 `FactorAnalyzer`

**适合**：批量计算、需要更多控制

**特点**：
- ✅ **可重用**：同一实例可分析多个因子
- ✅ **可扩展**：可访问中间步骤，自定义流程
- ✅ **高效**：可重用 data_provider，减少初始化开销

**示例**：
```python
from src.analyze import FactorAnalyzer

# 创建分析器
analyzer = FactorAnalyzer()

# 批量分析多个因子
factors = ['scc', 'tcc', 'cc']
results = {}

for factor_col in factors:
    pred_label_df, pivot_df = analyzer.analyze(
        factor_data[[factor_col]],  # 只传入需要的列
        factor_col=factor_col
    )
    results[factor_col] = pivot_df

# 比较结果
for factor_name, pivot_df in results.items():
    print(f"{factor_name} 因子 - 第1组收益: {pivot_df[1].mean():.4f}")
```

### 对比表

| 特性 | `factor_group_analysis()` | `FactorAnalyzer` |
|------|--------------------------|------------------|
| **代码量** | 1行 | 3-5行 |
| **适用场景** | 单次分析、快速验证 | 批量分析、复杂流程 |
| **可重用性** | ❌ 每次创建新实例 | ✅ 同一实例多次使用 |
| **自定义** | 基础控制 | ✅ 可访问中间步骤 |
| **过滤控制** | ✅ 支持参数控制 | ✅ 灵活控制停牌和涨跌停 |
| **性能** | 适合小规模 | 适合大规模（重用provider） |
| **学习曲线** | 低 | 中 |

**注意**：两者现在都支持灵活的过滤控制！

---

## 数据过滤控制

### 概述

`FactorAnalyzer` 类支持灵活的数据过滤控制，允许用户自主选择是否过滤停牌和涨跌停股票。

### 过滤参数

在创建 `FactorAnalyzer` 实例时，可以通过以下参数控制过滤行为：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `filter_suspended` | bool | True | 是否过滤停牌股票（0=停牌，非0=正常交易） |
| `filter_limit` | bool | True | 是否过滤涨跌停股票（1=涨停，-1=跌停，0=正常） |

### 使用示例

#### 1. 默认配置（推荐）

**使用 FactorAnalyzer**:
```python
from src.analyze import FactorAnalyzer

# 默认：过滤停牌和涨跌停股票
analyzer = FactorAnalyzer()

# 等价于
analyzer = FactorAnalyzer(filter_suspended=True, filter_limit=True)
```

**使用 factor_group_analysis()**:
```python
from src.analyze import factor_group_analysis

# 默认：过滤停牌和涨跌停股票
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    group_count=10
)

# 等价于显式指定
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    group_count=10,
    filter_suspended=True,
    filter_limit=True
)
```

**适用场景**：大多数因子研究
- 停牌股票无法交易，未来收益为0，会干扰因子效果
- 涨跌停股票流动性受限，收益可能失真

#### 2. 不过滤任何股票

```python
# 保留所有股票（包括停牌和涨跌停）
analyzer = FactorAnalyzer(
    filter_suspended=False,
    filter_limit=False
)

pred_label_df, pivot_df = analyzer.analyze(
    factor_data,
    factor_col='scc',
    group_count=10
)
```

**适用场景**：
- 需要完整市场样本的研究
- 研究停牌/涨跌停对因子的影响
- 分析特殊事件下的因子表现

#### 3. 只过滤涨跌停

```python
# 保留停牌股票，但过滤涨跌停
analyzer = FactorAnalyzer(
    filter_suspended=False,  # 不过滤停牌
    filter_limit=True         # 过滤涨跌停
)
```

**适用场景**：
- 研究停牌风险暴露
- 分析长期停牌对因子的影响
- 评估因子对停牌事件的预测能力

#### 4. 只过滤停牌

```python
# 过滤停牌股票，但保留涨跌停
analyzer = FactorAnalyzer(
    filter_suspended=True,  # 过滤停牌
    filter_limit=False       # 不过滤涨跌停
)
```

**适用场景**：
- 研究涨跌停机制下的因子表现
- 分析流动性限制的影响
- 评估因子在极端行情下的表现

### 批量对比不同过滤配置

```python
import pandas as pd
from src.analyze import FactorAnalyzer

# 测试不同过滤配置的影响
configs = [
    ('过滤停牌+涨跌停', True, True),
    ('不过滤', False, False),
    ('只过滤涨跌停', False, True),
    ('只过滤停牌', True, False),
]

results = {}
for name, filter_suspended, filter_limit in configs:
    analyzer = FactorAnalyzer(
        filter_suspended=filter_suspended,
        filter_limit=filter_limit
    )

    pred_label_df, pivot_df = analyzer.analyze(
        factor_data,
        factor_col='scc',
        group_count=10
    )

    # 计算IC
    ic = pivot_df.corrwith(pd.Series(range(1, 11)), axis=1).mean()
    results[name] = {
        'IC均值': ic,
        '样本数': len(pred_label_df)
    }

# 对比结果
comparison = pd.DataFrame(results).T
print("不同配置下的IC对比：")
print(comparison)
```

**输出示例**：
```
不同配置下的IC对比：
                    IC均值  样本数
过滤停牌+涨跌停    0.0830  24500
不过滤            0.0721  28000
只过滤涨跌停      0.0758  26800
只过滤停牌        0.0801  25700
```

### 注意事项

1. **数据质量**：
   - 过滤停牌和涨跌停股票通常能获得更准确的因子效果
   - 停牌股票的未来收益为0，会降低IC值
   - 涨跌停股票的收益可能失真

2. **样本量**：
   - 过滤会减少样本量，需要权衡准确性和统计显著性
   - 建议在对比不同配置时，同时关注IC值和样本数

3. **一致性**：
   - 同一研究中应保持一致的过滤策略
   - 不同策略的IC值不可直接比较

4. **特殊研究**：
   - 如果研究目标是分析停牌/涨跌停的影响，则不过滤
   - 如果研究目标是因子预测能力，建议过滤

---

## 使用场景

### 场景1：快速验证因子效果

**需求**：刚计算完因子，想快速看看分组效果

**推荐**：`factor_group_analysis()` 函数

```python
from src.generator import NetworkCentralityFactor
from src.analyze import factor_group_analysis

# 1. 计算因子
factor_engine = NetworkCentralityFactor(
    codes="csi300",
    start_date="2020-01-01",
    end_date="2023-12-31"
)
factor_engine.fetch_data()
scc_factor = factor_engine.get_scc_factor(window=20)

# 2. 转换为 MultiIndex 格式
factor_data = scc_factor.stack().to_frame('scc')

# 3. 快速分析
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    group_count=10
)

# 4. 查看单调性
print("各组平均收益：")
print(pivot_df.mean())

# 5. 可视化
pivot_df.mean().plot(kind='bar', title='SCC因子分组收益')
```

### 场景2：批量分析多个因子

**需求**：有SCC、TCC、CC三个因子，需要批量分析

**推荐**：`FactorAnalyzer` 类

```python
from src.analyze import FactorAnalyzer

# 1. 准备因子数据
factor_data = ...  # 包含 'scc', 'tcc', 'cc' 三列

# 2. 创建分析器（只初始化一次）
analyzer = FactorAnalyzer()

# 3. 批量分析
results = {}
for factor_col in ['scc', 'tcc', 'cc']:
    pred_label_df, pivot_df = analyzer.analyze(
        factor_data[[factor_col]],  # 只取需要的列
        factor_col=factor_col,
        group_count=10
    )
    results[factor_col] = {
        'pred_label_df': pred_label_df,
        'pivot_df': pivot_df
    }

# 4. 比较结果
import pandas as pd
comparison = pd.DataFrame({
    factor: results[factor]['pivot_df'].mean()
    for factor in results
})
print("各因子分组收益对比：")
print(comparison)
```

**优势**：
- 只创建一个 `FactorAnalyzer` 实例
- 如果需要可以注入 `data_provider` 重用连接

### 场景3：分析不同窗口的因子

**需求**：比较20天、60天、120天窗口的因子效果

**推荐**：`FactorAnalyzer` 类 + 循环

```python
from src.generator import NetworkCentralityFactor
from src.analyze import FactorAnalyzer

# 1. 准备因子
factor_engine = NetworkCentralityFactor(
    codes="csi300",
    start_date="2020-01-01",
    end_date="2023-12-31"
)
factor_engine.fetch_data()

# 2. 创建分析器
analyzer = FactorAnalyzer()

# 3. 计算不同窗口的因子
windows = [20, 60, 120]
results = {}

for window in windows:
    print(f"计算 {window} 天窗口...")
    scc_df = factor_engine.get_scc_factor(window=window)
    factor_data = scc_df.stack().to_frame('scc')

    # 分析
    pred_label_df, pivot_df = analyzer.analyze(
        factor_data,
        factor_col='scc',
        group_count=10
    )

    results[window] = pivot_df

# 4. 比较
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, window in enumerate(windows):
    results[window].mean().plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'{window}天窗口')
    axes[i].set_xlabel('分组')
    axes[i].set_ylabel('平均收益')

plt.tight_layout()
plt.savefig('docs/images/window_comparison.png')
```

### 场景4：自定义未来收益周期

**需求**：想测试5日收益、10日收益等不同周期

**推荐**：两者都可以，使用 `forward_expr` 参数

```python
from src.analyze import factor_group_analysis

# 测试不同未来收益周期
forward_periods = [2, 5, 10, 20]
results = {}

for period in forward_periods:
    # 自定义未来收益表达式
    forward_expr = f"Ref($close,-{period})/Ref($close,-1)-1"

    pred_label_df, pivot_df = factor_group_analysis(
        factor_data,
        factor_col='scc',
        forward_expr=forward_expr,  # 自定义表达式
        group_count=10
    )

    results[period] = pivot_df.mean()
    print(f"{period}日收益 - IC: {pivot_df.corr().iloc[0, 1]:.4f}")

# 比较不同周期
import pandas as pd
comparison = pd.DataFrame(results)
print(comparison)
```

### 场景5：使用等宽分组

**需求**：因子值分布不均匀，想用固定宽度分组而非等频

**推荐**：使用 `bin_width` 参数

```python
from src.analyze import factor_group_analysis

# 查看因子分布
factor_values = factor_data['scc'].values
print(f"因子范围: [{factor_values.min():.4f}, {factor_values.max():.4f}]")
print(f"因子标准差: {factor_values.std():.4f}")

# 使用等宽分组（宽度 = 标准差）
bin_width = factor_values.std() / 2

pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    bin_width=bin_width,  # 等宽分组
    group_count=None      # 忽略等频分组
)

print(f"分组数量: {len(pivot_df.columns)}")
print("各分组样本数：")
print(pred_label_df.groupby('factor_quantile').size())
```

---

## 完整示例

### 示例1：完整的因子分析流程

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.generator import NetworkCentralityFactor
from src.analyze import factor_group_analysis

# ========== 1. 计算因子 ==========
print("Step 1: 计算SCC因子...")
factor_engine = NetworkCentralityFactor(
    codes="csi300",
    start_date="2020-01-01",
    end_date="2023-12-31"
)
factor_engine.fetch_data()
scc_factor = factor_engine.get_scc_factor(window=20)

# ========== 2. 转换数据格式 ==========
print("Step 2: 转换为MultiIndex格式...")
factor_data = scc_factor.stack().to_frame('scc')
print(f"因子数据形状: {factor_data.shape}")

# ========== 3. 因子分组分析 ==========
print("Step 3: 因子分组分析...")
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    forward_expr="Ref($close,-2)/Ref($close,-1)-1",  # 2日收益
    group_count=10
)

print(f"分组数据形状: {pred_label_df.shape}")
print(f"透视表形状: {pivot_df.shape}")

# ========== 4. 结果分析 ==========
print("\n========== 分析结果 ==========")

# 4.1 单调性检验
print("\n1. 各组平均收益：")
group_returns = pivot_df.mean()
print(group_returns)

# 计算单调性（相关系数）
monotonicity = group_returns.corr(pd.Series(range(len(group_returns))))
print(f"\n单调性相关系数: {monotonicity:.4f}")
print("（越接近1越好，>0.9为优秀）")

# 4.2 多空收益
long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]
print(f"\n多空收益（第10组 - 第1组）: {long_short_return:.4f}")

# 4.3 胜率
win_rate = (pivot_df.iloc[-1] > pivot_df.iloc[0]).sum() / len(pivot_df)
print(f"胜率: {win_rate:.2%}")

# 4.4 IC和IC_IR
ic_values = pivot_df.corrwith(pd.Series(range(1, 11)), axis=1)
ic_mean = ic_values.mean()
ic_std = ic_values.std()
ic_ir = ic_mean / ic_std if ic_std > 0 else 0

print(f"\nIC均值: {ic_mean:.4f}")
print(f"IC标准差: {ic_std:.4f}")
print(f"IC_IR: {ic_ir:.4f}")

# ========== 5. 可视化 ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 5.1 分组收益（均值）
group_returns.plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('各组平均收益')
axes[0, 0].set_xlabel('分组')
axes[0, 0].set_ylabel('平均收益')
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 5.2 分组收益（时间序列）
pivot_df.iloc[:, [0, 4, 9]].plot(ax=axes[0, 1])
axes[0, 1].set_title('分组收益时间序列（第1、5、10组）')
axes[0, 1].set_xlabel('日期')
axes[0, 1].set_ylabel('收益')
axes[0, 1].legend(['第1组', '第5组', '第10组'])

# 5.3 IC时间序列
ic_values.plot(ax=axes[1, 0])
axes[1, 0].set_title('IC时间序列')
axes[1, 0].set_xlabel('日期')
axes[1, 0].set_ylabel('IC')
axes[1, 0].axhline(y=ic_mean, color='r', linestyle='--', label=f'均值={ic_mean:.4f}')
axes[1, 0].legend()

# 5.4 累积收益
cumulative_returns = (1 + pivot_df).cumprod()
cumulative_returns.iloc[:, [0, 4, 9]].plot(ax=axes[1, 1])
axes[1, 1].set_title('累积收益（第1、5、10组）')
axes[1, 1].set_xlabel('日期')
axes[1, 1].set_ylabel('累积收益')
axes[1, 1].legend(['第1组', '第5组', '第10组'])

plt.tight_layout()
plt.savefig('docs/images/scc_factor_analysis.png', dpi=150)
print("\n图表已保存: docs/images/scc_factor_analysis.png")

# ========== 6. 保存结果 ==========
print("\nStep 5: 保存结果...")
pred_label_df.to_csv('data/scc_pred_label.csv')
pivot_df.to_csv('data/scc_group_returns.csv')
print("结果已保存到 data/ 目录")

factor_engine.cleanup_memory()
print("\n分析完成！")
```

### 示例2：使用 FactorAnalyzer 进行高级分析

```python
import pandas as pd
from src.analyze import FactorAnalyzer
from src.data_provider import QlibDataProvider

# ========== 1. 准备数据 ==========
# 假设已经有因子数据
factor_data = ...  # MultiIndex [date, instrument]，包含 'scc', 'tcc', 'cc'

# ========== 2. 创建分析器（可注入data_provider）==========
provider = QlibDataProvider("csi300", "2020-01-01", "2023-12-31")
analyzer = FactorAnalyzer(data_provider=provider)

# ========== 3. 批量分析多个因子 ==========
factors = ['scc', 'tcc', 'cc']
results = {}

for factor_col in factors:
    print(f"\n分析 {factor_col} 因子...")

    pred_label_df, pivot_df = analyzer.analyze(
        factor_data[[factor_col]],
        factor_col=factor_col,
        group_count=10
    )

    results[factor_col] = {
        'pred_label_df': pred_label_df,
        'pivot_df': pivot_df,
        'IC均值': pivot_df.corrwith(pd.Series(range(1, 11)), axis=1).mean(),
        '多空收益': pivot_df.iloc[-1].mean() - pivot_df.iloc[0].mean()
    }

# ========== 4. 汇总对比 ==========
summary = pd.DataFrame({
    factor: {
        'IC均值': results[factor]['IC均值'],
        '多空收益': results[factor]['多空收益']
    }
    for factor in factors
}).T

print("\n========== 因子对比 ==========")
print(summary)

# ========== 5. 自定义分析 ==========
# 例如：分析某一特定时间段
print("\n========== 2020年表现 ==========")
for factor in factors:
    pivot_df = results[factor]['pivot_df']
    pivot_2020 = pivot_df.loc['2020']

    print(f"\n{factor.upper()} 因子 2020年：")
    print(f"  IC均值: {pivot_2020.corrwith(pd.Series(range(1, 11)), axis=1).mean():.4f}")
    print(f"  多空收益: {pivot_2020.iloc[-1].mean() - pivot_2020.iloc[0].mean():.4f}")
```

---

## 最佳实践

### 1. 数据格式准备

**原则**：确保因子数据是正确的 MultiIndex 格式

```python
# ❌ 错误：普通DataFrame
#          股票A    股票B    股票C
# 2020-01-21  0.52    0.48    0.51
# 2020-01-22  0.53    0.49    0.50

# ✅ 正确：MultiIndex [date, instrument]
# date        instrument
# 2020-01-21  股票A        0.52
#             股票B        0.48
#             股票C        0.51
# 2020-01-22  股票A        0.53
# ...

# 转换方法
scc_factor = ...  # DataFrame, index=日期, columns=股票
factor_data = scc_factor.stack().to_frame('scc')  # 转换为MultiIndex
```

### 2. 选择合适的接口

**决策树**：

```
需要分析因子？
├─ 只分析一次，快速验证？
│  └─ ✅ 使用 factor_group_analysis() 函数
│
├─ 需要批量分析多个因子？
│  └─ ✅ 使用 FactorAnalyzer 类
│
└─ 需要自定义流程（如访问中间步骤）？
   └─ ✅ 使用 FactorAnalyzer 类，调用单个方法
```

### 3. 参数选择

**分组数量（group_count）**：
- 5组：快速测试，每组样本多
- 10组：**推荐**，平衡精度和样本量
- 20组：精细分析，但每组样本少

**未来收益周期（forward_expr）**：
```python
# 常用表达式
"Ref($close,-2)/Ref($close,-1)-1"   # 2日收益（默认）
"Ref($close,-5)/Ref($close,-1)-1"   # 5日收益
"Ref($close,-10)/Ref($close,-1)-1"  # 10日收益
"Ref($close,-20)/Ref($close,-1)-1"  # 20日收益（约1个月）
```

**分组方式（等频 vs 等宽）**：
- 等频（group_count）：**推荐**，每组样本数相同
- 等宽（bin_width）：因子值分布均匀时使用

### 4. 性能优化

**批量分析时重用 analyzer**：
```python
# ✅ 推荐：重用实例
analyzer = FactorAnalyzer()
for factor in factors:
    analyzer.analyze(factor_data[[factor]], factor_col=factor)

# ❌ 不推荐：每次创建新实例
for factor in factors:
    analyzer = FactorAnalyzer()  # 重复初始化
    analyzer.analyze(...)
```

**使用 data_provider 重用连接**（大规模数据）：
```python
provider = QlibDataProvider("ashares", "2020-01-01", "2023-12-31")
analyzer = FactorAnalyzer(data_provider=provider)
# 批量分析...
```

### 5. 内存管理

```python
# 分析完成后清理内存
factor_engine.cleanup_memory()

# 如果使用 FactorAnalyzer
del analyzer
import gc
gc.collect()
```

### 6. 结果验证

**检查清单**：
- [ ] 各组样本数是否均衡（等频分组应相近）
- [ ] 分组收益是否单调（第1组 < 第10组）
- [ ] IC均值是否显著（绝对值 > 0.03）
- [ ] IC_IR是否合理（> 0.5）
- [ ] 多空收益是否为正

```python
# 快速验证脚本
def validate_factor_analysis(pred_label_df, pivot_df):
    """验证因子分析结果"""

    # 1. 样本均衡性
    group_sizes = pred_label_df.groupby('factor_quantile').size()
    print("1. 样本均衡性：")
    print(group_sizes)
    print(f"   变异系数: {group_sizes.std() / group_sizes.mean():.2%} (<10%为佳)")

    # 2. 单调性
    group_returns = pivot_df.mean()
    monotonicity = group_returns.corr(pd.Series(range(len(group_returns))))
    print(f"\n2. 单调性: {monotonicity:.4f} (>0.9为优秀)")

    # 3. IC统计
    ic_values = pivot_df.corrwith(pd.Series(range(1, 11)), axis=1)
    print(f"\n3. IC均值: {ic_values.mean():.4f}")
    print(f"   IC_IR: {ic_values.mean() / ic_values.std():.4f}")

    # 4. 多空收益
    long_short = group_returns.iloc[-1] - group_returns.iloc[0]
    print(f"\n4. 多空收益: {long_short:.4f}")

    return {
        'monotonicity': monotonicity,
        'IC_mean': ic_values.mean(),
        'IC_IR': ic_values.mean() / ic_values.std(),
        'long_short': long_short
    }

# 使用
metrics = validate_factor_analysis(pred_label_df, pivot_df)
```

---

## 常见问题

### Q1: factor_group_analysis() 和 FactorAnalyzer 有什么区别？

**A**:
- **factor_group_analysis()**: 便捷函数，适合快速分析、一次性计算
- **FactorAnalyzer**: 完整类，适合批量分析、需要更多控制的场景

**核心区别**：函数是类的简单包装器，内部调用类的方法。

详见：[接口对比](#接口对比)

### Q2: 我应该使用哪个接口？

**A**: 使用决策树：

```
快速验证一个因子？
→ factor_group_analysis()

批量分析多个因子/窗口？
→ FactorAnalyzer

需要自定义流程/访问中间步骤？
→ FactorAnalyzer
```

### Q3: 因子数据格式要求是什么？

**A**: 必须是 `MultiIndex [date, instrument]`：

```python
# 正确格式
factor_data.index.names = ['date', 'instrument']
factor_data.columns = ['scc']  # 因子列名

# 转换示例
scc_df = ...  # DataFrame, index=日期, columns=股票
factor_data = scc_df.stack().to_frame('scc')
```

### Q4: group_count 和 bin_width 有什么区别？

**A**:
- **group_count**: 等频分组，每组样本数相同（推荐）
- **bin_width**: 等宽分组，每组因子值范围相同

```python
# 等频分组（10组）
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    group_count=10  # 每组约10%的样本
)

# 等宽分组（宽度=0.1）
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    bin_width=0.1  # 每组因子值范围为0.1
)
```

### Q5: 如何自定义未来收益周期？

**A**: 使用 `forward_expr` 参数：

```python
# 5日收益
forward_expr = "Ref($close,-5)/Ref($close,-1)-1"

pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    forward_expr=forward_expr
)
```

### Q6: 如何解读分析结果？

**A**: 关注以下指标：

1. **单调性**: 相关系数，>0.9为优秀
2. **IC均值**: 因子预测能力，绝对值>0.03为显著
3. **IC_IR**: IC稳定性，>0.5为合理
4. **多空收益**: 最优组-最差组收益，应显著>0

```python
# 快速解读
group_returns = pivot_df.mean()
monotonicity = group_returns.corr(pd.Series(range(len(group_returns))))
print(f"单调性: {monotonicity:.4f} (>0.9优秀)")
print(f"多空收益: {group_returns.iloc[-1] - group_returns.iloc[0]:.4f} (>0佳)")
```

### Q7: 遇到 "factor_col 不存在" 错误怎么办？

**A**: 检查列名：

```python
# 查看可用列
print(factor_data.columns)

# 确保列名正确
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc'  # 必须在 factor_data.columns 中
)
```

### Q8: 如何提高分析性能？

**A**:
1. 批量分析时重用 `FactorAnalyzer` 实例
2. 大规模数据时注入 `data_provider`
3. 合理选择分组数量（10组足够）
4. 分析完成后清理内存

### Q9: 为什么要过滤停牌和涨跌停股票？

**A**: 过滤这两类股票的原因：

1. **停牌股票**：
   - 无法交易，未来收益为0
   - 会降低因子IC值
   - 干扰因子真实预测能力

2. **涨跌停股票**：
   - 流动性受限
   - 收益可能失真
   - 无法完全反映市场预期

**建议**：大多数因子研究应过滤这两类股票（默认配置）

```python
# 推荐配置（默认）
analyzer = FactorAnalyzer(filter_suspended=True, filter_limit=True)
```

### Q10: 什么情况下不过滤停牌/涨跌停股票？

**A**: 特殊研究场景下可能需要保留：

1. **研究停牌影响**：
   ```python
   analyzer = FactorAnalyzer(filter_suspended=False, filter_limit=True)
   ```
   - 研究停牌风险暴露
   - 评估因子对停牌事件的预测能力

2. **研究涨跌停机制**：
   ```python
   analyzer = FactorAnalyzer(filter_suspended=True, filter_limit=False)
   ```
   - 分析涨跌停下的因子表现
   - 研究流动性限制的影响

3. **完整市场分析**：
   ```python
   analyzer = FactorAnalyzer(filter_suspended=False, filter_limit=False)
   ```
   - 需要完整市场样本
   - 对比不同过滤策略的影响

**注意**：不同过滤策略的IC值不可直接比较，需同时关注样本数变化。

### Q11: factor_group_analysis() 能自定义过滤配置吗？

**A**: 能！`factor_group_analysis()` 从 v1.1.0 开始支持自定义过滤配置。

```python
from src.analyze import factor_group_analysis

# 默认配置（过滤停牌和涨跌停）
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    group_count=10
)

# 自定义过滤配置（不过滤停牌和涨跌停）
pred_label_df, pivot_df = factor_group_analysis(
    factor_data,
    factor_col='scc',
    group_count=10,
    filter_suspended=False,  # 不过滤停牌
    filter_limit=False       # 不过滤涨跌停
)
```

**选择建议**：
- 快速分析 + 自定义过滤 → `factor_group_analysis()` ✅
- 批量分析 + 复杂流程 → `FactorAnalyzer` ✅

```python
# 高性能批量分析
analyzer = FactorAnalyzer()
for factor in factors:
    analyzer.analyze(factor_data[[factor]], factor_col=factor)
```

---

## API参考

### 函数接口

#### factor_group_analysis()

```python
factor_group_analysis(
    factor_data: pd.DataFrame,
    *,
    factor_col: str = "factor",
    forward_expr: str = "Ref($close,-2)/Ref($close,-1)-1",
    group_count: int = 10,
    bin_width: Optional[int] = None,
    filter_suspended: bool = True,
    filter_limit: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

**参数**：
- `factor_data`: 因子数据，MultiIndex [date, instrument]
- `factor_col`: 因子列名
- `forward_expr`: 未来收益Qlib表达式
- `group_count`: 分组数量（等频）
- `bin_width`: 等宽分组宽度（与group_count二选一）
- `filter_suspended`: 是否过滤停牌股票（默认True）⭐ v1.1.0新增
- `filter_limit`: 是否过滤涨跌停股票（默认True）⭐ v1.1.0新增

**返回**：
- `pred_label_df`: MultiIndex [date, instrument]，包含 factor, label, factor_quantile
- `pivot_df`: index=日期, columns=分组, values=平均收益

### 类接口

#### FactorAnalyzer

```python
class FactorAnalyzer:
    def __init__(
        self,
        data_provider: Optional[QlibDataProvider] = None,
        filter_suspended: bool = True,
        filter_limit: bool = True
    )

    def analyze(
        self,
        factor_data: pd.DataFrame,
        factor_col: str = "factor",
        forward_expr: str = "Ref($close,-2)/Ref($close,-1)-1",
        group_count: int = 10,
        bin_width: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]

    # 其他方法（高级用法）
    def _infer_parameters(factor_data)
    def fetch_forward_returns(instruments, start_date, end_date, forward_expr)
    def filter_tradable(data, forward_col)
    def merge_data(factor_data, factor_col, forward_returns, forward_col)
    def apply_grouping(data, group_count, bin_width)
```

**初始化参数**：
- `data_provider`: 数据提供者（可选）
- `filter_suspended`: 是否过滤停牌股票（默认True）
- `filter_limit`: 是否过滤涨跌停股票（默认True）

**方法说明**：
- `analyze()`: 完整分析流程（推荐使用）
- `_infer_parameters()`: 从数据推断参数（私有）
- `fetch_forward_returns()`: 获取未来收益（高级用法）
- `filter_tradable()`: 过滤可交易数据（高级用法，支持灵活配置）
- `merge_data()`: 合并因子和收益（高级用法）
- `apply_grouping()`: 应用因子分组（高级用法）

---

## 参考资料

1. **项目文档**：
   - [README.md](../README.md) - 项目说明
   - [CLAUDE.md](../CLAUDE.md) - 技术文档
   - [使用指南_网络中心度因子](使用指南_网络中心度因子.md) - 因子计算使用指南

2. **核心代码**：
   - [src/analyze.py](../src/analyze.py) - 本模块源码
   - [src/factor_algo.py](../src/factor_algo.py) - 核心算法
   - [src/generator.py](../src/generator.py) - 因子生成器

3. **外部依赖**：
   - [alphalens文档](https://quantopian.github.io/alphalens/) - 因子分析框架
   - [Qlib文档](https://qlib.readthedocs.io/) - 量化投资平台

---

**文档版本**: 1.0.0
**最后更新**: 2025-12-24
**维护者**: Hugo

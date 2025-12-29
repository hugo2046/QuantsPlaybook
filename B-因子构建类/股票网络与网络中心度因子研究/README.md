# 股票网络与网络中心度因子研究

## 项目简介

本项目是**华西证券金融工程专题研究**的复现实现，基于复杂网络理论，通过分析股票间的相关性构建股票网络，并计算网络中心度因子用于选股策略。

**核心研究论文**：华西证券金融工程专题报告（2021年3月）《股票网络与网络中心度因子研究》

### 核心因子

- **SCC（Spatial Centrality Centrality）**：空间网络中心度因子
  - 基于股票间Pearson相关系数的平均距离
  - SCC值越大，股票在网络中越中心

- **TCC（Temporal Centrality Centrality）**：时间网络中心度因子
  - 基于收益率偏离的时间稳定性
  - TCC值越大，股票收益率越稳定

- **CC（Composite Centrality）**：综合网络中心度因子
  - SCC与TCC的1:1合成
  - 综合考虑空间和时间维度的网络中心度

## 项目状态

✅ **核心算法已完成，进入测试验证阶段**

### 已完成功能

- ✅ SCC因子计算（`src/factor_algo.py:calculate_scc`）
- ✅ TCC因子计算（`src/factor_algo.py:calculate_tcc`）
- ✅ CC因子合成（`src/generator.py:get_cc_factor`）
- ✅ 滑动窗口因子生成器（`src/factor_algo.py:generate_factor`）
- ✅ NetworkCentralityFactor类（`src/generator.py`）
- ✅ 因子分析模块（`src/analyze.py`）
  - factor_group_analysis() 便捷函数
  - FactorAnalyzer 分析器类
- ✅ 完整的中文文档（Sphinx规范）
- ✅ 性能优化和错误处理

### 待实现功能

- ⚠️ 单元测试（tests/）
- ⚠️ 回测验证

## 快速开始

### 环境配置

```bash
# 激活Python环境
conda activate [your-environment]

# 安装依赖
pip install pandas numpy qlib alphalens loguru tqdm psutil pytest

# 设置PYTHONPATH
export PYTHONPATH="/data1/hugo/workspace:/data1/hugo/workspace/qlib_ddb:$PYTHONPATH"
```

### 基础使用

```python
import numpy as np
import pandas as pd
from src.factor_algo import calculate_scc, calculate_tcc, generate_factor

# 1. 准备收益率数据
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
stocks = [f'stock_{i:03d}' for i in range(50)]
returns = pd.DataFrame(
    np.random.randn(100, 50) * 0.02,
    index=dates,
    columns=stocks
)

# 2. 计算SCC因子时间序列（20天滑动窗口）
scc_factor = generate_factor(returns, calculate_scc, window=20)

# 3. 计算TCC因子时间序列
tcc_factor = generate_factor(returns, calculate_tcc, window=20)

print(f"SCC因子形状: {scc_factor.shape}")  # (81, 50)
print(f"TCC因子形状: {tcc_factor.shape}")  # (81, 50)
```

### 实际数据示例

```python
from src.data_provider import QlibDataProvider
from src.factor_algo import generate_factor, calculate_scc

# 1. 获取沪深300收益率数据
provider = QlibDataProvider(
    codes="csi300",
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# 2. 获取收盘价并计算收益率
close_prices = provider.get_features(
    instruments=provider.instruments_,
    start_time="2020-01-01",
    end_time="2023-12-31",
    fields=["$close"]
)

# 3. 计算收益率
returns = close_prices.unstack().pct_change().dropna()

# 4. 生成SCC因子（20天窗口）
scc_factor = generate_factor(returns, calculate_scc, window=20)

print(f"因子时间范围: {scc_factor.index[0]} 至 {scc_factor.index[-1]}")
print(scc_factor.head())
```

> 💡 **提示**：想要更详细的使用说明？请查看 **[使用指南](docs/使用指南_网络中心度因子.md)**，包含：
> - 完整的 `NetworkCentralityFactor` 类使用示例
> - 更多实际场景的工作流
> - 高级用法和最佳实践
> - 常见问题解答

## 核心算法

### SCC因子（空间网络中心度）

**原理**：基于股票间Pearson相关系数的平均距离

**简化公式**：
```
SCC = 1 / [2 * (1 - ρ̄)]
```

其中 `ρ̄` 是该股票与其他所有股票的平均相关系数。

**特点**：
- 向量化计算，高效处理相关系数矩阵
- 使用简化公式，避免不必要的平方和开方运算
- 完整的输入验证

### TCC因子（时间网络中心度）

**原理**：基于收益率偏离的时间稳定性

**简化公式**：
```
TCC = 1 / E[z²]
```

其中 `z` 是收益率相对市场平均的标准化偏离度，`E[z²]` 是时间维度上的平均平方偏离。

**特点**：
- 正确使用 axis=0 进行时间维度平均
- 性能优化：减少重复的 nan_to_num 调用
- 广播机制高效计算

### 因子生成器

**功能**：使用滑动窗口生成因子时间序列

**特性**：
- 📊 tqdm 进度条显示
- 🛡️ 完善的错误处理机制
- 📝 loguru 结构化日志（精简输出）
- ⏱️ 性能统计（耗时、成功率）

## 论文回测结果（供验证）

| 因子 | IC均值 | IC_IR | 多空组合年化收益 | 最大回撤 |
|------|--------|-------|------------------|----------|
| SCC | 8.30% | 3.97 | 26.80% | -1.79% |
| TCC | 9.05% | 3.55 | 22.10% | -15.02% |
| CC  | 9.21% | 4.19 | 24.86% | -7.95% |

## 项目结构

```
股票网络与网络中心度因子研究/
├── src/
│   ├── __init__.py           # 包初始化
│   ├── data_provider.py      # Qlib数据提供者
│   ├── factor_algo.py        # ⭐ 核心算法模块
│   └── analyze.py            # 因子分析（待重构）
├── docs/
│   └── 20210316-华西证券-金融工程专题报告：股票网络与网络中心度因子研究.md
├── dev/
│   └── dev.ipynb             # 开发环境
├── tests/                    # 测试目录（待添加）
├── CLAUDE.md                 # 📖 项目技术文档
└── README.md                 # 本文件
```

## 测试验证

```bash
# 测试核心模块导入
python -c "from src.factor_algo import calculate_scc, calculate_tcc, generate_factor; print('导入成功')"

# 测试SCC因子计算
python -c "
import numpy as np
from src.factor_algo import calculate_scc
np.random.seed(42)
returns = np.random.randn(20, 100) * 0.02
scc = calculate_scc(returns)
print(f'SCC因子值: {scc.shape}, 范围: [{scc.min():.4f}, {scc.max():.4f}]')
"

# 测试TCC因子计算
python -c "
import numpy as np
from src.factor_algo import calculate_tcc
np.random.seed(42)
returns = np.random.randn(20, 100) * 0.02
tcc = calculate_tcc(returns)
print(f'TCC因子值: {tcc.shape}, 范围: [{tcc.min():.4f}, {tcc.max():.4f}]')
"
```

## 性能优化

### 已实现的优化

1. **SCC因子**：
   - 使用简化公式 `1 / [2(1-ρ̄)]` 避免 `d = √(2(1-ρ̄))` 再 `1/d²`
   - 向量化计算相关系数矩阵，避免循环
   - 减少数组拷贝操作

2. **TCC因子**：
   - 减少重复的 `np.nan_to_num()` 调用（从3次降至1次）
   - 先平方再平均，避免重复计算平方根
   - 使用广播机制高效计算标准化偏离度

3. **因子生成器**：
   - 精简日志输出，仅在关键时刻记录
   - 使用 tqdm 显示进度，替代频繁的日志输出
   - 错误处理机制，避免单个窗口失败影响整体

## 开发路线图

### ✅ 已完成（2025-12-24）

- SCC、TCC、CC因子完整实现
- NetworkCentralityFactor 类
- 滑动窗口因子生成器
- 因子分析模块（analyze.py）
  - factor_group_analysis() 便捷函数
  - FactorAnalyzer 分析器类
  - 完整重构，消除代码重复
- 完整的中文文档和使用示例
  - 使用指南_网络中心度因子.md
  - analyze使用指南.md
- 性能优化和错误处理

### 🔄 当前优先级

1. **单元测试**（test_factor_algo.py, test_generator.py, test_analyze.py）
2. **回测验证**（对比论文结果）

### 📋 待办事项

3. 性能优化（大规模数据）
4. 开发环境文档（dev.ipynb）

## 技术文档

详细的技术文档请参考：

### 📖 使用文档
- **[使用指南_网络中心度因子](docs/使用指南_网络中心度因子.md)** ⭐ - **强烈推荐先阅读**
  - 5分钟快速上手
  - 核心算法详解（SCC、TCC、CC）
  - 完整工作流示例
  - 高级用法和最佳实践
  - 常见问题FAQ
  - 完整API参考

- **[analyze使用指南](docs/analyze使用指南.md)** - 因子分析模块使用指南
  - factor_group_analysis() 函数 vs FactorAnalyzer 类
  - 使用场景和接口选择
  - 完整分析流程示例
  - 结果解读和验证
  - 最佳实践和性能优化
  - 常见问题FAQ

### 🔧 技术文档
- **[CLAUDE.md](CLAUDE.md)** - 完整的项目技术文档
  - 项目架构和开发路线
  - 核心算法实现细节
  - 数据流架构说明
  - 验证标准和测试指南

### 💻 源代码文档
- **[src/factor_algo.py](src/factor_algo.py)** - 核心算法模块
  - `calculate_scc()`：SCC因子计算
  - `calculate_tcc()`：TCC因子计算
  - `generate_factor()`：滑动窗口因子生成
  - 完整的Sphinx中文文档

- **[src/generator.py](src/generator.py)** - 因子生成器模块
  - `NetworkCentralityFactor`类
  - 统一的因子计算接口
  - 批量计算和内存管理

- **[src/analyze.py](src/analyze.py)** - 因子分析模块
  - `factor_group_analysis()`：便捷函数（推荐快速分析）
  - `FactorAnalyzer`：分析器类（推荐批量分析）
  - 因子分组、回测分析、绩效评估
  - 基于alphalens的高性能实现

### 📚 研究论文
- **[20210316-华西证券-金融工程专题报告：股票网络与网络中心度因子研究](docs/20210316-华西证券-金融工程专题报告：股票网络与网络中心度因子研究.md)**
  - 第1节：股票网络构建理论
  - 第2节：网络中心度因子定义和回测结果

## 注意事项

- 本项目基于华西证券2021年研究论文复现
- 核心算法已通过数学推导验证
- 待进行回测验证，对比论文中的IC和IC_IR值
- 所有投资决策应基于充分的风险评估和历史回测验证

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

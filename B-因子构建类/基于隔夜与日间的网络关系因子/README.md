# 基于隔夜与日间的网络关系因子

本项目实现了基于论文《A tug of war across the market: overnight-vs-daytime lead-lag networks and clustering-based portfolio strategies》的d-LE-SC算法，用于检测金融市场中的领先-滞后关系并构建基于聚类的投资组合策略。

## 🚀 快速开始

### 环境配置
```bash
# 1. 进入项目目录
cd QuantsPlaybook/B-因子构建类/基于隔夜与日间的网络关系因子

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置DolphinDB连接（如需使用真实数据）
export DOLPHINDB_URI="dolphindb://your_username:your_password@your_host:8848"
```

### 批量因子计算
```bash
# 运行主脚本，自动计算所有组合的因子
python loade_factor.py
```

### 自定义因子计算
```python
from factor_pipeline import FactorPipeline

pipeline = FactorPipeline(
    codes="ashares",
    start_dt="2020-01-01",
    end_dt="2025-10-27",
    window=60,
    network_type="preclose_lead_close",
    correlation_method="spearman"
)

final_factor_df = pipeline.run()
```

---

## 📊 项目状态

- **🎯 当前主入口**: `loade_factor.py` (批量因子计算和保存)
- **⭐ 核心流水线**: `factor_pipeline.py` (FactorPipeline类实现)
- **🚀 GPU加速**: `dlesc_clustering.py` (PyTorch + CUDA支持)
- **📋 开发参考**: `test_main.py` (学习和调试参考)
- **🔧 测试工具**: `test_random_seed_fix.py` (随机种子可复现性测试)

## 项目概述

该研究通过将日收益率分解为隔夜和日间成分，构建有向网络来捕捉股票间隔夜投机与日间价格修正之间的领先-滞后关系。我们开发了专门的d-LE-SC（directed Likelihood Estimation Spectral Clustering）算法来识别有向领先-滞后网络中的领导者股票组和滞后股票组。

## 核心特性

- **d-LE-SC算法实现**: 基于PyTorch的高效实现，支持GPU加速
- **多种网络构建**: 支持隔夜-领先-日间、日间-领先-隔夜、收盘-领先-收盘等网络类型
- **相关性方法选择**: 支持Pearson和Spearman两种相关性计算方法，适应不同数据特征
- **多数据源支持**: 支持模拟数据和qlib真实金融数据
- **组合策略构建**: 基于聚类结果构建多空投资组合
- **因子化改造**: 适合A股市场的因子计算模块，支持多种因子化方案
- **高效工具函数**: 包含内存优化的滑动窗口等实用工具
- **回测分析**: 完整的策略回测框架和性能评估
- **可视化分析**: 丰富的图表展示分析结果

## 项目结构

```
基于隔夜与日间的网络关系因子/
├── 核心代码文件
│   ├── loade_factor.py          # 🎯 主入口脚本（批量因子计算）
│   ├── factor_pipeline.py       # ⭐ FactorPipeline流水线实现
│   ├── dlesc_clustering.py      # 🚀 GPU加速d-LE-SC算法
│   ├── qlib_data_provider.py    # ⭐ qlib数据提供者
│   ├── lead_lag_network.py      # ⭐ 网络构建和相关性计算
│   ├── factor_computation.py    # ⭐ 因子计算器
│   ├── utils.py                 # 🔧 工具函数（滑动窗口等）
│   ├── DeltaLag.py              # 辅助工具
├── docs/                        # 文档目录
│   ├── ssrn-5371952.md          # 原始论文
│   └── 基于隔夜与日间的网络关系因子.md # 技术总结
├── examples/                    # 示例代码目录
│   ├── d_le_sc_v3.ipynb         # FactorPipeline使用示例
│   └── 因子分析.ipynb           # 因子分析示例
├── tests/                       # 测试目录
│   ├── __init__.py              # 测试模块初始化
│   └── test_main.py             # 基础功能测试
├── data/                        # 数据输出目录
├── requirements.txt             # 依赖包列表
├── README.md                   # 项目说明文档
└── CLAUDE.md                   # Claude Code指导文档
```

## 算法原理

### d-LE-SC算法 (4.2节)

d-LE-SC算法是一个基于最大似然估计的迭代谱聚类方法，专门用于检测领先-滞后结构：

1. **Hermitian矩阵构建**:
   ```
   H = i * log((1-η)/η) * (A - A^T) + log(1/(4η(1-η))) * (A + A^T)
   ```

2. **特征向量分解**: 计算H的顶部特征向量

3. **聚类**: 基于[Re(v1), Im(v1)]嵌入进行k-means聚类

4. **参数更新**: 迭代更新有向SBM参数η

### 网络构建 (3.2节)

支持三种类型的领先-滞后网络，每种网络都支持两种相关性计算方法：

1. **隔夜-领先-日间**: `Corr(overnight_returns_i, daytime_returns_j)`
2. **日间-领先-隔夜**: `Corr(daytime_returns_i[t-1], overnight_returns_j[t])`
3. **收盘-领先-收盘**: `Corr(close_returns_i[t-1], close_returns_j[t])`

**相关性方法选择**：
- **Pearson相关性**: 线性相关性，计算效率高，适用于数据分布接近正态的情况
- **Spearman相关性**: 单调相关性，基于秩次计算，对异常值稳健，适用于非线性关系

根据论文6.2节稳健性分析，不同相关性方法可能会产生不同的网络结构特征。系统提供了智能推荐功能，可基于数据特征自动选择合适的相关性方法。

### 投资组合构建 (4.1节)

三步组合构建流程：
1. 基于移动窗口方法构建相似性矩阵
2. 应用基于有向图的聚类算法识别领导者和滞后者
3. 从领导者组生成方向信号，在滞后者组内构建多空投资组合

## 安装与使用

### 环境要求

- Python 3.8+
- PyTorch 1.9.0+ (GPU版本推荐)
- pandas, numpy, matplotlib, seaborn
- qlib (可选，用于真实数据)
- 其他依赖见requirements.txt

### 安装步骤

1. 进入项目目录：
```bash
cd QuantsPlaybook/B-因子构建类/基于隔夜与日间的网络关系因子
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. (可选) 安装qlib以使用真实金融数据：
```bash
pip install pyqlib
```

注意：qlib需要配置相应的数据库连接（如DolphinDB）。

### 使用方法

#### 🎯 主入口：批量因子计算

**直接运行主脚本**：
```bash
python loade_factor.py
```

脚本将自动计算以下所有组合的因子：
- **网络类型**: daytime_lead_overnight, overnight_lead_daytime, preclose_lead_close
- **相关性方法**: pearson, spearman
- **时间范围**: 2020-01-01 到 2025-10-31
- **输出**: 每个组合生成两个parquet文件（做多因子和做空因子）

**输出文件**：
```
../data/{network_type}_{method}_long.parquet   # 做多因子
../data/{network_type}_{method}_short.parquet  # 做空因子
```

#### ⭐ 自定义因子计算

**使用FactorPipeline类**：
```python
import sys
from pathlib import Path

# 添加项目路径到PYTHONPATH（根据实际情况调整）
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.insert(0, str(project_root / "qlib_ddb"))

from factor_pipeline import FactorPipeline

pipeline = FactorPipeline(
    codes="ashares",
    start_dt="2020-01-01",
    end_dt="2025-10-27",
    window=60,
    network_type="preclose_lead_close",
    correlation_method="spearman",
    top_percentile=0.2,
    bottom_percentile=0.2,
    lead_percentile=0.5,
)

final_factor_df = pipeline.run()
```

#### 🔧 工具函数使用

**滑动窗口功能**：
```python
from pathlib import Path
import sys

# 添加项目路径到PYTHONPATH（根据实际情况调整）
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils import sliding_window
import numpy as np

# 示例数据
data = np.arange(10)
window_size = 3

# 生成滑动窗口
windows = list(sliding_window(data, window_size, step=1))
print(f"滑动窗口结果: {windows}")
# 输出: [array([0, 1, 2]), array([1, 2, 3]), array([2, 3, 4]), ...]

# 金融时间序列应用
returns = np.random.randn(100, 5)  # 100天，5只股票的收益率
for window in sliding_window(returns, window=20):  # 20天滚动窗口
    # 在每个窗口内进行相关性分析或其他计算
    correlation_matrix = np.corrcoef(window.T)
    # ... 后续分析
```

#### 📊 参数说明

**网络类型**：
- `daytime_lead_overnight`: 日间收益率 → 隔夜收益率
- `overnight_lead_daytime`: 隔夜收益率 → 日间收益率
- `preclose_lead_close`: 前收盘价 → 收盘价

**相关性方法**：
- `pearson`: 线性相关性，计算效率高
- `spearman`: 单调相关性，对异常值稳健

**因子配置**：
- `top_percentile`: 做多股票比例（默认0.2）
- `bottom_percentile`: 做空股票比例（默认0.2）
- `lead_percentile`: 领先股票筛选比例（默认0.5）

运行演示将执行完整的分析流程，包括：
- 网络构建
- d-LE-SC聚类
- 投资组合构建
- 回测分析
- 结果可视化

#### 命令行参数说明

**数据源参数**：
- `--data_source`: 数据源选择 (`synthetic` 或 `qlib`)
- `--n_stocks`: 模拟数据的股票数量
- `--n_days`: 模拟数据的交易日数量
- `--start_date`: qlib数据开始日期 (YYYY-MM-DD)
- `--end_date`: qlib数据结束日期 (YYYY-MM-DD)
- `--stock_pool`: qlib股票代码列表
- `--database_uri`: qlib数据库连接URI
- `--region`: 数据区域 (`REG_CN` 或 `REG_US`)
- `--market`: 股票市场 (`ashares`, `a-shares`等)

**其他参数**：
- `--mode`: 运行模式 (`demo` 或 `backtest`)
- `--output_dir`: 结果输出目录

#### 主要API使用

**使用模拟数据**：
```python
from dle_sc_algorithm import DLESCAlgorithm
from lead_lag_network import LeadLagNetworkBuilder, create_sample_returns_data
from portfolio_strategy import PortfolioConstructor

# 创建模拟数据
returns_data = create_sample_returns_data(n_stocks=50, n_days=100)

# 初始化组件
dlesc = DLESCAlgorithm(n_iterations=20, random_state=42)
network_builder = LeadLagNetworkBuilder(lookback_window=30)
portfolio_constructor = PortfolioConstructor()

# 构建网络
M, A = network_builder.build_network(returns_data, date, 'overnight_lead_daytime')

# 应用聚类
clustering_results = dlesc.fit(A.values)

# 构建投资组合
portfolio = portfolio_constructor.construct_portfolio(
    A, M, returns_data, date, clustering_results
)
```

**使用qlib数据（重构后的新方法）**：
```python
import sys
from pathlib import Path

# 添加项目路径到PYTHONPATH（根据实际情况调整）
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.insert(0, str(project_root / "qlib_ddb"))

from lead_lag_network import LeadLagNetworkBuilder
from qlib_data_provider import QlibDataProvider

# 初始化qlib数据提供者（需先设置环境变量）
# export DOLPHINDB_URI="dolphindb://username:password@host:port"
qlib_provider = QlibDataProvider("ashares", "2025-01-01", "2025-10-27")

# 生成网络构建器（使用qlib数据提供者）
network_builder = LeadLagNetworkBuilder(qlib_provider, 60)

# 构建网络
# M为有方向的邻接矩阵，A为无方向的邻接矩阵
# 参数"overnight_lead_daytime"表示构建隔夜领先日间的网络
# 输出形状为(n,m,m)，第0维为时间，第1-2维表示当日的矩阵
M, A = network_builder.build_network("overnight_lead_daytime", True)

# 后续可以进行聚类和投资组合构建...
```

### 运行因子计算（重构后的main.py）

```bash
python main.py
```

**输出示例**：
```
初始化因子计算器...
开始计算因子值...
因子值计算完成，因子矩阵形状：(100, 500)
因子值统计：
  非零因子数量：15234
  平均因子值：0.002456
  因子值标准差：0.015234
因子值已保存到：lead_lag_factor.csv
因子计算流程完成！
```

### 使用因子计算器模块

```python
from factor_computation import LeadLagFactorCalculator
from dle_sc_algorithm import DLESCClustering
from lead_lag_network import LeadLagNetworkBuilder
from qlib_data_provider import QlibDataProvider

# 初始化组件
calculator = LeadLagFactorCalculator(
    lead_percentile=0.5,
    top_percentile=0.4,
    bottom_percentile=0.2
)

# 获取数据
provider = QlibDataProvider("ashares", "2025-01-01", "2025-10-27")
network_builder = LeadLagNetworkBuilder(provider, 60)
M, A = network_builder.build_network("overnight_lead_daytime", True)

# DLE-SC聚类
model = DLESCClustering(n_iterations=20, random_state=42)
clustering_results = model.fit(M)

# 根据网络类型选择正确的领先部分收益率
network_type = "overnight_lead_daytime"  # 或 "daytime_lead_overnight"

if network_type == "overnight_lead_daytime":
    # 隔夜领先日间：领先部分是 overnight_returns
    returns_data = provider.overnight_return_df.iloc[60:].values
    print("使用隔夜收益率作为领先部分")
elif network_type == "daytime_lead_overnight":
    # 日间领先隔夜：领先部分是 daytime_returns.shift(1)
    returns_data = provider.daytime_return_df.shift(1).iloc[60:].values
    print("使用日间收益率(shift(1))作为领先部分")

stock_codes = list(provider.overnight_return_df.columns)
date_index = provider.overnight_return_df.index[60:]

# 计算因子
factor_df = calculator.compute_factor_values(
    adjacency_matrices=A,
    signed_matrices=M,
    clustering_results=clustering_results,
    returns_matrix=returns_data,
    stock_codes=stock_codes,
    date_index=date_index,
    network_type=network_type
)
```

## 核心类说明

### DLESCAlgorithm

d-LE-SC算法的主要实现类。

**主要方法**:
- `fit(A)`: 对邻接矩阵进行聚类
- `predict(A)`: 预测新数据的聚类标签

**参数**:
- `n_iterations`: 算法迭代次数
- `random_state`: 随机种子
- `device`: 计算设备（'cuda'或'cpu'）

### LeadLagNetworkBuilder

领先-滞后网络构建类，支持多种相关性计算方法。

**主要功能**:
- **网络构建**: 构建三种类型的领先-滞后网络
  - 隔夜-领先-日间: `Corr(overnight_returns_i, daytime_returns_j)`
  - 日间-领先-隔夜: `Corr(daytime_returns_i[t-1], overnight_returns_j[t])`
  - 收盘-领先-收盘: `Corr(close_returns_i[t-1], close_returns_j[t])`
- **相关性方法**: 支持Pearson和Spearman两种相关性计算
- **批量分析**: 支持多种网络类型和相关性方法的批量构建和比较

**主要方法**:
- `build_network()`: 构建完整的领先-滞后网络
- `build_multiple_networks()`: 批量构建多种网络组合
- `compare_correlation_methods()`: 比较不同相关性方法的结果
- `get_method_recommendation()`: 基于数据特征推荐合适的相关性方法
- `set_correlation_method()`: 动态设置相关性计算方法

**使用示例**:
```python
from lead_lag_network import LeadLagNetworkBuilder
from qlib_data_provider import QlibDataProvider

# 初始化（指定相关性方法）
provider = QlibDataProvider("ashares", "2025-01-01", "2025-10-27")
builder_pearson = LeadLagNetworkBuilder(provider, 60, correlation_method="pearson")
builder_spearman = LeadLagNetworkBuilder(provider, 60, correlation_method="spearman")

# 构建网络
M, A = builder_pearson.build_network("overnight_lead_daytime", True)

# 运行时切换方法
M_spearman, A_spearman = builder_pearson.build_network(
    "overnight_lead_daytime", correlation_method="spearman"
)

# 批量构建多种网络
networks = builder_pearson.build_multiple_networks(
    network_types=["overnight_lead_daytime", "daytime_lead_overnight"],
    correlation_methods=["pearson", "spearman"]
)

# 比较相关性方法
comparison = builder_pearson.compare_correlation_methods()
print(f"Pearson平均相关性: {comparison['pearson']['mean_correlation']:.4f}")
print(f"Spearman平均相关性: {comparison['spearman']['mean_correlation']:.4f}")

# 获取方法推荐
recommendation = builder_pearson.get_method_recommendation()
print(f"推荐方法: {recommendation['recommended_method']}")
print(f"推荐理由: {recommendation['reason']}")
```

**相关性方法说明**:
- **Pearson**: 线性相关性，计算效率高，适用于线性关系
- **Spearman**: 单调相关性，基于秩次，对异常值稳健，适用于非线性关系

### PortfolioConstructor

投资组合策略构建类。

**主要方法**:
- `construct_portfolio()`: 构建多空投资组合
- `backtest_strategy()`: 回测完整策略
- `calculate_metrics()`: 计算性能指标

### LeadLagFactorCalculator

领先-滞后因子计算器（新增模块）。

**主要功能**:
- 基于d-LE-SC聚类结果计算领先-滞后得分
- 生成交易信号
- 选择多空股票
- 计算完整的因子值时间序列

**主要方法**:
- `compute_lead_lag_scores()`: 计算领先-滞后得分
- `generate_trading_signal()`: 生成交易信号
- `select_top_and_bottom_stocks()`: 选择多空股票
- `compute_factor_values()`: 计算完整因子值矩阵

**使用示例**:
```python
from factor_computation import LeadLagFactorCalculator

# 初始化因子计算器
calculator = LeadLagFactorCalculator(
    lead_percentile=0.5,
    top_percentile=0.2,
    bottom_percentile=0.2
)

# 计算因子值
factor_df = calculator.compute_factor_values(
    adjacency_matrices=A,
    signed_matrices=M,
    clustering_results=clustering_results,
    returns_matrix=returns,
    stock_codes=stock_codes,
    date_index=date_index
)
```

**向后兼容函数**:
```python
# 旧版函数接口仍然可用
from factor_computation import (
    compute_lead_lag_scores,
    sorted_values,
    generate_trading_signal,
    select_top_and_bottom_stocks
)
```

### QlibDataProvider

qlib数据提供者类，用于获取真实金融数据。

**主要方法**:
- `get_stock_pool()`: 获取股票池
- `get_decomposed_returns()`: 获取分解后的收益率数据
- `initialize()`: 初始化qlib连接

**使用示例**:
```python
import os
from qlib_data_provider import QlibDataProvider

# 方法1：使用环境变量（推荐）
# export DOLPHINDB_URI="dolphindb://username:password@host:port"
provider = QlibDataProvider(
    database_uri=os.getenv("DOLPHINDB_URI"),
    region="REG_CN",
    market="ashares"
)

# 方法2：直接传入连接字符串
provider = QlibDataProvider(
    database_uri="dolphindb://your_username:your_password@your_host:8848",
    region="REG_CN",
    market="ashares"
)

# 获取股票池
stock_pool = provider.get_stock_pool("2024-01-01", "2024-03-31")

# 获取收益率数据
returns_data = provider.get_decomposed_returns(
    stock_pool[:10],  # 取前10只股票
    "2024-01-01",
    "2024-03-31",
    min_data_points=20
)
```

**[重构] 新的简化使用方式**：
```python
import sys
from pathlib import Path

# 添加项目路径到PYTHONPATH（根据实际情况调整）
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.insert(0, str(project_root / "qlib_ddb"))

from qlib_data_provider import QlibDataProvider
from lead_lag_network import LeadLagNetworkBuilder

# 初始化数据提供者（需先设置环境变量）
# export DOLPHINDB_URI="dolphindb://username:password@host:port"
provider = QlibDataProvider("ashares", "2025-01-01", "2025-10-27")

# 直接与网络构建器集成
network_builder = LeadLagNetworkBuilder(provider, 60)

# 构建网络
M, A = network_builder.build_network("overnight_lead_daytime", True)
```

## qlib数据源配置

### 数据库连接

本项目支持通过qlib连接到DolphinDB数据库获取真实金融数据：

**连接配置**：
- **数据库URI**: `dolphindb://username:password@host:port`
- **数据区域**: `REG_CN` (中国) 或 `REG_US` (美国)
- **股票市场**: `ashares` (A股), `a-shares` 等

### 收益率计算

qlib数据提供者自动计算三种收益率类型：

1. **日收益率**: `$close/$preclose-1`
2. **日间收益率**: `$close/$open-1`
3. **隔夜收益率**: `(1+r_daily)/(1+r_daytime)-1`

### 数据格式

返回的数据格式与模拟数据完全兼容：
```python
returns_data = {
    'STOCK_CODE': {
        'overnight': pd.Series,      # 隔夜收益率
        'daytime': pd.Series,        # 日间收益率
        'close_to_close': pd.Series  # 收盘到收盘收益率
    }
}
```

### 故障处理

如果qlib连接失败或数据获取失败，系统会自动回退到模拟数据：

```python
# 示例：qlib不可用时的处理
if not QLIB_AVAILABLE:
    logger.warning("qlib不可用，使用模拟数据替代")
    returns_data = create_sample_returns_data(n_stocks=30, n_days=100)
```

### 注意事项

1. **网络连接**: 确保能够访问DolphinDB数据库服务器
2. **数据权限**: 需要相应的数据库访问权限
3. **数据质量**: 检查返回数据的完整性和准确性
4. **性能考虑**: 大量股票的长时间数据获取可能较慢

## 性能指标

实现的主要性能指标包括：

- **年化收益率**: 投资组合的年化收益
- **夏普比率**: 风险调整后收益
- **最大回撤**: 历史最大损失
- **胜率**: 正收益日的比例
- **卡尔玛比率**: 年化收益/最大回撤

## 配置参数

### PortfolioConfig

```python
@dataclass
class PortfolioConfig:
    lead_percentile: float = 0.5      # 领导者股票中用于信号生成的比例
    top_percentile: float = 0.2       # 做多位置的比例
    bottom_percentile: float = 0.2    # 做空位置的比例
    min_cluster_size: int = 5         # 有效策略的最小聚类大小
    use_absolute_returns: bool = True # 是否使用绝对收益进行排序
```

## 实验结果

根据论文中的实证结果，使用d-LE-SC算法的策略表现优异：

### 隔夜-领先-日间策略
- 年化收益率: 32.11%
- 夏普比率: 2.37
- 最大回撤: 17.44%
- 胜率: 57.58%

### 日间-领先-隔夜策略
- 年化收益率: 15.79%
- 夏普比率: 2.09
- 最大回撤: 11.12%
- 胜率: 55.67%

## 注意事项

1. **Python路径配置**: 代码中已添加必要的路径配置以使用qlib和DataFeed库
2. **GPU支持**: 算法自动检测并使用GPU（如果可用）
3. **内存使用**: 大规模数据集可能需要大量内存，建议使用GPU加速
4. **数据质量**: 确保输入数据的完整性和准确性
5. **qlib依赖**: 使用真实数据需要安装pyqlib并配置数据库连接
6. **数据延迟**: qlib数据获取可能较慢，建议先用少量股票测试
7. **网络稳定性**: qlib连接需要稳定的网络环境访问数据库

## 文件输出

运行分析后，将在`results`目录生成以下文件：
- `analysis_results.png`: 分析结果可视化
- `backtest_results.png`: 回测性能图表
- `analysis_report.md`: 详细分析报告

## 引用

如果您使用本代码，请引用原论文：

```
Lu, Y., Zhang, N., Reinert, G., & Cucuringu, M. (2025).
A tug of war across the market: overnight-vs-daytime lead-lag networks
and clustering-based portfolio strategies.
```

## 许可证

本项目仅用于学术研究目的。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目维护者: Hugo <shen.lan123@gmail.com>
- 项目地址: https://github.com/hugo2046/QuantsPlaybook
- 基于论文: Oxford University Statistical Department
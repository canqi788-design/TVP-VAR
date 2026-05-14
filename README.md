# TVP-VAR 时变参数向量自回归分析框架

一个基于贝叶斯方法的时变参数向量自回归（Time-Varying Parameter Vector Autoregression）分析工具，用于多维财务数据的动态因果分析、脉冲响应和预测。本人纯新手小白，希望大佬们能指点下我。

## 项目结构

```
├── run_tvp_var_analysis.py      # 主入口脚本
├── dual_model_agent.py          # 双模型智能路由 Agent
├── dual_model_config.json       # 双模型配置（五粮液示例）
├── run_diagnostic.py            # 诊断测试脚本
├── forecast_research.csv        # 预测结果示例输出
├── config/                      # 配置加载模块
│   └── loader.py
└── tvp_var_framework/           # 核心框架
    ├── core/                    # 基础抽象
    │   ├── base_model.py        # 模型统一接口 (BaseTVPVARModel)
    │   ├── model_result.py      # 结果数据结构
    │   └── runtime_context.py   # 运行时上下文
    ├── models/                  # 模型实现
    │   ├── analyst.py           # TVP-VAR 分析器
    │   ├── bayesian.py          # 贝叶斯估计
    │   ├── fully_bayesian.py    # 全贝叶斯估计 (FFBS)
    │   ├── ffbs.py              # Forward Filtering Backward Sampling
    │   └── research_grade.py    # 研究级采样器 (IRF/FEVD/Forecast)
    ├── runtime/                 # 执行引擎
    │   ├── inference_engine.py  # DAG 推理引擎 (核心)
    │   ├── context.py           # 执行上下文
    │   ├── contracts.py         # 接口契约校验
    │   ├── estimators.py        # 估计后端调度
    │   └── formatter.py         # 输出格式化
    ├── diagnostics/             # 诊断模块
    │   ├── convergence.py       # 收敛诊断 (R-hat, ESS, Geweke)
    │   └── dependency_graph.py  # 依赖图分析
    ├── reporting/               # 报告生成
    │   └── report_generator.py  # Markdown 报告输出
    ├── utils/                   # 工具模块
    │   ├── data_loader.py       # 数据加载
    │   ├── stationarity.py      # 平稳性检验 (ADF)
    │   ├── backtesting.py       # 回测工具
    │   └── long_run.py          # 长期效应分析
    └── ir/                      # 中间表示
        └── inference_graph.json # DAG 执行图定义
```

## 核心特性

- **DAG 驱动执行引擎**: 基于有向无环图的推理流水线，节点通过拓扑排序自动调度
- **多种估计后端**: 支持 fully_bayesian、bayesian、v2、research 四种模式
- **Gibbs 采样**: 研究级采样器实现 FFBS（前向滤波后向采样），支持随机波动率（SV）
- **联合后验链**: 每次迭代保存耦合的后验快照（theta, sigma, sv），保证后验依赖完整性
- **结构分析**: 脉冲响应函数（IRF）、方差分解（FEVD）、Granger 因果检验
- **蒙特卡洛预测**: 基于后验样本的区间预测
- **收敛诊断**: R-hat、有效样本量（ESS）、Geweke 诊断
- **双模型路由**: 基于关键词匹配 + Granger 显著性的智能模型选择

## 快速开始

### 演示模式

```bash
python run_tvp_var_analysis.py --demo
```

### 使用 JSON 配置文件

```bash
python run_tvp_var_analysis.py config.json
```

### 指定企业数据

```bash
python run_tvp_var_analysis.py --company "企业名称" --csv data.csv --vars "revenue,profit,cost"
```

### 生成示例配置

```bash
python run_tvp_var_analysis.py --create-config my_config.json
```

### 双模型 Agent

```bash
# 运行关键词路由分析
python dual_model_agent.py "研发投入对利润率的影响"

# 列出可用模型
python dual_model_agent.py --list

# 使用自定义配置
python dual_model_agent.py --config my_config.json "查询内容"
```

## 配置说明

配置文件采用 JSON 格式，主要段落如下：

```json
{
    "data_ingestion": {
        "data_path": "data.csv",
        "vars": ["revenue", "profit"],
        "normalize": true,
        "handle_missing": "interpolate"
    },
    "model_specification": {
        "mode": "research"
    },
    "inference_control": {
        "n_iter": 3000,
        "burnin": 1000,
        "thin": 2
    },
    "stochastic_volatility": {
        "enabled": true,
        "sv_n_iter": 500,
        "sv_burnin": 200
    },
    "structural_analysis": {
        "irf_periods": 6,
        "change_point_threshold": 1.5
    },
    "forecasting": {
        "steps": 4,
        "n_samples": 500
    },
    "convergence_diagnostics": {
        "rhat_threshold": 1.1,
        "ess_minimum": 100
    }
}
```

### 模型模式

| 模式 | 说明 |
|------|------|
| `fully_bayesian` | 全贝叶斯估计 |
| `bayesian` | 标准贝叶斯估计 |
| `v2` | 第二版估计器 |
| `research` | 研究级 Gibbs 采样（推荐） |
| `full` | 同时运行所有模式 |

## 输出

- **Markdown 报告**: 包含 IRF、FEVD、Granger 因果检验、预测、后验参数估计
- **CSV 导出**: 预测结果的均值和置信区间
- **控制台日志**: 执行过程、收敛诊断、参数路由信息

## 依赖

- Python 3.8+
- NumPy

## 数据格式

输入 CSV 文件需包含时间索引列和数值变量列：

```csv
year,revenue,profit,cost
2018,100.0,25.0,75.0
2019,120.0,30.0,90.0
...
```

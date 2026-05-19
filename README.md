# TVP-VAR 贝叶斯时变参数向量自回归分析框架

基于 DAG 推理引擎的 TVP-VAR（Time-Varying Parameter Vector Autoregression）分析工具，支持 A 股财务数据的累计口径自动转换、贝叶斯 MCMC 采样、脉冲响应分析与因果检验。

## 项目结构

```
TVP-VAR/
├── run_tvp_var_analysis.py          # 主入口（单模型）
├── dual_model_agent.py              # 双模型 Agent（关键词路由 + Granger 覆盖）
├── run_diagnostic.py                # 诊断工具
├── config/                          # 配置加载器
│   └── loader.py                    # 深合并多配置文件
├── tvp_var_framework/               # 核心框架
│   ├── core/                        # 基础组件
│   │   ├── base_model.py            # 基类
│   │   ├── model_result.py          # 后验结果结构
│   │   ├── theta_layout.py          # theta 向量布局
│   │   └── runtime_context.py       # 运行上下文
│   ├── models/                      # 采样器
│   │   ├── bayesian.py              # 贝叶斯 TVP-VAR（Kalman + FFBS）
│   │   ├── fully_bayesian.py        # 全贝叶斯 Gibbs 采样
│   │   ├── ffbs.py                  # 前向滤波后向采样
│   │   ├── analyst.py               # TVP_VAR_Analyst（逐期 Kalman 更新）
│   │   └── research_grade.py        # 研究级模块
│   ├── runtime/                     # 运行时
│   │   ├── inference_engine.py      # DAG 推理引擎（InferenceGraphEngine）
│   │   ├── estimators.py            # 估计器
│   │   ├── contracts.py             # 接口契约
│   │   └── formatter.py             # 输出格式化
│   ├── diagnostics/                 # 诊断
│   │   ├── convergence.py           # MCMC 收敛诊断
│   │   └── dependency_graph.md      # 后验依赖图
│   ├── utils/                       # 工具
│   │   ├── data_loader.py           # 数据加载 + YTD 转单季 + 季节变换
│   │   ├── stationarity.py          # 平稳性检验（ADF/KPSS）
│   │   ├── stability.py             # 谱半径稳定性
│   │   ├── backtesting.py           # 回测
│   │   └── long_run.py              # 长期均衡
│   ├── reporting/
│   │   └── report_generator.py      # Markdown/CSV 报告生成
│   └── ir/
│       └── inference_graph.json     # DAG 节点定义
├── wuliangye_full_config.json       # 五粮液双模型配置
├── wuliangye_dual_model_config.json # 五粮液双模型配置（含季节参数）
├── wuliangye_full_quarterly.csv     # 五粮液季度财务数据（A 股累计口径）
└── 跨行业通用适配与参数调优指南.md    # 行业适配文档
```

## 快速开始

### 环境要求

- Python 3.9+
- numpy

### Demo 模式

```bash
python3 run_tvp_var_analysis.py --demo
```

生成示例数据并运行完整分析流程，输出到 `./demo/`。

### 使用配置文件

```bash
python3 run_tvp_var_analysis.py wuliangye_full_config.json
```

### 命令行指定数据

```bash
python3 run_tvp_var_analysis.py --company "企业名" --csv data.csv --vars "var1,var2,var3"
```

### 双模型 Agent

```bash
# 列出可用模型
python3 dual_model_agent.py --config wuliangye_full_config.json --list

# 按关键词路由运行
python3 dual_model_agent.py --config wuliangye_full_config.json "盈利分析"

# 指定模型运行
python3 dual_model_agent.py --config wuliangye_full_config.json --model profitability_driver "分析"

# 全量对照 + Granger 覆盖
python3 dual_model_agent.py --config wuliangye_full_config.json --compare "盈利分析"
```

## A 股数据处理

### 累计口径自动转换

A 股财报数据为累计口径（Q2 包含 Q1 数据），框架自动处理：

1. **检测**：`_looks_like_ytd()` 判断数据是否为累计格式（Q1 重置 + 年内单调递增）
2. **差分**：`ytd_to_single_period()` 将累计值转为单季发生额（Q1 不动，Q2-Q4 减上期累计）
3. **比率重算**：`ytd_financial_table_to_single_period()` 用转换后的流量重新计算派生比率：
   - 毛利率 = (营业总收入 - 营业成本) / 营业总收入
   - 销售费用率 = 销售费用 / 营业总收入
   - 管理费用率 = 管理费用 / 营业总收入
   - 净利率 = 净利润 / 营业总收入

配置项 `flow_transform`：`"auto"`（自动检测）| `"ytd_to_quarter"`（强制转换）| `"none"`（不处理）

### 季节变换

配置项 `seasonal_transform`：

| 值 | 说明 |
|------|------|
| `"none"` | 不处理 |
| `"ttm"` | 滚动四季合计（Trailing Twelve Months） |
| `"yoy"` | 同比变化率（Year-over-Year） |

## 配置格式

```json
{
    "data_source": "data.csv",
    "mode": "bayesian",
    "flow_transform": "ytd_to_quarter",
    "seasonal_transform": "none",
    "bayesian_priors": {
        "q_grid": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    },
    "inference_control": {
        "n_iter": 2000,
        "burnin": 500,
        "thin": 1
    },
    "stochastic_volatility": {
        "enabled": true,
        "sv_n_iter": 300,
        "sv_burnin": 100
    },
    "stationarity": {
        "test": "both",
        "max_d": 2,
        "significance": 0.05
    },
    "forecasting": { "steps": 4, "n_samples": 500 },
    "structural_analysis": { "irf_periods": 6 },
    "stability_guard": { "enforce": true, "spectral_radius_threshold": 0.98 },
    "convergence_diagnostics": { "rhat_threshold": 1.1, "ess_minimum": 100 },
    "models": {
        "model_key": {
            "name": "模型名称",
            "vars": ["变量1", "变量2"],
            "exog_vars": ["外生变量"],
            "keywords": ["关键词"]
        }
    }
}
```

## 执行流程

DAG 推理引擎按以下节点顺序执行：

```
data → stationarity → state_update → likelihood → sampling_basic → sampling_research → diagnostics → reporting
```

各节点功能：

| 节点 | 说明 |
|------|------|
| `data` | 加载 CSV、YTD 转单季、季节变换、归一化 |
| `stationarity` | ADF/KPSS 平稳性检验，确定差分阶数 |
| `state_update` | Kalman 滤波 / FFBS 状态更新 |
| `likelihood` | 似然计算 |
| `sampling_basic` | 贝叶斯/全贝叶斯 MCMC 采样 |
| `sampling_research` | 研究级模块（可选） |
| `diagnostics` | R-hat、ESS、Geweke 收敛诊断 |
| `reporting` | Markdown 报告 + CSV 导出 |

## 输出内容

- **MCMC 收敛诊断**：R-hat、ESS、Geweke p 值
- **平稳性检验**：ADF/KPSS 联合判定，差分阶数
- **脉冲响应（IRF）**：结构化冲击的动态响应路径
- **方差分解（FEVD）**：各冲击对变量波动的贡献度
- **Granger 因果检验**：变量间因果方向与显著性
- **预测（Forecast）**：多步预测与置信区间
- **结构突变检测**：系数时变的突变时点
- **稳定性诊断**：谱半径、接近单位根比例
- **后验参数估计**：系数均值与 95% CI

## 采样模式

| 模式 | 说明 |
|------|------|
| `bayesian` | Kalman 滤波 + 解析后验（快速） |
| `fully_bayesian` | Gibbs 采样（FFBS + SV） |
| `v2` | TVP-VAR v2 研究级模块 |
| `research` | 研究级分析 |
| `full` | 运行所有模式 |

## 注意事项

- A 股季度数据样本量有限（通常 28-32 个观测），ADF 检验力受限，平稳性结论需谨慎解读
- 平稳性检验的 `test` 参数建议用 `"adf"` 而非 `"both"`，`max_d` 建议设为 1，避免过度差分
- 模型有效性门禁会标记未通过平稳性检验的输出为"不建议用于业务结论"
- 时变参数（theta_traj）可通过 `analyst.history` 获取每一步的时变 A 矩阵
- ADF + KPSS在对波动性较大的行业时必须改为：原配置 + ADF 单独判定 + max_d=1

"""
TVP-VAR 分析工具 — 单一执行路径

所有入口统一走 InferenceGraphEngine (DAG 驱动)。

使用方式:
  python run_tvp_var_analysis.py --demo
  python run_tvp_var_analysis.py config.json
  python run_tvp_var_analysis.py --company "企业" --csv data.csv
"""

import numpy as np
import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvp_var_framework.utils.data_loader import load_company_data, load_json_config, split_exogenous_columns
from tvp_var_framework.runtime.inference_engine import InferenceGraphEngine
from config.loader import load_config

logger = logging.getLogger("tvp_var")

IR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tvp_var_framework", "ir", "inference_graph.json",
)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


# ============================================================
# 配置校验
# ============================================================

def normalize_config_schema(cfg):
    """Accept compact single-model configs and map them to engine sections."""
    cfg = dict(cfg)

    data_ingestion = dict(cfg.get("data_ingestion", {}))
    if "data_source" in cfg and "data_path" not in data_ingestion:
        data_ingestion["data_path"] = cfg["data_source"]
    if "flow_transform" in cfg and "flow_transform" not in data_ingestion:
        data_ingestion["flow_transform"] = cfg["flow_transform"]
    if "seasonal_transform" in cfg and "seasonal_transform" not in data_ingestion:
        data_ingestion["seasonal_transform"] = cfg["seasonal_transform"]
    cfg["data_ingestion"] = data_ingestion

    model_spec = dict(cfg.get("model_specification", {}))
    if "mode" in cfg and "mode" not in model_spec:
        model_spec["mode"] = cfg["mode"]
    cfg["model_specification"] = model_spec

    return cfg


def validate_config(cfg):
    """校验 JSON 配置"""
    if "data_ingestion" not in cfg:
        raise ValueError("缺少必填段 data_ingestion")
    if "model_specification" not in cfg:
        raise ValueError("缺少必填段 model_specification")
    if cfg["data_ingestion"].get("data_path") is None:
        raise ValueError("data_ingestion.data_path 不能为空")
    if not cfg["data_ingestion"].get("vars"):
        raise ValueError("data_ingestion.vars 不能为空")


# ============================================================
# Logging
# ============================================================

def setup_logging(log_level="INFO"):
    """配置日志系统"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ============================================================
# 数据加载 + 上下文组装
# ============================================================

def load_data_and_build_ctx(cfg):
    """加载数据，组装 engine 所需的 initial_data dict"""
    data_cfg = cfg.get("data_ingestion", {})

    load_vars = list(data_cfg.get("vars", []))
    exog_vars = list(data_cfg.get("exog_vars", []))
    for name in exog_vars:
        if name not in load_vars:
            load_vars.append(name)

    Y_loaded, ql, loaded_names, loaded_mean, loaded_std = load_company_data(
        data_cfg["data_path"], load_vars,
        noise_scale=data_cfg.get("noise_scale", 0.02),
        normalize_data=data_cfg.get("normalize", True),
        handle_missing=data_cfg.get("handle_missing", "none"),
        random_state=data_cfg.get("random_state"),
        flow_transform=data_cfg.get("flow_transform", "auto"),
        seasonal_transform=data_cfg.get("seasonal_transform", "none"),
    )
    Y, X_exog, vn, exog_names = split_exogenous_columns(Y_loaded, loaded_names, exog_vars)
    endog_idx = [loaded_names.index(name) for name in vn]
    mean = loaded_mean[endog_idx]
    std = loaded_std[endog_idx]

    ctx = {
        "config": cfg,
        "Y": Y,
        "var_names": vn,
        "time_index": ql,
        "mean": mean,
        "std": std,
        "normalize": data_cfg.get("normalize", True),
        "debug": cfg.get("debug_mode", False),
    }
    if X_exog is not None:
        ctx["X_exog"] = X_exog
        ctx["exog_names"] = exog_names
    return ctx


def run_engine(cfg):
    """统一执行入口：加载数据 → 组装 ctx → engine.run()"""
    cfg = normalize_config_schema(cfg)
    setup_logging(cfg.get("log_level", "INFO"))
    validate_config(cfg)

    ctx = load_data_and_build_ctx(cfg)
    engine = InferenceGraphEngine(IR_PATH)
    return engine.run(ctx)


def run_from_config_dir(config_dir, data_path=None, vars_list=None):
    """从 config/ 目录加载 3 个配置文件执行"""
    setup_logging("INFO")
    cfg = load_config(config_dir)

    # 数据路径可以从参数或 config 中获取
    if data_path:
        cfg.setdefault("data_ingestion", {})["data_path"] = data_path
    if vars_list:
        cfg.setdefault("data_ingestion", {})["vars"] = vars_list

    validate_config(cfg)
    ctx = load_data_and_build_ctx(cfg)
    engine = InferenceGraphEngine(IR_PATH)
    return engine.run(ctx)


# ============================================================
# 示例数据 / 配置生成
# ============================================================

def create_example_config():
    """生成示例配置文件 (完整格式)

    配置层级规范 (SSOT):
      - model_specification: 模型结构参数 (mode, 先验类型)
      - inference_control:   MCMC 采样控制 (n_iter, burnin, thin)
      - stochastic_volatility: SV 开关及参数
    """
    return {
        "debug_mode": False,
        "log_level": "INFO",
        "report_control": {
            "enabled": True,
            "fast_mode": False,
            "section_logging": True,
            "disable_full_markdown": False,
        },
        "safety_fallbacks": {
            "max_irf_periods": 50,
            "max_forecast_samples": 500,
        },
        "project_metadata": {
            "company": "示例企业",
            "industry_category": "制造业",
            "report_title": "TVP-VAR 多维财务动态因果分析报告",
        },
        "data_ingestion": {
            "data_path": "example_data.csv",
            "vars": ["revenue", "depreciation"],
            "var_names": ["revenue", "depreciation"],
            "normalize": True,
            "handle_missing": "interpolate",
            "noise_scale": 0.02,
            "flow_transform": "auto",
            "seasonal_transform": "none",
        },
        "model_specification": {
            "mode": "full",
        },
        "inference_control": {
            "n_iter": 2000,
            "burnin": 1000,
            "thin": 2,
        },
        "stochastic_volatility": {
            "enabled": True,
            "sv_n_iter": 500,
            "sv_burnin": 200,
        },
        "forecasting": {
            "steps": 4,
            "n_samples": 1000,
        },
        "stability_guard": {
            "enforce": False,
            "spectral_radius_threshold": 1.0,
        },
        "structural_analysis": {
            "irf_periods": 8,
            "identification": "Cholesky",
            "change_point_threshold": 1.5,
        },
        "stationarity": {
            "enabled": True,
            "test": "both",
            "max_d": 2,
            "significance": 0.05,
        },
        "convergence_diagnostics": {
            "enabled": True,
            "rhat_threshold": 1.1,
            "ess_minimum": 100,
        },
        "output_control": {
            "export_csv": True,
            "output_dir": "./analysis_results/",
        },
    }


def create_example_csv(path="example_data.csv"):
    """生成示例 CSV 数据文件"""
    rng = np.random.default_rng(42)
    years = list(range(2018, 2025))
    with open(path, "w", encoding="utf-8") as f:
        f.write("year,revenue,depreciation,gross_margin\n")
        base_rev = 100
        base_dep = 30
        for y in years:
            rev = base_rev + (y - 2018) * 20 + rng.normal(0, 10)
            dep = base_dep + (y - 2018) * 8 + rng.normal(0, 3)
            gm = 25 + rng.normal(0, 3)
            f.write(f"{y},{rev:.2f},{dep:.2f},{gm:.2f}\n")
    return path


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TVP-VAR 分析工具")
    parser.add_argument("config", nargs="?", help="JSON 配置文件路径")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--company", help="企业名称")
    parser.add_argument("--csv", help="CSV 数据文件路径")
    parser.add_argument("--vars", help="变量列表, 逗号分隔")
    parser.add_argument("--mode", default="full",
                        choices=["fully_bayesian", "bayesian", "v2", "research", "full"])
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--create-config", help="生成示例配置到指定路径")
    parser.add_argument("--config-dir", help="从 config/ 目录加载 3 个配置文件")

    args = parser.parse_args()

    if args.create_config:
        cfg = create_example_config()
        with open(args.create_config, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
        print(f"示例配置已写入: {args.create_config}")
        sys.exit(0)

    if args.demo:
        csv_path = create_example_csv()
        config = create_example_config()
        config["data_ingestion"]["data_path"] = csv_path
        config["output_control"]["output_dir"] = "./demo/"
        if args.mode:
            config["model_specification"]["mode"] = args.mode
        run_engine(config)
        os.remove(csv_path)
        sys.exit(0)

    if args.config_dir:
        run_from_config_dir(args.config_dir)
        sys.exit(0)

    if args.company and args.csv:
        vars_list = args.vars.split(",") if args.vars else None
        if vars_list is None:
            import csv as csv_mod
            with open(args.csv, "r") as f:
                reader = csv_mod.DictReader(f)
                headers = [h for h in reader.fieldnames if h != reader.fieldnames[0]]
            vars_list = headers

        # 输出目录: ./<company>/
        output_dir = f"./{args.company}/"
        os.makedirs(output_dir, exist_ok=True)

        config = {
            "project_metadata": {"company": args.company},
            "data_ingestion": {
                "data_path": args.csv, "vars": vars_list, "var_names": vars_list,
                "normalize": not args.no_normalize,
                "flow_transform": "auto",
                "seasonal_transform": "none",
            },
            "model_specification": {"mode": args.mode},
            "stationarity": {"significance": 0.05, "max_d": 2, "test": "both"},
            "stochastic_volatility": {"enabled": True, "sv_n_iter": 300, "sv_burnin": 100},
            "structural_analysis": {"irf_periods": 6, "change_point_threshold": 1.5},
            "forecasting": {"steps": 4, "n_samples": 500},
            "stability_guard": {"enforce": False, "spectral_radius_threshold": 1.0},
            "convergence_diagnostics": {"rhat_threshold": 1.1, "ess_minimum": 100},
            "output_control": {"export_csv": True, "output_dir": output_dir},
            "report_control": {"enabled": True},
            "inference_control": {"n_iter": 2000, "burnin": 500, "thin": 1},
        }

        # 复制原始数据到输出目录
        import shutil
        shutil.copy2(args.csv, os.path.join(output_dir, os.path.basename(args.csv)))

        run_engine(config)
        sys.exit(0)

    if args.config:
        cfg = load_json_config(args.config)
        run_engine(cfg)
        sys.exit(0)

    parser.print_help()
    print(f"""
示例:
  python {sys.argv[0]} --demo
  python {sys.argv[0]} config.json
  python {sys.argv[0]} --company "企业" --csv data.csv
  python {sys.argv[0]} --create-config config.json
""")

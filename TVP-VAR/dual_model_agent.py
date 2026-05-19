"""
双模型 TVP-VAR Agent — 配置驱动

从 dual_model_config.json 读取模型定义、关键词、数据源，
默认根据查询关键词先路由到一个经济结构模型，再只运行该模型。
如需全量对照，可使用 --compare 跑完所有模型并通过 Granger 显著性置信度覆盖。

用法:
  python3 dual_model_agent.py "你的查询"
  python3 dual_model_agent.py --config my_config.json "你的查询"
  python3 dual_model_agent.py --compare "你的查询"
  python3 dual_model_agent.py --model profitability_driver "盈利能力分析"
  python3 dual_model_agent.py --list  # 列出可用模型和关键词
"""
import sys
import os
import json
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dual_agent")

from run_tvp_var_analysis import run_engine
from tvp_var_framework.runtime.inference_engine import _compute_granger_causality


# ============================================================
# 配置加载
# ============================================================

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dual_model_config.json")


def load_dual_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# DualModelRunner
# ============================================================

class DualModelRunner:
    """用不同的变量子集运行 engine。"""

    def __init__(self, config):
        self.config = config

    def validate_vars(self, model_keys=None):
        """校验指定模型的变量是否存在于 CSV 表头中。"""
        import csv as csv_mod
        data_path = self.config["data_source"]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            headers = csv_mod.reader(f).__next__()

        if model_keys is None:
            model_keys = list(self.config["models"].keys())
        elif isinstance(model_keys, str):
            model_keys = [model_keys]

        all_vars = set()
        for key in model_keys:
            mc = self.config["models"][key]
            all_vars.update(mc["vars"])
            all_vars.update(mc.get("exog_vars", []))

        missing = [v for v in all_vars if v not in headers]
        if missing:
            raise KeyError(
                f"CSV 列名不匹配: {missing}\n"
                f"  可用列: {headers}\n"
                f"  请检查 config 中 models.*.vars / exog_vars 是否与 CSV 表头一致"
            )
        logger.info(f"列名校验通过 ({', '.join(model_keys)}): {headers}")

    def _build_engine_config(self, model_key):
        mc = self.config["models"][model_key]
        forecasting = dict(self.config.get("forecasting", {"steps": 4, "n_samples": 500}))
        if mc.get("forecast_bounds") is not None:
            forecasting["bounds"] = mc["forecast_bounds"]
        cfg = {
            "log_level": "WARNING",
            "data_ingestion": {
                "data_path": self.config["data_source"],
                "vars": mc["vars"],
                "exog_vars": mc.get("exog_vars", []),
                "var_names": mc["vars"],
                "normalize": True,
                "handle_missing": "interpolate",
                "noise_scale": 0.02,
                "flow_transform": self.config.get("flow_transform", "none"),
                "seasonal_transform": self.config.get("seasonal_transform", "none"),
            },
            "model_specification": {"mode": self.config.get("mode", "bayesian")},
            "bayesian_priors": self.config.get("bayesian_priors", {}),
            "prior_hyperparameters": self.config.get("prior_hyperparameters", {}),
            "inference_control": self.config.get("inference_control", {"n_iter": 3000, "burnin": 1000, "thin": 2}),
            "mcmc_options": self.config.get("mcmc_options", {}),
            "stochastic_volatility": self.config.get("stochastic_volatility", {"enabled": True}),
            "forecasting": forecasting,
            "structural_analysis": self.config.get("structural_analysis", {"irf_periods": 6}),
            "stationarity": self.config.get("stationarity", {"test": "adf", "max_d": 1, "significance": 0.05, "log_transform": True}),
            "stationarity_options": self.config.get("stationarity_options", {}),
            "stability_guard": self.config.get("stability_guard", {"enforce": False, "spectral_radius_threshold": 1.0}),
            "convergence_diagnostics": self.config.get("convergence_diagnostics", {"rhat_threshold": 1.1, "ess_minimum": 100}),
            "output_control": {"export_csv": True, "output_dir": f"./{model_key}_output/"},
            "report_control": {"enabled": True},
        }
        return cfg

    def run_model(self, model_key):
        mc = self.config["models"][model_key]
        logger.info(f"运行 {model_key}: {mc['name']} — 变量: {mc['vars']}")
        cfg = self._build_engine_config(model_key)
        return run_engine(cfg)

    def run_selected(self, model_key):
        return {model_key: self.run_model(model_key)}

    def run_all(self):
        results = {}
        for key in self.config["models"]:
            results[key] = self.run_model(key)
        return results


# ============================================================
# QueryRouter
# ============================================================

class QueryRouter:
    """根据查询中的关键词选择模型。"""

    def __init__(self, config):
        self.models = config["models"]

    def route(self, query):
        q = query.lower()
        scores = {}
        for key, mc in self.models.items():
            kws = mc.get("keywords", [])
            scores[key] = sum(1 for kw in kws if kw in q)

        # 按得分降序排列
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best_key, best_score = ranked[0]

        # 平局: 选第一个模型
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            best_key = list(self.models.keys())[0]

        return best_key, scores


# ============================================================
# ConfidenceChecker
# ============================================================

class ConfidenceChecker:
    """检查 Granger 显著性，必要时覆盖路由。"""

    @staticmethod
    def _extract_granger(ctx):
        chain = ctx.outputs.get("_joint_chain", [])
        if not chain:
            return {}
        n = len(ctx.var_names)
        return _compute_granger_causality(chain, ctx.var_names, n)

    @staticmethod
    def _count_significant(granger):
        return sum(1 for v in granger.values() if v.get("significant", False))

    def check(self, all_results, keyword_choice):
        granger_results = {}
        sig_counts = {}
        for key, ctx in all_results.items():
            g = self._extract_granger(ctx)
            granger_results[key] = g
            sig_counts[key] = self._count_significant(g)
            logger.info(f"Granger 显著路径: {key} = {sig_counts[key]}")

        chosen_sig = sig_counts.get(keyword_choice, 0)

        # 如果关键词选中的模型有显著路径，确认
        if chosen_sig > 0:
            return keyword_choice, granger_results, "confirmed"

        # 否则找显著路径最多的模型
        best = max(sig_counts, key=sig_counts.get)
        if sig_counts[best] > chosen_sig:
            logger.warning(f"覆盖: {keyword_choice} 无显著路径，切换到 {best} ({sig_counts[best]} 条)")
            return best, granger_results, "override"

        return keyword_choice, granger_results, "confirmed"


# ============================================================
# 报告输出
# ============================================================

def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_granger_summary(granger, var_names, model_name):
    print_separator(f"Granger 因果检验 — {model_name}")
    sig_paths = []
    for (src, dst), info in sorted(granger.items()):
        sig = "是" if info["significant"] else "否"
        ci = info["ci_95"]
        print(f"  {src} -> {dst}:  mean={info['mean']:>8.4f}  CI=[{ci[0]:>7.3f}, {ci[1]:>7.3f}]  显著={sig}  P(正)={info['prob_positive']:.1%}")
        if info["significant"]:
            sig_paths.append((src, dst, info))
    if not sig_paths:
        print("  (无显著因果路径)")
    return sig_paths


def print_decision(config, scores, final_choice, reason, sig_counts):
    print_separator("Agent 决策")
    models = config["models"]
    for key, score in scores.items():
        name = models[key]["name"]
        print(f"  关键词得分: {key} ({name}) = {score}")
    print(f"  Granger 显著: {', '.join(f'{k}={v}' for k, v in sig_counts.items())}")
    chosen_name = models[final_choice]["name"]
    chosen_vars = models[final_choice]["vars"]
    if reason == "override":
        print(f"  决策: 覆盖 → {chosen_name} (显著性优先)")
    else:
        print(f"  决策: 确认 → {chosen_name} (关键词匹配)")
    print(f"  最终选用变量: {chosen_vars}")


def print_list(config):
    """列出配置中的模型和关键词。"""
    print_separator("可用模型")
    for key, mc in config["models"].items():
        print(f"\n  {key}: {mc['name']}")
        print(f"    变量: {mc['vars']}")
        print(f"    关键词: {', '.join(mc['keywords'])}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="双模型 TVP-VAR Agent")
    parser.add_argument("query", nargs="?", help="分析查询")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--model", help="显式指定模型 key，跳过关键词路由")
    parser.add_argument("--compare", action="store_true", help="运行所有模型做对照，并允许 Granger 显著性覆盖")
    parser.add_argument("--list", action="store_true", help="列出可用模型和关键词")
    args = parser.parse_args()

    config = load_dual_config(args.config)

    if args.list:
        print_list(config)
        print(f"\n  数据源: {config['data_source']}")
        sys.exit(0)

    if not args.query:
        parser.print_help()
        print(f"\n示例:")
        print(f'  python3 {sys.argv[0]} "研发投入对利润率的影响"')
        print(f'  python3 {sys.argv[0]} "资本开支对现金流的影响"')
        print(f'  python3 {sys.argv[0]} --config company_config.json "查询"')
        print(f'  python3 {sys.argv[0]} --compare "查询"')
        print(f'  python3 {sys.argv[0]} --model profitability_driver "查询"')
        print(f'  python3 {sys.argv[0]} --list')
        sys.exit(1)

    query = args.query
    models = config["models"]
    if args.model and args.model not in models:
        raise KeyError(f"未知模型: {args.model}; 可用模型: {list(models.keys())}")

    # 1. 列出模型
    print_separator("启动双模型分析")
    print(f"  查询: {query}")
    print(f"  执行模式: {'全量对照 (--compare)' if args.compare else '选择性使用'}")
    for key, mc in models.items():
        print(f"  {key} ({mc['name']}): {mc['vars']}")

    # 2. 先在业务语义层路由，决定本次使用哪个经济结构模型
    router = QueryRouter(config)
    keyword_choice, scores = router.route(query)
    selected_key = args.model or keyword_choice
    print(f"\n  关键词得分: {scores}")
    if args.model:
        print(f"  手动指定模型: {selected_key}")
    else:
        print(f"  路由选中模型: {selected_key}")

    # 3. 校验列名 + 运行模型。默认只运行路由选中的模型；--compare 才运行全部。
    runner = DualModelRunner(config)
    if args.compare:
        runner.validate_vars()
        all_results = runner.run_all()
        checker = ConfidenceChecker()
        final_choice, granger_results, reason = checker.check(all_results, selected_key)
    else:
        runner.validate_vars(selected_key)
        all_results = runner.run_selected(selected_key)
        checker = ConfidenceChecker()
        granger_results = {}
        sig_counts_for_selected = {}
        for key, ctx in all_results.items():
            g = checker._extract_granger(ctx)
            granger_results[key] = g
            sig_counts_for_selected[key] = checker._count_significant(g)
            logger.info(f"Granger 显著路径: {key} = {sig_counts_for_selected[key]}")
        final_choice = selected_key
        reason = "manual" if args.model else "selected"

    # 4. 打印已运行模型的 Granger 结果
    sig_counts = {}
    for key, ctx in all_results.items():
        g = granger_results[key]
        name = config["models"][key]["name"]
        sig_paths = print_granger_summary(g, ctx.var_names, name)
        sig_counts[key] = len(sig_paths)

    # 5. 打印决策
    print_decision(config, scores, final_choice, reason, sig_counts)

    # 6. 输出最终报告
    final_ctx = all_results[final_choice]
    report_path = final_ctx.outputs.get("_report_path", "未生成")
    final_name = config["models"][final_choice]["name"]
    final_vars = config["models"][final_choice]["vars"]

    print_separator("最终报告")
    print(f"  报告路径: {report_path}")
    print(f"  选用模型: {final_name}")
    print(f"  变量: {final_vars}")
    print()


if __name__ == "__main__":
    main()

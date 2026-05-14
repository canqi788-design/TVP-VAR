"""
双模型 TVP-VAR Agent — 配置驱动

从 dual_model_config.json 读取模型定义、关键词、数据源，
根据查询关键词路由到对应模型，并通过 Granger 显著性置信度覆盖。

用法:
  python3 dual_model_agent.py "你的查询"
  python3 dual_model_agent.py --config my_config.json "你的查询"
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

    def validate_vars(self):
        """校验所有模型的变量是否存在于 CSV 表头中"""
        import csv as csv_mod
        data_path = self.config["data_source"]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            headers = csv_mod.reader(f).__next__()

        all_vars = set()
        for key, mc in self.config["models"].items():
            all_vars.update(mc["vars"])

        missing = [v for v in all_vars if v not in headers]
        if missing:
            raise KeyError(
                f"CSV 列名不匹配: {missing}\n"
                f"  可用列: {headers}\n"
                f"  请检查 config 中 models.*.vars 是否与 CSV 表头一致"
            )
        logger.info(f"列名校验通过: {headers}")

    def _build_engine_config(self, model_key):
        mc = self.config["models"][model_key]
        cfg = {
            "log_level": "WARNING",
            "data_ingestion": {
                "data_path": self.config["data_source"],
                "vars": mc["vars"],
                "var_names": mc["vars"],
                "normalize": True,
                "handle_missing": "interpolate",
                "noise_scale": 0.02,
            },
            "model_specification": {"mode": "research"},
            "inference_control": self.config.get("inference_control", {"n_iter": 3000, "burnin": 1000, "thin": 2}),
            "stochastic_volatility": self.config.get("stochastic_volatility", {"enabled": True}),
            "forecasting": self.config.get("forecasting", {"steps": 4, "n_samples": 500}),
            "structural_analysis": self.config.get("structural_analysis", {"irf_periods": 6}),
            "stationarity": self.config.get("stationarity", {"test": "adf", "max_d": 1, "significance": 0.05, "log_transform": True}),
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
        print(f'  python3 {sys.argv[0]} --list')
        sys.exit(1)

    query = args.query

    # 1. 列出模型
    print_separator("启动双模型分析")
    print(f"  查询: {query}")
    for key, mc in config["models"].items():
        print(f"  {key} ({mc['name']}): {mc['vars']}")

    # 2. 校验列名 + 运行所有模型
    runner = DualModelRunner(config)
    runner.validate_vars()
    all_results = runner.run_all()

    # 3. 关键词路由
    router = QueryRouter(config)
    keyword_choice, scores = router.route(query)
    print(f"\n  关键词得分: {scores}")

    # 4. Granger 置信度检查
    checker = ConfidenceChecker()
    final_choice, granger_results, reason = checker.check(all_results, keyword_choice)

    # 5. 打印所有模型的 Granger 结果
    sig_counts = {}
    for key, ctx in all_results.items():
        g = granger_results[key]
        name = config["models"][key]["name"]
        sig_paths = print_granger_summary(g, ctx.var_names, name)
        sig_counts[key] = len(sig_paths)

    # 6. 打印决策
    print_decision(config, scores, final_choice, reason, sig_counts)

    # 7. 输出最终报告
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

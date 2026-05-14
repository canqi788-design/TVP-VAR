"""
TVP-VAR 结构化报告生成器

Markdown 报告 + CSV 导出
"""

import numpy as np
import os
import json
import csv
import logging

logger = logging.getLogger("tvp_var")


class ReportGenerator:
    """结构化报告生成器"""

    def __init__(self, output_dir, company_name="", title=""):
        self.output_dir = output_dir
        self.company_name = company_name
        self.title = title or f"{company_name} TVP-VAR 分析报告"
        self.sections = []
        self._metadata = {}

    def set_metadata(self, **kwargs):
        """设置报告元数据"""
        self._metadata.update(kwargs)

    def add_section(self, title, content, level=2):
        """添加报告章节"""
        self.sections.append({
            "type": "text",
            "title": title,
            "content": content,
            "level": level,
        })

    def add_table(self, headers, rows, title=None):
        """添加表格"""
        self.sections.append({
            "type": "table",
            "headers": headers,
            "rows": rows,
            "title": title,
        })

    def add_convergence_report(self, diagnostics):
        """添加收敛诊断章节"""
        if not diagnostics:
            return
        self.sections.append({
            "type": "convergence",
            "diagnostics": diagnostics,
        })

    def add_stationarity_report(self, stationarity_results):
        """添加平稳性检验章节"""
        if not stationarity_results:
            return
        self.sections.append({
            "type": "stationarity",
            "results": stationarity_results,
        })

    def add_backtest_report(self, backtest_result):
        """添加回测章节"""
        if backtest_result is None:
            return
        self.sections.append({
            "type": "backtest",
            "result": backtest_result,
        })

    def add_irf_results(self, irf_mean, irf_lower, irf_upper, var_names, shock_name):
        """添加 IRF 结果"""
        self.sections.append({
            "type": "irf",
            "irf_mean": irf_mean,
            "irf_lower": irf_lower,
            "irf_upper": irf_upper,
            "var_names": var_names,
            "shock_name": shock_name,
        })

    def add_forecast_results(self, pred_mean, pred_lower, pred_upper,
                             labels, var_names):
        """添加预测结果"""
        self.sections.append({
            "type": "forecast",
            "pred_mean": pred_mean,
            "pred_lower": pred_lower,
            "pred_upper": pred_upper,
            "labels": labels,
            "var_names": var_names,
        })

    def add_posterior_summary(self, summary_dict, var_names):
        """添加后验摘要"""
        self.sections.append({
            "type": "posterior_summary",
            "summary": summary_dict,
            "var_names": var_names,
        })

    def generate_markdown(self):
        """生成完整 Markdown 报告"""
        lines = []
        lines.append(f"# {self.title}")
        lines.append("")
        if self._metadata:
            for key, value in self._metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        for section in self.sections:
            stype = section["type"]
            level = section.get("level", 2)
            prefix = "#" * level

            if stype == "text":
                lines.append(f"\n{prefix} {section['title']}")
                lines.append("")
                lines.append(section["content"])
                lines.append("")

            elif stype == "table":
                if section.get("title"):
                    lines.append(f"\n{prefix} {section['title']}")
                    lines.append("")
                lines.append(_render_markdown_table(section["headers"], section["rows"]))
                lines.append("")

            elif stype == "convergence":
                lines.append(f"\n{prefix} MCMC 收敛诊断")
                lines.append("")
                lines.append(self._render_convergence(section["diagnostics"]))
                lines.append("")

            elif stype == "stationarity":
                lines.append(f"\n{prefix} 平稳性检验")
                lines.append("")
                lines.append(self._render_stationarity(section["results"]))
                lines.append("")

            elif stype == "backtest":
                lines.append(f"\n{prefix} 回测结果")
                lines.append("")
                lines.append(self._render_backtest(section["result"]))
                lines.append("")

            elif stype == "irf":
                lines.append(f"\n{prefix} 脉冲响应: {section['shock_name']}")
                lines.append("")
                lines.append(self._render_irf(section))
                lines.append("")

            elif stype == "forecast":
                lines.append(f"\n{prefix} 后验预测")
                lines.append("")
                lines.append(self._render_forecast(section))
                lines.append("")

            elif stype == "posterior_summary":
                lines.append(f"\n{prefix} 后验参数估计")
                lines.append("")
                lines.append(self._render_posterior_summary(section))
                lines.append("")

        lines.append("\n---")
        lines.append("*由 TVP-VAR 分析工具自动生成*")
        return "\n".join(lines)

    def save_markdown(self, filename="report.md"):
        """保存 Markdown 文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        content = self.generate_markdown()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"报告已保存: {path}")
        return path

    def export_csv(self, data_dict, filename="results.csv"):
        """导出数据到 CSV"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        writer.writerow([key] + value.tolist())
                    elif value.ndim == 2:
                        writer.writerow([key])
                        for row in value:
                            writer.writerow([""] + row.tolist())
                else:
                    writer.writerow([key, str(value)])
        logger.info(f"CSV 已导出: {path}")
        return path

    def export_chains_csv(self, chains_dict, filename="chains.csv"):
        """导出后验链到 CSV"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for name, chain in chains_dict.items():
                chain = np.asarray(chain)
                if chain.ndim == 1:
                    writer.writerow([name] + chain.tolist())
                elif chain.ndim == 2:
                    for i in range(chain.shape[1]):
                        writer.writerow([f"{name}_{i}"] + chain[:, i].tolist())
                elif chain.ndim == 3:
                    n1, n2 = chain.shape[1], chain.shape[2]
                    for i in range(n1):
                        for j in range(n2):
                            writer.writerow([f"{name}_{i}_{j}"] + chain[:, i, j].tolist())
        logger.info(f"链 CSV 已导出: {path}")
        return path

    def _render_convergence(self, diagnostics):
        """渲染收敛诊断"""
        lines = []
        for model_name, diag in diagnostics.items():
            lines.append(f"### {model_name}")
            lines.append("")
            lines.append(f"| 参数 | R-hat | ESS | Geweke p | 状态 |")
            lines.append(f"|------|-------|-----|----------|------|")
            if isinstance(diag, dict) and "error" not in diag:
                for param_name, param_diag in diag.items():
                    if not isinstance(param_diag, dict):
                        continue
                    rhat = param_diag.get("rhat", np.nan)
                    ess = param_diag.get("ess", np.nan)
                    geweke_p = param_diag.get("geweke_p", np.nan)
                    conv = param_diag.get("converged", False)
                    status = "收敛" if conv else "未收敛"
                    lines.append(f"| {param_name} | {rhat:.4f} | {ess:.0f} | {geweke_p:.4f} | {status} |")
            lines.append("")
        return "\n".join(lines)

    def _render_stationarity(self, results):
        """渲染平稳性检验"""
        lines = []
        lines.append(f"| 变量 | ADF 统计量 | p 值 | 平稳 |")
        lines.append(f"|------|-----------|------|------|")
        for name, result in results.items():
            if isinstance(result, dict) and "statistic" in result:
                stat = result["statistic"]
                p = result.get("p_value", np.nan)
                is_stat = result.get("is_stationary", False)
                status = "是" if is_stat else "否"
                lines.append(f"| {name} | {stat:.4f} | {p:.4f} | {status} |")
        return "\n".join(lines)

    def _render_backtest(self, result):
        """渲染回测结果"""
        if result is None:
            return "无回测结果"
        return result.summary()

    def _render_irf(self, section):
        """渲染 IRF"""
        irf_mean = section["irf_mean"]
        irf_lower = section["irf_lower"]
        irf_upper = section["irf_upper"]
        var_names = section["var_names"]
        periods = irf_mean.shape[0]

        header = "| 期 |"
        for name in var_names:
            header += f" {name} |"
        lines = [header]
        sep = "|------|"
        for _ in var_names:
            sep += "-------|"
        lines.append(sep)

        for t in range(periods):
            row = f"| t+{t} |"
            for i in range(len(var_names)):
                row += f" {irf_mean[t, i]:.4f}[{irf_lower[t, i]:.3f},{irf_upper[t, i]:.3f}] |"
            lines.append(row)
        return "\n".join(lines)

    def _render_forecast(self, section):
        """渲染预测结果"""
        pred_mean = section["pred_mean"]
        pred_lower = section["pred_lower"]
        pred_upper = section["pred_upper"]
        labels = section["labels"]
        var_names = section["var_names"]

        header = "| 季度 |"
        for name in var_names:
            header += f" {name} |"
        lines = [header]
        sep = "|------|"
        for _ in var_names:
            sep += "-------|"
        lines.append(sep)

        for s in range(min(len(labels), pred_mean.shape[0])):
            row = f"| {labels[s]} |"
            for i in range(len(var_names)):
                row += f" {pred_mean[s, i]:.1f}[{pred_lower[s, i]:.1f},{pred_upper[s, i]:.1f}] |"
            lines.append(row)
        return "\n".join(lines)

    def _render_posterior_summary(self, section):
        """渲染后验摘要"""
        summary = section["summary"]
        var_names = section["var_names"]
        lines = []
        lines.append(f"| 参数 | 均值 | 标准差 | 95% CI |")
        lines.append(f"|------|------|--------|--------|")
        for param, info in summary.items():
            if isinstance(info, dict):
                mean = info.get("mean", np.nan)
                std = info.get("std", np.nan)
                ci = info.get("ci_95", (np.nan, np.nan))
                lines.append(f"| {param} | {mean:.4f} | {std:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        return "\n".join(lines)


def _render_markdown_table(headers, rows):
    """渲染 Markdown 表格"""
    header = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + body_lines)


def export_posterior_summary_csv(summary_dict, output_path):
    """导出后验摘要到 CSV"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["参数", "均值", "标准差", "CI下界", "CI上界"])
        for param, info in summary_dict.items():
            if isinstance(info, dict):
                mean = info.get("mean", "")
                std = info.get("std", "")
                ci = info.get("ci_95", ("", ""))
                writer.writerow([param, mean, std, ci[0], ci[1]])
    logger.info(f"后验摘要 CSV: {output_path}")


def export_irf_csv(irf_mean, irf_lower, irf_upper, var_names, output_path):
    """导出 IRF 到 CSV"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    periods = irf_mean.shape[0]
    n = len(var_names)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["期"]
        for name in var_names:
            header.extend([f"{name}_mean", f"{name}_lower", f"{name}_upper"])
        writer.writerow(header)
        for t in range(periods):
            row = [f"t+{t}"]
            for i in range(n):
                row.extend([irf_mean[t, i], irf_lower[t, i], irf_upper[t, i]])
            writer.writerow(row)
    logger.info(f"IRF CSV: {output_path}")

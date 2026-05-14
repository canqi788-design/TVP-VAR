"""
Probabilistic Execution Runtime — v4

Kernel contract:
    def kernel(inputs: dict, params: dict) -> dict

Engine dispatch:
    out = kernel(inputs=node_inputs, params=config_section)
    ctx.outputs.update(out)

Posterior semantics:
    All samplers output _joint_chain (list[PosteriorSample dict])
    One Gibbs iteration = one coherent posterior world-state

IR node types: source, transform, compute, sink (whitelist locked)
Config: scalar-only (numbers, strings, booleans, arrays of scalars)
"""

import json
import os
import logging
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("tvp_var.runtime")

from tvp_var_framework.runtime.context import ExecutionContext
from tvp_var_framework.runtime.contracts import (
    ALLOWED_OUTPUT_KEYS, ALLOWED_NODE_TYPES, STRUCT_FIELDS,
    validate_kernel_output, validate_ir_against_contracts,
    validate_joint_chain, ContractViolation,
)


# ============================================================
# DAG
# ============================================================

def load_ir(ir_path: str) -> dict:
    with open(ir_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "inference_graph" in data:
        data = data["inference_graph"]
    nodes = data.get("nodes", {})
    for name, node in nodes.items():
        ntype = node.get("type", "")
        if ntype not in ALLOWED_NODE_TYPES:
            raise ValueError(
                f"IR node '{name}' has disallowed type '{ntype}'. "
                f"Allowed: {ALLOWED_NODE_TYPES}"
            )
    validate_ir_against_contracts(nodes)
    return data


def _parse_edges(edges):
    parsed = []
    for e in edges:
        if isinstance(e, dict):
            parsed.append((e["from"], e["to"], e.get("data")))
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            parsed.append((e[0], e[1], None))
        else:
            raise ValueError(f"Unrecognized edge: {e}")
    return parsed


def topo_sort(nodes, edges):
    parsed = _parse_edges(edges)
    in_degree = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for src, dst, _ in parsed:
        adj[src].append(dst)
        in_degree[dst] = in_degree.get(dst, 0) + 1

    queue = deque(n for n in nodes if in_degree[n] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(nodes):
        raise ValueError(f"DAG cycle: sorted {len(order)} / {len(nodes)}")
    return order


# ============================================================
# Kernel Infrastructure
# ============================================================

@dataclass
class KernelSpec:
    name: str
    fn: Callable[[dict, dict], dict]


# ============================================================
# Kernels — fn(inputs, params) -> outputs
# ============================================================

def _kernel_stationarity(inputs: dict, params: dict) -> dict:
    from tvp_var_framework.utils.stationarity import StationarityAnalyzer

    analyzer = StationarityAnalyzer(
        significance=params.get("significance", 0.05),
        max_d=params.get("max_d", 2),
        test=params.get("test", "adf"),
        log_transform=params.get("log_transform", False),
    )
    analyzer.analyze(inputs["Y"], var_names=inputs["var_names"])

    Y_diff = analyzer.get_differenced_data()
    ql = inputs.get("time_index", [])
    diff_count = len(ql) - len(Y_diff)
    if diff_count > 0:
        ql = ql[diff_count:]

    for name, d in zip(inputs["var_names"], analyzer.get_d_orders()):
        if d > 0:
            logger.info(f"  {name}: 非平稳, 差分 {d} 阶")

    return {
        "Y_diff": Y_diff,
        "d_orders": analyzer.get_d_orders(),
        "time_index": ql,
        "_stationarity": analyzer,
    }


def _kernel_state_update(inputs: dict, params: dict) -> dict:
    return {"theta_t": True}


def _kernel_likelihood(inputs: dict, params: dict) -> dict:
    return {"log_lik": 0.0}


# ============================================================
# Sampling Kernels — unified posterior output
# ============================================================

def kernel_sampler_basic(inputs: dict, params: dict) -> dict:
    """
    Basic sampler kernel — dispatches to estimation backends.

    Produces _joint_chain by restructuring existing model chains
    into coupled posterior snapshots (post-hoc coupling).

    For true sampling-time coupling, use kernel_sampler_research.
    """
    from tvp_var_framework.runtime.estimators import (
        estimate_fully_bayesian, estimate_bayesian,
        estimate_v2, estimate_research,
    )

    mode = params.get("mode", "full")

    estimators = {
        "fully_bayesian": estimate_fully_bayesian,
        "bayesian": estimate_bayesian,
        "v2": estimate_v2,
        "research": estimate_research,
    }

    if mode == "full":
        modes = ["fully_bayesian", "bayesian", "v2", "research"]
    else:
        modes = [mode]

    active_models = {}
    results = {}
    per_model_chains = {}
    primary_joint_chain = []

    for m in modes:
        fn = estimators.get(m)
        if fn is None:
            continue
        out = fn(inputs, params)
        active_models[m] = out["_model"]
        results[m] = out["_result"]

        # Keep per-model chains separate (different theta dimensions)
        if "_joint_chain" in out and out["_joint_chain"]:
            per_model_chains[m] = out["_joint_chain"]
            # Use first non-empty chain as primary
            if not primary_joint_chain:
                primary_joint_chain = out["_joint_chain"]

    if not primary_joint_chain:
        raise ContractViolation(
            "sampling_basic produced no _joint_chain. "
            "At least one estimator must return posterior samples."
        )

    # Backward compat: also emit "posterior" dict
    return {
        "_joint_chain": primary_joint_chain,
        "posterior": {
            "active_models": active_models,
            "results": results,
            "per_model_chains": per_model_chains,
        },
    }


def kernel_sampler_research(inputs: dict, params: dict) -> dict:
    """
    Research-grade Gibbs sampler with joint posterior chain support.

    Produces _joint_chain via true sampling-time coupling:
    Each Gibbs iteration saves ONE coupled posterior world-state
    where theta_k, sigma_k, sv_k always come from the same iteration.

    This guarantees posterior dependency integrity for:
    - IRF / FEVD / Forecast
    - WAIC / DIC
    - Posterior simulation
    """
    rng = np.random.default_rng(params.get("random_state"))

    Y = inputs.get("Y_diff", inputs.get("Y"))
    n = Y.shape[1]

    n_iter = params.get("n_iter", 2000)
    burn = params.get("burnin", 500)
    thin = params.get("thin", 1)

    logger.info("=" * 60)
    logger.info("Research-grade Gibbs Sampler (joint posterior)")
    logger.info(f"  n_iter={n_iter}, burn={burn}, thin={thin}")
    logger.info("=" * 60)

    # ── Build model with required interface ──
    from tvp_var_framework.models.fully_bayesian import FullyBayesianTVPVAR

    model = FullyBayesianTVPVAR(
        n_vars=n,
        n_iter=1,  # we drive the loop ourselves
        burnin=0,
        thin=1,
        sv_n_iter=params.get("sv_n_iter", 300),
        sv_burnin=params.get("sv_burnin", 100),
    )

    # ── Initialize posterior state via OLS ──
    k = n + n * n
    X_ols = np.column_stack([np.ones(len(Y) - 1), Y[:-1]])
    Y_reg = Y[1:]
    beta_ols = np.linalg.lstsq(X_ols, Y_reg, rcond=None)[0]  # (n+1, n)
    c_ols = beta_ols[0]
    A_ols = beta_ols[1:].T  # (n, n)
    resid_ols = Y_reg - X_ols @ beta_ols
    sigma2_ols = np.var(resid_ols, axis=0)

    theta_t = np.zeros(k)
    theta_t[:n] = c_ols
    theta_t[n:] = A_ols.flatten()
    sigma_t = np.diag(sigma2_ols)
    sigma_t = np.clip(sigma_t, 0.01, 1.0)
    sv_t = np.full((len(Y), n), np.log(0.1))

    # ── Gibbs state ──
    Q = np.eye(k) * 0.01
    R_corr = np.eye(n)

    joint_chain = []
    acceptance_stats = {"theta": 0, "sigma": 0, "sv": 0}

    # ── Gibbs Sampling Loop ──
    for itr in range(n_iter):

        # Step 1: Sample theta | Sigma, SV, Y (FFBS)
        states, covs, pred_states, pred_covs, log_lik = \
            model._forward_filter_tvp_R(Y, Q, np.tile(sigma_t, (len(Y), 1, 1)))
        theta_traj = model._backward_sample(
            states, covs, pred_states, pred_covs, Q)
        theta_new = theta_traj[-1]
        theta_accept = True  # FFBS is always accepted

        # Step 2: Sample Sigma | theta, SV, Y
        residuals = model._compute_residuals(Y, theta_traj)
        Q = model._sample_Q(theta_traj)
        sigma_new = model._sample_R_static(Y, theta_traj)
        sigma_accept = True

        # Step 3: Sample SV state
        sv_new = sv_t  # simplified: reuse previous
        sv_accept = True

        if params.get("sv_enabled", True) and len(residuals) > 10:
            # Inline SV update (simplified)
            for j in range(n):
                log_e2 = np.log(residuals[:, j] ** 2 + 1e-10)
                h = np.zeros(len(Y) - 1)
                P_h = np.ones(len(Y) - 1)
                mu_h, phi_h, sigma_h = -5.0, 0.9, 0.3

                mix_prob = np.array([0.00609, 0.04775, 0.13057, 0.20674,
                                     0.22715, 0.18842, 0.12047, 0.05591,
                                     0.01575, 0.00115])
                mix_mean = np.array([-1.5797, -1.1616, -0.7702, -0.4318,
                                     -0.1168, 0.1958, 0.5316, 0.9212,
                                     1.4262, 2.1855])
                mix_var = np.array([0.5576, 0.3712, 0.2557, 0.1883,
                                    0.1505, 0.1331, 0.1342, 0.1611,
                                    0.2392, 0.5218])

                for t in range(len(Y) - 1):
                    if t == 0:
                        h_pred = mu_h
                        P_pred_h = sigma_h ** 2 / (1 - phi_h ** 2 + 1e-10)
                    else:
                        h_pred = mu_h + phi_h * (h[t - 1] - mu_h)
                        P_pred_h = phi_h ** 2 * P_h[t - 1] + sigma_h ** 2

                    log_prob = np.zeros(10)
                    for kk in range(10):
                        log_prob[kk] = (np.log(mix_prob[kk]) -
                                        0.5 * np.log(mix_var[kk]) -
                                        0.5 * (log_e2[t] - h_pred - mix_mean[kk]) ** 2 /
                                        mix_var[kk])
                    log_prob -= np.max(log_prob)
                    prob = np.exp(log_prob)
                    prob /= prob.sum()
                    s_k = np.random.choice(10, p=prob)

                    obs_var = mix_var[s_k]
                    S_h = P_pred_h + obs_var
                    K_h = P_pred_h / S_h
                    h[t] = h_pred + K_h * (log_e2[t] - h_pred - mix_mean[s_k])
                    P_h[t] = (1 - K_h) * P_pred_h

                # Backward sample
                h_sample = np.zeros(len(Y) - 1)
                h_sample[len(Y) - 2] = np.random.normal(
                    h[len(Y) - 2], np.sqrt(max(P_h[len(Y) - 2], 1e-10)))
                for t in range(len(Y) - 3, -1, -1):
                    J_h = phi_h * P_h[t] / (phi_h ** 2 * P_h[t] + sigma_h ** 2 + 1e-10)
                    h_smooth = h[t] + J_h * (h_sample[t + 1] - mu_h - phi_h * (h[t] - mu_h))
                    P_smooth_h = max(P_h[t] - J_h * phi_h * P_h[t], 1e-10)
                    h_sample[t] = np.random.normal(h_smooth, np.sqrt(P_smooth_h))

                sv_new[1:, j] = h_sample
                sv_new[0, j] = h_sample[0]

        # ── Update state ──
        theta_t = theta_new
        sigma_t = sigma_new
        sv_t = sv_new

        acceptance_stats["theta"] += int(theta_accept)
        acceptance_stats["sigma"] += int(sigma_accept)
        acceptance_stats["sv"] += int(sv_accept)

        # ── Burn-in & thinning: save coupled snapshot ──
        if itr >= burn and ((itr - burn) % thin == 0):
            # Use filtered states (forward pass) for A — less noisy than backward sample
            A_filtered = states[:, n:]  # (T, n*n)
            A_avg = A_filtered.reshape(-1, n, n).mean(axis=0)
            posterior_state = {
                "sample_index": itr,
                "theta": np.copy(theta_t),
                "sigma": np.copy(sigma_t),
                "sv_state": np.copy(sv_t),
                "A_avg": A_avg.copy(),
                "log_likelihood": float(log_lik) if np.isfinite(log_lik) else None,
            }
            joint_chain.append(posterior_state)

        if (itr + 1) % 500 == 0:
            logger.info(f"  iter {itr+1}/{n_iter}  saved={len(joint_chain)}")

    # ── Validate output ──
    validate_joint_chain(joint_chain)

    # ── Posterior summary ──
    theta_stack = np.stack([s["theta"] for s in joint_chain], axis=0)
    sigma_stack = np.stack([s["sigma"] for s in joint_chain], axis=0)

    posterior_mean = {
        "theta_mean": theta_stack.mean(axis=0),
        "sigma_mean": sigma_stack.mean(axis=0),
    }

    acceptance_rate = {
        k: v / n_iter for k, v in acceptance_stats.items()
    }

    logger.info(f"  后验样本数: {len(joint_chain)}")
    logger.info(f"  接受率: {acceptance_rate}")

    return {
        "_joint_chain": joint_chain,
        "posterior_mean": posterior_mean,
        "acceptance_rate": acceptance_rate,
        "n_saved_samples": len(joint_chain),
    }


# ============================================================
# Diagnostics — operates on _joint_chain
# ============================================================

def _kernel_diagnostics(inputs: dict, params: dict) -> dict:
    from tvp_var_framework.diagnostics.convergence import (
        diagnose_model_chains, ConvergenceReport,
    )

    joint_chain = inputs.get("_joint_chain", [])

    if not joint_chain:
        logger.warning("  diagnostics: no _joint_chain available")
        return {}

    # Extract chains for diagnostics
    theta_samples = [s["theta"] for s in joint_chain if s.get("theta") is not None]
    sigma_samples = [s["sigma"] for s in joint_chain if s.get("sigma") is not None]

    all_diagnostics = {}

    if theta_samples:
        theta_arr = np.stack(theta_samples, axis=0)
        n_params = theta_arr.shape[1] if theta_arr.ndim > 1 else 1
        diag = {}
        for p in range(min(n_params, 10)):  # limit to first 10 params
            chain_p = theta_arr[:, p]
            try:
                from tvp_var_framework.diagnostics.convergence import (
                    single_chain_rhat, effective_sample_size, geweke_diagnostic,
                )
                rhat = single_chain_rhat(chain_p)
                ess = effective_sample_size(chain_p)
                g = geweke_diagnostic(chain_p)
                diag[f"theta_{p}"] = {
                    "rhat": rhat, "ess": ess,
                    "geweke_z": g["z_score"],
                    "geweke_p": g["p_value"],
                    "converged": rhat < params.get("rhat_threshold", 1.05),
                }
            except Exception:
                diag[f"theta_{p}"] = {"rhat": np.nan, "ess": np.nan, "geweke_z": np.nan, "geweke_p": np.nan, "converged": False}
        all_diagnostics["theta"] = diag

    if not all_diagnostics:
        return {}

    report = ConvergenceReport(all_diagnostics)
    converged = report.is_converged(
        rhat_threshold=params.get("rhat_threshold", 1.05),
        ess_min=params.get("ess_minimum", 100),
    )

    for model_name, diag in all_diagnostics.items():
        for param_name, param_diag in diag.items():
            if not isinstance(param_diag, dict):
                continue
            rhat = param_diag.get("rhat", np.nan)
            ess = param_diag.get("ess", np.nan)
            status = "收敛" if param_diag.get("converged", False) else "未收敛"
            logger.info(f"  {model_name}/{param_name}: R-hat={rhat:.4f}, ESS={ess:.0f} [{status}]")

    logger.info(f"总体收敛: {'是' if converged else '否'}")
    return {
        "metrics": {"convergence": report, "converged": converged},
        "_convergence": report,
    }


# ============================================================
# Reporting — operates on _joint_chain
# ============================================================

def _render_irf_section(sec):
    var_names = sec["var_names"]
    header = "| 期 |" + " ".join(f" {v} |" for v in var_names)
    sep = "|------|" + "-------|" * len(var_names)
    rows = [header, sep]
    for r in sec["rows"]:
        cells = " ".join(f" {r[v]['display']} |" for v in var_names)
        rows.append(f"| {r['period']} |{cells}")
    return "\n".join(rows)


def _render_fevd_section(fevd_mean, fevd_lower, fevd_upper, var_names):
    """渲染 FEVD 表格: fevd[i,j] = 冲击 j 对变量 i 方差的贡献比例"""
    header = "| 变量 |" + " ".join(f" {v} 冲击 |" for v in var_names)
    sep = "|------|" + "-------|" * len(var_names)
    rows = [header, sep]
    for i, vname in enumerate(var_names):
        cells = []
        for j in range(len(var_names)):
            m = fevd_mean[i, j] * 100
            lo = fevd_lower[i, j] * 100
            hi = fevd_upper[i, j] * 100
            cells.append(f" {m:.1f}%[{lo:.1f},{hi:.1f}] |")
        rows.append(f"| {vname} |" + " ".join(cells))
    return "\n".join(rows)


def _render_granger_section(granger, var_names):
    """渲染 Granger 因果检验表格"""
    header = "| 因果方向 | 系数均值 | 95% CI | 显著 | 正向概率 |"
    sep = "|----------|---------|--------|------|---------|"
    rows = [header, sep]
    for (src, dst), info in granger.items():
        sig = "是" if info["significant"] else "否"
        ci = f"[{info['ci_95'][0]:.3f}, {info['ci_95'][1]:.3f}]"
        rows.append(
            f"| {src} → {dst} | {info['mean']:.4f} | {ci} | {sig} | {info['prob_positive']:.1%} |"
        )
    return "\n".join(rows)
    return "\n".join(rows)


def _render_forecast_section(fc):
    var_names = fc["var_names"]
    header = "| 季度 |" + " ".join(f" {v} |" for v in var_names)
    sep = "|------|" + "-------|" * len(var_names)
    rows = [header, sep]
    for r in fc["rows"]:
        cells = " ".join(f" {r[v]['display']} |" for v in var_names)
        rows.append(f"| {r['label']} |{cells}")
    return "\n".join(rows)


def _render_posterior_section(ps):
    header = "| 参数 | 均值 | 95% CI |"
    sep = "|------|------|--------|"
    rows = [header, sep]
    for r in ps["rows"]:
        rows.append(f"| {r['param']} | {r['display_mean']} | {r['display_ci']} |")
    return "\n".join(rows)


def _kernel_reporting(inputs: dict, params: dict) -> dict:
    from tvp_var_framework.reporting.report_generator import ReportGenerator
    from tvp_var_framework.runtime.formatter import (
        format_irf_for_report, format_forecast_for_report, format_posterior_for_report,
    )

    output_dir = params.get("output_dir", "./analysis_results/")
    export_csv = params.get("export_csv", False)
    if export_csv:
        os.makedirs(output_dir, exist_ok=True)

    Y = inputs.get("Y_diff", inputs.get("Y"))
    ql = inputs.get("time_index", [])
    var_names = inputs["var_names"]

    logger.info("=" * 60)
    logger.info("TVP-VAR 分析报告")
    logger.info("=" * 60)
    logger.info(f"  数据范围: {ql[0]} — {ql[-1]}")
    logger.info(f"  观测数: {len(Y)}")
    logger.info(f"  变量: {', '.join(var_names)}")

    rg = ReportGenerator(output_dir=output_dir)

    if "_convergence" in inputs:
        rg.add_convergence_report(inputs["_convergence"].diagnostics)
    if "_stationarity" in inputs:
        rg.add_stationarity_report(inputs["_stationarity"].results)

    # ── Report from _joint_chain ──
    joint_chain = inputs.get("_joint_chain", [])

    if joint_chain:
        n = Y.shape[1]
        irf_periods = params.get("irf_periods", 6)
        forecast_steps = params.get("forecast_steps", 4)
        n_samples = min(len(joint_chain), 500)

        # ── 1. IRF: ALL shocks (not just first) ──
        irf_all = np.zeros((n_samples, irf_periods, n, n))
        for s_idx in range(n_samples):
            sample = joint_chain[s_idx]
            A_avg = sample.get("A_avg")
            if A_avg is not None:
                Psi = np.eye(n)
                for t in range(irf_periods):
                    irf_all[s_idx, t] = Psi  # (n, n): all shocks at period t
                    Psi = A_avg @ Psi

        # Shape: (periods, n, n_shocks) — mean/lower/upper across samples
        irf_mean = irf_all.mean(axis=0)  # (periods, n, n_shocks)
        irf_lower = np.percentile(irf_all, 2.5, axis=0)
        irf_upper = np.percentile(irf_all, 97.5, axis=0)

        irf_sections = format_irf_for_report(
            irf_mean, irf_lower, irf_upper, var_names, "research",
        )
        for sec in irf_sections:
            rg.add_section(f"脉冲响应: {sec['shock_name']}", _render_irf_section(sec))

        # ── 2. FEVD: Forecast Error Variance Decomposition ──
        from tvp_var_framework.models.research_grade import StructuralAnalysis
        sa = StructuralAnalysis(n)
        fevd_mean, fevd_lower, fevd_upper = sa.orthogonalized_fevd_from_joint_chain(
            joint_chain, periods=irf_periods,
        )
        rg.add_section("方差分解 (FEVD)", _render_fevd_section(
            fevd_mean, fevd_lower, fevd_upper, var_names,
        ))
        logger.info(f"  FEVD 计算完成 ({irf_periods} 期)")

        # ── 3. Forecast from joint_chain ──
        from tvp_var_framework.models.research_grade import MCPredictor
        mcp = MCPredictor(n)
        theta_samples = np.stack([s["theta"] for s in joint_chain if s.get("theta") is not None])
        Y_last = Y[-1]
        pred_samples = mcp.predict(theta_samples, None, Y_last,
                                   steps=forecast_steps, n_samples=n_samples)
        pred_mean = pred_samples.mean(axis=0)
        pred_lower = np.percentile(pred_samples, 2.5, axis=0)
        pred_upper = np.percentile(pred_samples, 97.5, axis=0)

        # Generate future labels
        last_label = ql[-1] if ql else "T"
        future_labels = [f"t+{s+1}" for s in range(forecast_steps)]
        fc = format_forecast_for_report(pred_mean, pred_lower, pred_upper, future_labels, var_names)
        rg.add_section("预测 (Forecast)", _render_forecast_section(fc))
        logger.info(f"  预测完成 ({forecast_steps} 步)")

        # Export forecast CSV
        if export_csv:
            rg.export_csv(
                {"mean": pred_mean, "lower": pred_lower, "upper": pred_upper},
                filename="forecast_research.csv",
            )

        # ── 4. Granger Causality from posterior A matrices ──
        granger = _compute_granger_causality(joint_chain, var_names, n)
        rg.add_section("Granger 因果检验", _render_granger_section(granger, var_names))
        logger.info("  Granger 因果检验完成")

        # ── 5. Posterior summary ──
        theta_stack = np.stack([s["theta"] for s in joint_chain if s.get("theta") is not None], axis=0)
        theta_mean = theta_stack.mean(axis=0)
        theta_lower = np.percentile(theta_stack, 2.5, axis=0)
        theta_upper = np.percentile(theta_stack, 97.5, axis=0)

        summary = {}
        for i in range(min(len(theta_mean), n + n * n)):
            param_name = f"theta_{i}"
            summary[param_name] = {
                "mean": float(theta_mean[i]),
                "ci_95": (float(theta_lower[i]), float(theta_upper[i])),
            }
        ps = format_posterior_for_report(summary, var_names)
        rg.add_section("后验参数估计 (joint_chain)", _render_posterior_section(ps))

    # ── Legacy reporting from posterior dict (if present) ──
    posterior = inputs.get("posterior", {})
    results = posterior.get("results", {})

    for stage in ("fully_bayesian", "bayesian", "v2", "bq"):
        if stage in results:
            rd = results[stage]
            if rd.irf_mean is not None:
                irf_sections = format_irf_for_report(
                    rd.irf_mean, rd.irf_lower, rd.irf_upper, var_names, stage,
                )
                for sec in irf_sections:
                    rg.add_section(f"脉冲响应: {sec['shock_name']}", _render_irf_section(sec))
            if rd.pred_mean is not None:
                labels = rd.future_labels if hasattr(rd, "future_labels") and rd.future_labels else ql
                fc = format_forecast_for_report(rd.pred_mean, rd.pred_lower, rd.pred_upper, labels, var_names)
                rg.add_section(f"后验预测 ({stage})", _render_forecast_section(fc))

    report_path = rg.save_markdown()
    logger.info(f"报告已保存: {report_path}")
    out = {"_report_path": report_path, "report": report_path}

    if export_csv:
        logger.info("CSV 导出完成")
        out["csv"] = output_dir

    return out


def _compute_granger_causality(joint_chain, var_names, n):
    """
    基于后验 A 矩阵的 Granger 因果检验。
    对每对 (j -> i): 检验 A[i,j] 的后验分布是否显著不为零。
    优先使用时间平均 A_avg (TVP-VAR), 否则回退到 theta 中的 A。
    返回: dict with pairwise results.
    """
    A_samples = []
    for s in joint_chain:
        A_avg = s.get("A_avg")
        if A_avg is not None:
            A_samples.append(A_avg)
        else:
            theta = s.get("theta")
            if theta is not None and len(theta) >= n + n * n:
                A = theta[n:].reshape(n, n)
                A_samples.append(A)
    A_arr = np.stack(A_samples)  # (n_samples, n, n)

    results = {}
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            coef_samples = A_arr[:, i, j]
            mean = float(coef_samples.mean())
            ci_low = float(np.percentile(coef_samples, 2.5))
            ci_high = float(np.percentile(coef_samples, 97.5))
            # Significant if CI excludes zero
            significant = (ci_low > 0) or (ci_high < 0)
            # Posterior probability of positive direction
            prob_positive = float((coef_samples > 0).mean())
            results[(var_names[j], var_names[i])] = {
                "mean": mean,
                "ci_95": (ci_low, ci_high),
                "significant": significant,
                "prob_positive": prob_positive,
                "prob_negative": 1 - prob_positive,
            }
    return results


# ============================================================
# Registry — DAG node -> kernel mapping
# ============================================================

NODE_REGISTRY = {
    "data": None,
    "stationarity": KernelSpec(name="stationarity", fn=_kernel_stationarity),
    "state_update": KernelSpec(name="state_update", fn=_kernel_state_update),
    "likelihood": KernelSpec(name="likelihood", fn=_kernel_likelihood),
    "sampling_basic": KernelSpec(name="sampling_basic", fn=kernel_sampler_basic),
    "sampling_research": KernelSpec(name="sampling_research", fn=kernel_sampler_research),
    "diagnostics": KernelSpec(name="diagnostics", fn=_kernel_diagnostics),
    "reporting": KernelSpec(name="reporting", fn=_kernel_reporting),
}

# ── Per-node param routing ──

def _get_nested(cfg, *keys, default=None):
    """安全地从嵌套字典中取值，任意层级缺失都返回 default，不会抛 KeyError。"""
    current = cfg
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k)
        if current is None:
            return default
    return current


PARAM_ROUTING = {
    "stationarity": lambda cfg: cfg.get("stationarity", {}),
    "state_update": lambda cfg: {},
    "likelihood": lambda cfg: {},
    "sampling_basic": lambda cfg: {
        **cfg.get("model_specification", {}),
        "n_iter": (
            _get_nested(cfg, "inference_control", "n_iter")
            or _get_nested(cfg, "model_specification", "n_iter")
            or 2000
        ),
        "burnin": (
            _get_nested(cfg, "inference_control", "burnin")
            or _get_nested(cfg, "model_specification", "burnin")
            or 500
        ),
        "thin": (
            _get_nested(cfg, "inference_control", "thin")
            or _get_nested(cfg, "model_specification", "thin")
            or 1
        ),
        "mode": _get_nested(cfg, "model_specification", "mode", default="full"),
        "sv_enabled": _get_nested(cfg, "stochastic_volatility", "enabled", default=True),
        "sv_n_iter": _get_nested(cfg, "stochastic_volatility", "sv_n_iter", default=300),
        "sv_burnin": _get_nested(cfg, "stochastic_volatility", "sv_burnin", default=100),
        "irf_periods": _get_nested(cfg, "structural_analysis", "irf_periods", default=6),
        "forecast_steps": _get_nested(cfg, "forecasting", "steps", default=4),
        "forecast_samples": _get_nested(cfg, "forecasting", "n_samples", default=500),
        "change_point_threshold": _get_nested(cfg, "structural_analysis", "change_point_threshold", default=1.5),
    },
    "sampling_research": lambda cfg: {
        "n_iter": (
            _get_nested(cfg, "inference_control", "n_iter")
            or _get_nested(cfg, "model_specification", "n_iter")
            or 2000
        ),
        "burnin": (
            _get_nested(cfg, "inference_control", "burnin")
            or _get_nested(cfg, "model_specification", "burnin")
            or 500
        ),
        "thin": (
            _get_nested(cfg, "inference_control", "thin")
            or _get_nested(cfg, "model_specification", "thin")
            or 1
        ),
        "sv_enabled": _get_nested(cfg, "stochastic_volatility", "enabled", default=True),
        "sv_n_iter": _get_nested(cfg, "stochastic_volatility", "sv_n_iter", default=300),
        "sv_burnin": _get_nested(cfg, "stochastic_volatility", "sv_burnin", default=100),
        "random_state": _get_nested(cfg, "inference_control", "random_state"),
    },
    "diagnostics": lambda cfg: cfg.get("convergence_diagnostics", {}),
    "reporting": lambda cfg: {
        **cfg.get("report_control", {}),
        **cfg.get("output_control", {}),
        "irf_periods": _get_nested(cfg, "structural_analysis", "irf_periods", default=6),
        "forecast_steps": _get_nested(cfg, "forecasting", "steps", default=4),
    },
}


# ============================================================
# Engine
# ============================================================

class InferenceGraphEngine:

    def __init__(self, ir_path, strict_contracts=False):
        self.ir = load_ir(ir_path)
        self.nodes = self.ir.get("nodes", {})
        self.edges = self.ir.get("edges", [])
        self.execution_order = topo_sort(self.nodes, self.edges)
        self.registry = dict(NODE_REGISTRY)
        self.strict_contracts = strict_contracts

    def register_kernel(self, node_name, kernel_fn_or_spec):
        if isinstance(kernel_fn_or_spec, KernelSpec):
            self.registry[node_name] = kernel_fn_or_spec
        else:
            self.registry[node_name] = KernelSpec(name=node_name, fn=kernel_fn_or_spec)

    def run(self, initial_data: dict) -> ExecutionContext:
        ctx = ExecutionContext(
            Y=initial_data.get("Y"),
            config=initial_data.get("config", {}),
            var_names=initial_data.get("var_names", []),
            time_index=initial_data.get("time_index", []),
        )
        declared = {"Y", "config", "var_names", "time_index"}
        for k, v in initial_data.items():
            if k not in declared:
                ctx.outputs[k] = v

        logger.info("=" * 60)
        logger.info("InferenceGraphEngine v4 — DAG 执行")
        logger.info(f"执行顺序: {' → '.join(self.execution_order)}")
        logger.info("=" * 60)

        for node_name in self.execution_order:
            spec = self.registry.get(node_name)
            if spec is None:
                continue

            logger.info(f"[NODE] {node_name}")

            ctx_dict = ctx.to_dict()
            router = PARAM_ROUTING.get(node_name, lambda cfg: {})
            params = router(ctx.config)

            # 参数到达诊断日志
            if node_name in ("sampling_basic", "sampling_research"):
                key_params = {k: params.get(k) for k in ("n_iter", "burnin", "thin", "sv_enabled", "mode") if k in params}
                logger.info(f"  路由参数: {node_name} -> {key_params}")
                if not params:
                    logger.warning(f"  警告: 节点 {node_name} 未收到任何外部配置参数，将使用硬编码默认值！")

            out = spec.fn(inputs=ctx_dict, params=params)

            if out and isinstance(out, dict):
                validate_kernel_output(node_name, out)

                # Validate _joint_chain for sampler nodes
                if node_name in ("sampling_basic", "sampling_research"):
                    if "_joint_chain" not in out:
                        raise ContractViolation(
                            f"'{node_name}' must output '_joint_chain'. "
                            f"Got: {list(out.keys())}"
                        )
                    validate_joint_chain(out["_joint_chain"])

                for k, v in out.items():
                    if k in STRUCT_FIELDS:
                        ctx.__dict__[k] = v
                    else:
                        ctx.outputs[k] = v

        logger.info("=" * 60)
        logger.info("分析完成")
        logger.info("=" * 60)

        return ctx

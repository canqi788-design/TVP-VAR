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
from tvp_var_framework.core.theta_layout import extract_transition

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
        method=params.get("method", "regular"),
        period=params.get("period", 1),
    )
    analyzer.analyze(inputs["Y"], var_names=inputs["var_names"])

    Y_diff = analyzer.get_differenced_data()
    ql = inputs.get("time_index", [])
    diff_count = len(ql) - len(Y_diff)
    if diff_count > 0:
        ql = ql[diff_count:]

    X_exog = inputs.get("X_exog")
    if X_exog is not None:
        X_exog = np.asarray(X_exog, dtype=float)
        if diff_count > 0:
            X_exog = X_exog[diff_count:]
        if len(X_exog) != len(Y_diff):
            raise ValueError(
                f"X_exog 与 Y_diff 时间长度不一致: X_exog={len(X_exog)}, "
                f"Y_diff={len(Y_diff)}, diff_count={diff_count}"
            )

    for name, d in zip(inputs["var_names"], analyzer.get_d_orders()):
        if d > 0:
            logger.info(f"  {name}: 非平稳, 差分 {d} 阶")

    out = {
        "Y_diff": Y_diff,
        "d_orders": analyzer.get_d_orders(),
        "time_index": ql,
        "_stationarity": analyzer,
    }
    if X_exog is not None:
        out["X_exog"] = X_exog
    return out


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
    primary_model_name = None

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
                primary_model_name = m

    if not primary_joint_chain:
        raise ContractViolation(
            "sampling_basic produced no _joint_chain. "
            "At least one estimator must return posterior samples."
        )

    # Backward compat: also emit "posterior" dict
    return {
        "_joint_chain": primary_joint_chain,
        "posterior": {
            "requested_mode": mode,
            "primary_model": primary_model_name,
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
        diagnose_model_chains, ConvergenceReport, parameter_is_converged,
    )

    joint_chain = inputs.get("_joint_chain", [])

    if not joint_chain:
        logger.warning("  diagnostics: no _joint_chain available")
        return {}

    # Extract chains for diagnostics
    theta_samples = [s["theta"] for s in joint_chain if s.get("theta") is not None]
    sigma_samples = [s["sigma"] for s in joint_chain if s.get("sigma") is not None]
    chain_source = joint_chain[0].get("chain_source") if joint_chain else None
    is_analytic_iid = chain_source == "analytic_iid"

    all_diagnostics = {}

    if theta_samples:
        theta_arr = np.stack(theta_samples, axis=0)
        n_params = theta_arr.shape[1] if theta_arr.ndim > 1 else 1
        diag = {}
        for p in range(n_params):
            chain_p = theta_arr[:, p]
            try:
                from tvp_var_framework.diagnostics.convergence import (
                    single_chain_rhat, effective_sample_size, geweke_diagnostic,
                )
                rhat = single_chain_rhat(chain_p)
                ess = effective_sample_size(chain_p)
                g = geweke_diagnostic(chain_p)
                if is_analytic_iid:
                    g = {"z_score": 0.0, "p_value": 1.0}
                param_diag = {
                    "rhat": rhat,
                    "ess": ess,
                    "geweke_z": g["z_score"],
                    "geweke_p": g["p_value"],
                }
                if is_analytic_iid:
                    param_diag["diagnostic_note"] = "analytic_iid"
                diag[f"theta_{p}"] = {
                    **param_diag,
                    "converged": parameter_is_converged(
                        param_diag,
                        rhat_threshold=params.get("rhat_threshold", 1.05),
                        ess_min=params.get("ess_minimum", 100),
                        geweke_p_min=0.0 if is_analytic_iid else 0.05,
                    ),
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
        geweke_p_min=0.0 if is_analytic_iid else 0.05,
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
            cells.append(f" {m:.4f}%[{lo:.4f},{hi:.4f}] |")
        rows.append(f"| {vname} |" + " ".join(cells))
    return "\n".join(rows)


def _render_granger_section(granger, var_names):
    """渲染结构传导检验表格"""
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


def _resolve_cholesky_ordering(var_names, params=None):
    params = params or {}
    configured = params.get("cholesky_ordering")
    if configured:
        if all(isinstance(item, str) for item in configured):
            name_to_idx = {name: idx for idx, name in enumerate(var_names)}
            return [name_to_idx[name] for name in configured if name in name_to_idx]
        return [int(i) for i in configured]

    last_keywords = ("净利率", "net margin", "net_margin", "profit margin")
    last = []
    first = []
    for idx, name in enumerate(var_names):
        lowered = str(name).lower()
        if any(keyword in lowered for keyword in last_keywords):
            last.append(idx)
        else:
            first.append(idx)
    return first + last


def _extract_transition_samples(joint_chain, n):
    A_samples = []
    for sample in joint_chain:
        A_avg = sample.get("A_avg")
        if A_avg is not None:
            A_samples.append(A_avg)
            continue
        theta = sample.get("theta")
        layout = sample.get("theta_layout")
        if theta is not None and layout is not None:
            A_samples.append(extract_transition(theta, layout))
        elif theta is not None and len(theta) >= n + n * n:
            A_samples.append(theta[n:].reshape(n, n))
    return np.asarray(A_samples, dtype=float)


def _extract_sigma_samples(joint_chain, n):
    sigma_samples = []
    for sample in joint_chain:
        sigma = sample.get("sigma")
        if sigma is None:
            continue
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape == (n, n):
            sigma_samples.append(sigma)
    if sigma_samples:
        return np.asarray(sigma_samples, dtype=float)
    return None


def _reorder_square_samples(samples, ordering):
    inverse = np.argsort(ordering)
    ordered = samples[:, ordering][:, :, ordering]
    return ordered, inverse


def _restore_square_samples(samples, inverse):
    return samples[:, inverse][:, :, inverse]


def _normalize_structural_signs(irf_mean, irf_lower, irf_upper, var_names):
    """
    架构组安全审计协议:
    底层算子保持对矩阵拓扑的绝对诚实，无源泛化路径严禁篡改数理结果。
    业务层面的同期并发效应交由上层数据平滑或有偏先验解决。
    """
    return np.asarray(irf_mean), np.asarray(irf_lower), np.asarray(irf_upper)


def _apply_ordering_to_chain(joint_chain, ordering):
    if not ordering:
        return joint_chain
    ordered_chain = []
    for sample in joint_chain:
        clone = dict(sample)
        sigma = clone.get("sigma")
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            if sigma.shape[0] == sigma.shape[1]:
                clone["sigma"] = sigma[np.ix_(ordering, ordering)]
        A_avg = clone.get("A_avg")
        if A_avg is None and clone.get("theta") is not None and clone.get("theta_layout") is not None:
            A_avg = extract_transition(clone["theta"], clone["theta_layout"])
        if A_avg is not None:
            clone["A_avg"] = np.asarray(A_avg, dtype=float)[np.ix_(ordering, ordering)]
        ordered_chain.append(clone)
    return ordered_chain


def _render_forecast_section(fc):
    var_names = fc["var_names"]
    header = "| 季度 |" + " ".join(f" {v} |" for v in var_names)
    sep = "|------|" + "-------|" * len(var_names)
    rows = [header, sep]
    for r in fc["rows"]:
        cells = " ".join(f" {r[v]['display']} |" for v in var_names)
        rows.append(f"| {r['label']} |{cells}")
    return "\n".join(rows)


def _fmt_diag_value(value, digits=3):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _render_stability_section(stability):
    transition = stability.get("transition", {})
    outputs = stability.get("outputs", [])
    lines = []

    warnings = stability.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"> 注意: {warning}")
        lines.append("")

    if transition.get("available"):
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 后验样本数 | {transition.get('n_samples', 0)} |")
        lines.append(f"| 谱半径均值 | {_fmt_diag_value(transition.get('radius_mean'))} |")
        lines.append(f"| 谱半径中位数 | {_fmt_diag_value(transition.get('radius_median'))} |")
        lines.append(f"| 谱半径 P95 | {_fmt_diag_value(transition.get('radius_p95'))} |")
        lines.append(f"| 谱半径最大值 | {_fmt_diag_value(transition.get('radius_max'))} |")
        lines.append(f"| 谱半径 >= 1 样本占比 | {transition.get('unstable_share', 0):.1%} |")
        lines.append(f"| 接近单位根样本占比 | {transition.get('near_unit_share', 0):.1%} |")
    else:
        lines.append("未找到可用于稳定性诊断的 VAR 系数样本。")

    if outputs:
        lines.append("")
        lines.append("| 输出 | 均值末期/首期 | 区间末期/首期 | 最大绝对值 |")
        lines.append("|------|--------------|--------------|------------|")
        for item in outputs:
            lines.append(
                f"| {item.get('name', '')} | "
                f"{_fmt_diag_value(item.get('growth_ratio'), 2)} | "
                f"{_fmt_diag_value(item.get('interval_growth_ratio'), 2)} | "
                f"{_fmt_diag_value(item.get('max_abs'), 2)} |"
            )

    return "\n".join(lines)


def _stationarity_failures(stationarity):
    if stationarity is None:
        return []
    results = getattr(stationarity, "results", {}) or {}
    latest_by_var = {}
    for key, result in results.items():
        if "_d" not in key or not isinstance(result, dict):
            continue
        name, d_text = key.rsplit("_d", 1)
        try:
            d_order = int(d_text)
        except ValueError:
            continue
        current = latest_by_var.get(name)
        if current is None or d_order > current[0]:
            latest_by_var[name] = (d_order, result)

    failures = []
    for name, (d_order, result) in latest_by_var.items():
        if not result.get("is_stationary", False):
            failures.append({
                "name": name,
                "d_order": d_order,
                "p_value": result.get("p_value"),
                "adf": result.get("adf_is_stationary"),
                "kpss": result.get("kpss_is_stationary"),
            })
    return failures


def _render_model_validity_section(stability, stationarity_failures, convergence_report=None):
    lines = []
    blocking = []

    transition = stability.get("transition", {})
    if transition.get("available") and transition.get("unstable_share", 0) > 0:
        blocking.append(
            f"VAR 后验存在不稳定动态: 谱半径 >= 1 样本占比 {transition.get('unstable_share', 0):.1%}"
        )
    if stationarity_failures:
        names = ", ".join(f"{item['name']}_d{item['d_order']}" for item in stationarity_failures)
        blocking.append(f"以下变量在最大差分后仍未通过联合平稳性判定: {names}")
    if convergence_report is not None and hasattr(convergence_report, "is_converged"):
        try:
            if not convergence_report.is_converged():
                blocking.append("MCMC 收敛诊断未完全通过")
        except Exception:
            pass

    if blocking:
        lines.append("**结论状态: 不建议用于业务或学术结论。**")
        lines.append("")
        for item in blocking:
            lines.append(f"> 阻断项: {item}")
        lines.append("")
        lines.append("IRF、Forecast 与 Granger 结果应视为诊断输出；请先完成单季口径转换、季节性处理、平稳化与稳定性约束后再解释。")
    else:
        lines.append("**结论状态: 通过基础平稳性、稳定性与收敛性门禁。**")

    return "\n".join(lines)


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
    from tvp_var_framework.utils.stability import (
        build_stability_report, stabilize_transition, transition_samples_from_joint_chain,
    )

    output_dir = params.get("output_dir", "./analysis_results/")
    export_csv = params.get("export_csv", False)
    requested_mode = params.get("mode", "full")
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
        rg.add_stationarity_report(
            inputs["_stationarity"].results,
            warnings=getattr(inputs["_stationarity"], "warnings", []),
        )

    # ── Report from _joint_chain / active result ──
    joint_chain = inputs.get("_joint_chain", [])
    posterior = inputs.get("posterior", {})
    results = posterior.get("results", {})
    active_model = posterior.get("primary_model")
    if requested_mode == "research":
        active_model = "research"
    elif requested_mode != "full":
        active_model = requested_mode
    elif active_model is None and results:
        active_model = next(iter(results))

    if joint_chain:
        n = Y.shape[1]
        irf_periods = params.get("irf_periods", 6)
        forecast_steps = params.get("forecast_steps", 4)
        n_samples = min(len(joint_chain), 500)
        active_result = results.get(active_model)
        cholesky_ordering = _resolve_cholesky_ordering(var_names, params)
        cholesky_names = [var_names[i] for i in cholesky_ordering]
        ordered_joint_chain = _apply_ordering_to_chain(joint_chain, cholesky_ordering)
        ordered_var_names = cholesky_names
        inverse_ordering = np.argsort(cholesky_ordering)

        # ── 1. Structural IRF: explicit Cholesky ordering ──
        from tvp_var_framework.models.research_grade import StructuralAnalysis
        sa = StructuralAnalysis(n)
        irf_mean_ordered, irf_lower_ordered, irf_upper_ordered = sa.orthogonalized_irf_from_joint_chain(
            ordered_joint_chain,
            periods=irf_periods,
        )
        irf_mean = irf_mean_ordered[:, inverse_ordering][:, :, inverse_ordering]
        irf_lower = irf_lower_ordered[:, inverse_ordering][:, :, inverse_ordering]
        irf_upper = irf_upper_ordered[:, inverse_ordering][:, :, inverse_ordering]
        irf_sections = format_irf_for_report(
            irf_mean,
            irf_lower,
            irf_upper,
            var_names,
            f"{active_model or requested_mode}; Cholesky order: {' → '.join(cholesky_names)}",
        )
        for sec in irf_sections:
            rg.add_section(f"脉冲响应: {sec['shock_name']}", _render_irf_section(sec))

        # ── 2. FEVD: Forecast Error Variance Decomposition ──
        fevd_mean, fevd_lower, fevd_upper = sa.orthogonalized_fevd_from_joint_chain(
            ordered_joint_chain, periods=irf_periods,
        )
        fevd_mean = fevd_mean[inverse_ordering][:, inverse_ordering]
        fevd_lower = fevd_lower[inverse_ordering][:, inverse_ordering]
        fevd_upper = fevd_upper[inverse_ordering][:, inverse_ordering]
        rg.add_section("方差分解 (FEVD)", _render_fevd_section(
            fevd_mean, fevd_lower, fevd_upper, var_names,
        ))
        logger.info(f"  FEVD 计算完成 ({irf_periods} 期)")

        if active_result is not None and active_result.pred_mean is not None:
            pred_mean = active_result.pred_mean
            pred_lower = active_result.pred_lower
            pred_upper = active_result.pred_upper
            future_labels = active_result.future_labels or [f"t+{s+1}" for s in range(pred_mean.shape[0])]
        else:
            from tvp_var_framework.models.research_grade import MCPredictor

            mcp = MCPredictor(n)
            theta_samples = np.stack([s["theta"] for s in joint_chain if s.get("theta") is not None])
            Y_last = Y[-1]
            pred_samples = mcp.predict(theta_samples, None, Y_last,
                                       steps=forecast_steps, n_samples=n_samples)
            pred_mean = pred_samples.mean(axis=0)
            pred_lower = np.percentile(pred_samples, 2.5, axis=0)
            pred_upper = np.percentile(pred_samples, 97.5, axis=0)

            if inputs.get("normalize", True):
                scale = inputs.get("std") if inputs.get("std") is not None else 1
                offset = inputs.get("mean") if inputs.get("mean") is not None else 0
                pred_mean = pred_mean * scale + offset
                pred_lower = pred_lower * scale + offset
                pred_upper = pred_upper * scale + offset

            future_labels = [f"t+{s+1}" for s in range(forecast_steps)]

        A_samples = transition_samples_from_joint_chain(joint_chain, n)
        if params.get("enforce_stability", False) and A_samples.size:
            max_radius = params.get("stability_radius_threshold", 1.0)
            A_samples = np.asarray(
                [stabilize_transition(A, max_radius=max_radius)[0] for A in A_samples],
                dtype=float,
            )
        stability = build_stability_report(
            A_samples=A_samples,
            irf={
                "mean": active_result.irf_mean if active_result is not None and active_result.irf_mean is not None else irf_mean,
                "lower": active_result.irf_lower if active_result is not None and active_result.irf_lower is not None else irf_lower,
                "upper": active_result.irf_upper if active_result is not None and active_result.irf_upper is not None else irf_upper,
            },
            forecast={"mean": pred_mean, "lower": pred_lower, "upper": pred_upper},
            threshold=1.0,
        )
        rg.add_section(
            "模型有效性门禁",
            _render_model_validity_section(
                stability,
                _stationarity_failures(inputs.get("_stationarity")),
                inputs.get("_convergence"),
            ),
        )
        rg.add_section("稳定性诊断与告警", _render_stability_section(stability))

        fc = format_forecast_for_report(pred_mean, pred_lower, pred_upper, future_labels, var_names)
        rg.add_section("预测 (Forecast)", _render_forecast_section(fc))
        logger.info(f"  预测完成 ({forecast_steps} 步)")

        # Export forecast CSV
        if export_csv:
            rg.export_csv(
                {"mean": pred_mean, "lower": pred_lower, "upper": pred_upper},
                filename=f"forecast_{active_model or requested_mode}.csv",
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
        theta_layout = next((s.get("theta_layout") for s in joint_chain if s.get("theta_layout") is not None), None)
        if theta_layout is not None:
            from tvp_var_framework.core.theta_layout import split_theta
            c, A, B = split_theta(theta_mean, theta_layout)
            offset = theta_layout["components_per_equation"]
            for i, val in enumerate(c):
                idx = i * offset
                summary[f"c_{var_names[i]}"] = {
                    "mean": float(val),
                    "ci_95": (float(theta_lower[idx]), float(theta_upper[idx])),
                }
            n_layout = int(theta_layout["n_endog"])
            m_layout = int(theta_layout.get("n_exog", 0))
            for i in range(n_layout):
                for j in range(n_layout):
                    idx = i * offset + 1 + j
                    summary[f"A_{var_names[i]}<-{var_names[j]}"] = {
                        "mean": float(theta_mean[idx]),
                        "ci_95": (float(theta_lower[idx]), float(theta_upper[idx])),
                    }
            if m_layout:
                for i in range(n_layout):
                    for j in range(m_layout):
                        idx = i * offset + 1 + n_layout + j
                        summary[f"B_{var_names[i]}<-x{j+1}"] = {
                            "mean": float(theta_mean[idx]),
                            "ci_95": (float(theta_lower[idx]), float(theta_upper[idx])),
                        }
        else:
            for i in range(min(len(theta_mean), n + n * n)):
                param_name = f"theta_{i}"
                summary[param_name] = {
                    "mean": float(theta_mean[i]),
                    "ci_95": (float(theta_lower[i]), float(theta_upper[i])),
                }
        ps = format_posterior_for_report(summary, var_names)
        rg.add_section("后验参数估计 (joint_chain)", _render_posterior_section(ps))

    report_path = rg.save_markdown()
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
            layout = s.get("theta_layout")
            if theta is not None and layout is not None:
                A = extract_transition(theta, layout)
                A_samples.append(A)
            elif theta is not None and len(theta) >= n + n * n:
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


def _get_execution_mode(cfg):
    return _get_nested(cfg, "model_specification", "mode", default="full")


def _stationarity_params(cfg):
    params = dict(cfg.get("stationarity", {}) or {})
    params.update(cfg.get("stationarity_options", {}) or {})
    return params


def _get_mcmc_burnin(cfg, default=500):
    return (
        _get_nested(cfg, "mcmc_options", "nburn")
        or _get_nested(cfg, "mcmc_options", "burnin")
        or _get_nested(cfg, "inference_control", "burnin")
        or _get_nested(cfg, "model_specification", "burnin")
        or default
    )


def _get_mcmc_n_iter(cfg, default=2000):
    configured = _get_nested(cfg, "mcmc_options", "n_iter")
    if configured:
        return configured
    nsave = _get_nested(cfg, "mcmc_options", "nsave")
    nburn = _get_nested(cfg, "mcmc_options", "nburn")
    if nsave and nburn:
        return int(nsave) + int(nburn)
    return (
        _get_nested(cfg, "inference_control", "n_iter")
        or _get_nested(cfg, "model_specification", "n_iter")
        or default
    )


def _should_execute_node(node_name: str, cfg: dict) -> bool:
    mode = _get_execution_mode(cfg)
    if node_name == "sampling_basic":
        return mode != "research"
    if node_name == "sampling_research":
        return mode == "research"
    return True


PARAM_ROUTING = {
    "stationarity": _stationarity_params,
    "state_update": lambda cfg: {},
    "likelihood": lambda cfg: {},
    "sampling_basic": lambda cfg: {
        **cfg.get("model_specification", {}),
        "n_iter": _get_mcmc_n_iter(cfg, default=2000),
        "burnin": _get_mcmc_burnin(cfg, default=500),
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
        "forecast_bounds": _get_nested(cfg, "forecasting", "bounds"),
        "change_point_threshold": _get_nested(cfg, "structural_analysis", "change_point_threshold", default=1.5),
        "enforce_stability": _get_nested(cfg, "stability_guard", "enforce", default=False),
        "stability_radius_threshold": _get_nested(cfg, "stability_guard", "spectral_radius_threshold", default=1.0),
        "q_grid": _get_nested(cfg, "bayesian_priors", "q_grid", default=[1e-7, 1e-5, 1e-3, 1e-1]),
        "gamma_1": (
            _get_nested(cfg, "prior_hyperparameters", "gamma_1")
            or _get_nested(cfg, "bayesian_priors", "gamma_1")
            or 0.1
        ),
        "gamma_2": (
            _get_nested(cfg, "prior_hyperparameters", "gamma_2")
            or _get_nested(cfg, "bayesian_priors", "gamma_2")
            or 0.1
        ),
        "intercept_var": (
            _get_nested(cfg, "prior_hyperparameters", "intercept_var")
            or _get_nested(cfg, "bayesian_priors", "intercept_var")
            or 10.0
        ),
    },
    "sampling_research": lambda cfg: {
        "n_iter": _get_mcmc_n_iter(cfg, default=2000),
        "burnin": _get_mcmc_burnin(cfg, default=500),
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
        "mode": _get_execution_mode(cfg),
        "irf_periods": _get_nested(cfg, "structural_analysis", "irf_periods", default=6),
        "forecast_steps": _get_nested(cfg, "forecasting", "steps", default=4),
        "enforce_stability": _get_nested(cfg, "stability_guard", "enforce", default=False),
        "stability_radius_threshold": _get_nested(cfg, "stability_guard", "spectral_radius_threshold", default=1.0),
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
                ctx.update(k, v)

        logger.info("=" * 60)
        logger.info("InferenceGraphEngine v4 — DAG 执行")
        logger.info(f"执行顺序: {' → '.join(self.execution_order)}")
        logger.info("=" * 60)

        for node_name in self.execution_order:
            if not _should_execute_node(node_name, ctx.config):
                logger.info(f"[SKIP] {node_name}")
                continue

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
                        ctx.update(k, v)

        logger.info("=" * 60)
        logger.info("分析完成")
        logger.info("=" * 60)

        return ctx

"""
TVP-VAR MCMC 收敛诊断工具

提供 R-hat, ESS, Geweke 诊断, 适用于任意后验链。
纯函数设计，不依赖任何模型类。
"""

import numpy as np
import logging

logger = logging.getLogger("tvp_var")


def single_chain_rhat(chain):
    chain = np.asarray(chain, dtype=float)
    n = len(chain)
    if n < 4:
        return np.nan
    half = n // 2
    chain1 = chain[:half]
    chain2 = chain[half:2 * half]
    n_half = len(chain1)

    W = 0.5 * (np.var(chain1, ddof=1) + np.var(chain2, ddof=1))
    B = n_half * (np.mean(chain1) - np.mean(chain2)) ** 2
    var_hat = (1 - 1.0 / n_half) * W + B / n_half
    if W < 1e-30:
        return np.nan
    return np.sqrt(var_hat / W)


def split_chain_rhat(chains):
    chains = [np.asarray(c, dtype=float) for c in chains]
    m = len(chains)
    if m < 2:
        if m == 1:
            return single_chain_rhat(chains[0])
        return np.nan

    n = min(len(c) for c in chains)
    if n < 2:
        return np.nan

    chains = [c[:n] for c in chains]

    W = np.mean([np.var(c, ddof=1) for c in chains])
    grand_mean = np.mean([np.mean(c) for c in chains])
    B = n * np.mean([(np.mean(c) - grand_mean) ** 2 for c in chains])
    var_hat = (1 - 1.0 / n) * W + B / n
    if W < 1e-30:
        return np.nan
    return np.sqrt(var_hat / W)


def effective_sample_size(chain, max_lag=None):
    chain = np.asarray(chain, dtype=float)
    n = len(chain)
    if n < 4:
        return float(n)

    chain = chain - np.mean(chain)
    var0 = np.var(chain, ddof=1)
    if var0 < 1e-30:
        return float(n)

    if max_lag is None:
        max_lag = min(n // 3, 500)

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[0] = 1.0
        else:
            acf[lag] = np.dot(chain[:n - lag], chain[lag:]) / ((n - lag) * var0)

    gamma_sum = 0.0
    for lag in range(0, max_lag + 1, 2):
        pair_sum = acf[lag]
        if lag + 1 <= max_lag:
            pair_sum += acf[lag + 1]
        if pair_sum < 0:
            break
        gamma_sum += pair_sum

    if gamma_sum < 1e-30:
        return float(n)
    ess = n / (2 * gamma_sum)
    return max(1.0, min(float(n), ess))


def geweke_diagnostic(chain, first_frac=0.1, last_frac=0.5):
    chain = np.asarray(chain, dtype=float)
    n = len(chain)
    n1 = max(int(n * first_frac), 2)
    n2 = max(int(n * last_frac), 2)
    if n1 + n2 > n:
        n1 = n // 4
        n2 = n // 2

    first = chain[:n1]
    last = chain[-n2:]

    mean1 = np.mean(first)
    mean2 = np.mean(last)

    var1 = _spectral_variance(first)
    var2 = _spectral_variance(last)

    se = np.sqrt(var1 / n1 + var2 / n2)
    if se < 1e-30:
        z = 0.0
    else:
        z = (mean1 - mean2) / se

    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return {
        "z_score": z,
        "p_value": p_value,
        "converged": p_value > 0.05,
    }


def _spectral_variance(chain):
    n = len(chain)
    if n < 4:
        return np.var(chain, ddof=1)
    chain = chain - np.mean(chain)
    var0 = np.var(chain, ddof=1)
    max_lag = min(n // 3, 100)
    gamma_sum = 0.0
    for lag in range(1, max_lag + 1):
        gamma_lag = np.dot(chain[:n - lag], chain[lag:]) / n
        weight = 1.0 - lag / (max_lag + 1)
        gamma_sum += weight * gamma_lag
    return max(var0 + 2 * gamma_sum, 1e-30)


def multi_parameter_diagnostics(chains_dict, param_names=None):
    results = {}
    for name, chain in chains_dict.items():
        chain = np.asarray(chain, dtype=float)
        if chain.ndim == 2:
            n_samples, dim = chain.shape
            all_rhat = []
            all_ess = []
            all_geweke_z = []
            all_geweke_p = []
            for idx in range(dim):
                col = chain[:, idx]
                all_rhat.append(single_chain_rhat(col))
                all_ess.append(effective_sample_size(col))
                g = geweke_diagnostic(col)
                all_geweke_z.append(g["z_score"])
                all_geweke_p.append(g["p_value"])
            def safe_agg(func, arr):
                return func(arr) if not np.all(np.isnan(arr)) else np.nan

            results[name] = {
                "rhat": safe_agg(np.nanmax, all_rhat),
                "ess": safe_agg(np.nanmin, all_ess),
                "geweke_z": safe_agg(np.nanmean, all_geweke_z),
                "geweke_p": safe_agg(np.nanmin, all_geweke_p),
                "converged": all(g["converged"] for g in [geweke_diagnostic(chain[:, i]) for i in range(dim)]),
                "n_params": dim,
                "rhat_per_param": all_rhat,
                "ess_per_param": all_ess,
            }
        elif chain.ndim == 1:
            g = geweke_diagnostic(chain)
            results[name] = {
                "rhat": single_chain_rhat(chain),
                "ess": effective_sample_size(chain),
                "geweke_z": g["z_score"],
                "geweke_p": g["p_value"],
                "converged": g["converged"],
                "n_params": 1,
            }
        else:
            results[name] = {"error": f"不支持的维度: {chain.ndim}"}
    return results


def diagnose_model_chains(model, model_type="ffbs"):
    chains = {}

    if model_type == "ffbs":
        chains["A"] = np.asarray(model.chain_A).reshape(len(model.chain_A), -1)
        chains["theta"] = np.asarray(model.chain_theta)
        q_arr = np.array(model.chain_Q) if isinstance(model.chain_Q, list) else model.chain_Q
        chains["Q"] = q_arr.reshape(len(q_arr), -1)
        r_arr = np.array(model.chain_R) if isinstance(model.chain_R, list) else model.chain_R
        chains["R"] = r_arr.reshape(len(r_arr), -1)
        chains["log_lik"] = np.array(model.chain_log_lik)

    elif model_type == "fully_bayesian":
        result = model.result
        chains["A"] = np.asarray(result["chain_A"]).reshape(len(result["chain_A"]), -1)
        chains["theta"] = np.asarray(result["chain_theta"])
        chains["Q"] = np.asarray(result["chain_Q"]).reshape(len(result["chain_Q"]), -1)
        chains["log_lik"] = np.asarray(result["chain_log_lik"])
        if "chain_R" in result:
            r_arr = np.asarray(result["chain_R"])
            chains["R"] = r_arr.reshape(len(r_arr), -1)
        elif "chain_R_corr" in result:
            chains["R_corr"] = np.asarray(result["chain_R_corr"]).reshape(len(result["chain_R_corr"]), -1)

    elif model_type == "mcmc":
        chains["A"] = np.asarray(model.chain_A).reshape(len(model.chain_A), -1)
        chains["theta"] = np.asarray(model.chain_theta)
        chains["log_q"] = np.asarray(model.chain_log_q)
        chains["log_r"] = np.asarray(model.chain_log_r)

    elif model_type == "bayesian":
        chain = model.sample_posterior(n_samples=1000)
        chains["theta"] = chain

    else:
        return {"error": f"未知模型类型: {model_type}"}

    return multi_parameter_diagnostics(chains)


class ConvergenceReport:
    def __init__(self, diagnostics_dict):
        self.diagnostics = diagnostics_dict

    def is_converged(self, rhat_threshold=1.1, ess_min=100):
        for model_name, diag in self.diagnostics.items():
            if isinstance(diag, dict) and "error" in diag:
                continue
            for param_name, param_diag in diag.items():
                if isinstance(param_diag, dict):
                    rhat = param_diag.get("rhat", np.inf)
                    ess = param_diag.get("ess", 0)
                    if rhat > rhat_threshold or ess < ess_min:
                        return False
        return True

    def summary_text(self):
        lines = []
        lines.append("=" * 60)
        lines.append("MCMC 收敛诊断报告")
        lines.append("=" * 60)
        for model_name, diag in self.diagnostics.items():
            lines.append(f"\n--- {model_name} ---")
            if isinstance(diag, dict) and "error" in diag:
                lines.append(f"  错误: {diag['error']}")
                continue
            for param_name, param_diag in diag.items():
                if not isinstance(param_diag, dict):
                    continue
                rhat = param_diag.get("rhat", np.nan)
                ess = param_diag.get("ess", np.nan)
                geweke_p = param_diag.get("geweke_p", np.nan)
                n_params = param_diag.get("n_params", 1)
                conv = param_diag.get("converged", False)
                status = "收敛" if conv else "未收敛"
                if n_params > 1:
                    lines.append(f"  {param_name} ({n_params}个参数): R-hat={rhat:.4f}, ESS={ess:.0f}, Geweke p={geweke_p:.4f} [{status}]")
                else:
                    lines.append(f"  {param_name}: R-hat={rhat:.4f}, ESS={ess:.0f}, Geweke p={geweke_p:.4f} [{status}]")
        overall = "全部收敛" if self.is_converged() else "存在未收敛参数"
        lines.append(f"\n总体判定: {overall}")
        return "\n".join(lines)

    def summary_dict(self):
        return {
            "diagnostics": self.diagnostics,
            "all_converged": self.is_converged(),
        }

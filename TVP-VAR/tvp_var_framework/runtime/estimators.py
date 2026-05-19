"""
Estimation Kernels — extracted from inference_engine.py

Each kernel is a pure function:
    fn(inputs: dict, params: dict) -> dict

All kernels output _joint_chain for unified posterior semantics.
For basic estimators, chains are restructured into joint_chain format
(post-hoc coupling for backward compatibility).
"""

import logging
import numpy as np

logger = logging.getLogger("tvp_var.runtime.estimators")


def _apply_forecast_bounds(means, lowers, uppers, bounds):
    """Apply optional config-driven bounds by output column index."""
    if not bounds:
        return means, lowers, uppers

    means = np.asarray(means, dtype=float).copy()
    lowers = np.asarray(lowers, dtype=float).copy()
    uppers = np.asarray(uppers, dtype=float).copy()

    specs = bounds if isinstance(bounds, list) else [bounds]
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        indices = spec.get("indices", [])
        lower = spec.get("lower")
        upper = spec.get("upper")
        for idx in indices:
            j = int(idx)
            if j < 0 or j >= means.shape[1]:
                raise IndexError(f"forecast_bounds index out of range: {j}")
            if lower is not None:
                means[:, j] = np.maximum(means[:, j], float(lower))
                lowers[:, j] = np.maximum(lowers[:, j], float(lower))
                uppers[:, j] = np.maximum(uppers[:, j], float(lower))
            if upper is not None:
                means[:, j] = np.minimum(means[:, j], float(upper))
                lowers[:, j] = np.minimum(lowers[:, j], float(upper))
                uppers[:, j] = np.minimum(uppers[:, j], float(upper))

    lowers = np.minimum(lowers, means)
    uppers = np.maximum(uppers, means)
    return means, lowers, uppers


def _build_joint_chain_from_model(model, model_type, n_samples=500):
    """
    Restructure existing model chains into _joint_chain format.

    NOTE: This is post-hoc restructuring, NOT sampling-time coupling.
    For true coupling, use kernel_sampler_research.

    Parameters
    ----------
    model : object
        Fitted model with chain attributes.
    model_type : str
        One of "fully_bayesian", "bayesian", "v2", "research".

    Returns
    -------
    list[dict]
        Each dict has keys: theta, sigma, sv_state, log_likelihood, sample_index
    """
    joint_chain = []

    if model_type == "fully_bayesian":
        result = model.result if hasattr(model, "result") else {}
        chain_theta = result.get("chain_theta", [])
        chain_Q = result.get("chain_Q", [])
        chain_R = result.get("chain_R", [])
        chain_R_t = result.get("chain_R_t", [])
        chain_log_lik = result.get("chain_log_lik", [])

        n_samples = len(chain_theta)
        for i in range(n_samples):
            theta_i = chain_theta[i] if i < len(chain_theta) else None
            if chain_R_t and i < len(chain_R_t):
                sigma_i = chain_R_t[i][-1]  # last time-point R_t
            elif i < len(chain_R):
                sigma_i = chain_R[i]
            else:
                sigma_i = None
            log_lik_i = chain_log_lik[i] if i < len(chain_log_lik) else None

            joint_chain.append({
                "sample_index": i,
                "theta": np.copy(theta_i) if theta_i is not None else None,
                "sigma": np.copy(sigma_i) if sigma_i is not None else None,
                "sv_state": None,
                "log_likelihood": float(log_lik_i) if log_lik_i is not None else None,
            })

    elif model_type == "ffbs":
        chain_theta = getattr(model, "chain_theta", [])
        chain_Q = getattr(model, "chain_Q", [])
        chain_R = getattr(model, "chain_R", [])
        chain_log_lik = getattr(model, "chain_log_lik", [])

        n_samples = len(chain_theta)
        for i in range(n_samples):
            theta_i = chain_theta[i] if i < len(chain_theta) else None
            sigma_i = chain_R[i] if i < len(chain_R) else None
            log_lik_i = chain_log_lik[i] if i < len(chain_log_lik) else None

            joint_chain.append({
                "sample_index": i,
                "theta": np.copy(theta_i) if theta_i is not None else None,
                "sigma": np.copy(sigma_i) if sigma_i is not None else None,
                "sv_state": None,
                "log_likelihood": float(log_lik_i) if log_lik_i is not None else None,
            })

    elif model_type == "bayesian":
        # BayesianTVP_VAR has analytical posterior, sample from it
        if hasattr(model, "sample_posterior"):
            # Extract observation noise covariance as default sigma
            innovations = []
            if hasattr(model, "analyst") and hasattr(model.analyst, "history"):
                innovations = [
                    np.asarray(record.get("innovation"), dtype=float)
                    for record in model.analyst.history
                    if record.get("innovation") is not None
                ]
            if len(innovations) > 1:
                innov_arr = np.vstack(innovations)
                default_sigma = np.cov(innov_arr.T) + np.eye(innov_arr.shape[1]) * 1e-8
            elif hasattr(model, "analyst") and hasattr(model.analyst, "R"):
                default_sigma = np.copy(model.analyst.R)
            else:
                n = model.n_vars if hasattr(model, "n_vars") else 2
                default_sigma = np.eye(n) * 0.1

            samples = model.sample_posterior(n_samples=n_samples)
            for i, theta_i in enumerate(samples):
                sample = {
                    "sample_index": i,
                    "theta": np.copy(theta_i),
                    "sigma": np.copy(default_sigma),
                    "sv_state": None,
                    "log_likelihood": None,
                    "chain_source": "analytic_iid",
                }
                if hasattr(model, "meta"):
                    sample["theta_layout"] = dict(model.meta)
                elif hasattr(model, "analyst") and hasattr(model.analyst, "meta"):
                    sample["theta_layout"] = dict(model.analyst.meta)
                joint_chain.append(sample)

    elif model_type == "research":
        # Research path uses CholeskyKalmanFilter — no MCMC chains
        # Return empty; actual research path uses kernel_sampler_research
        pass

    return joint_chain


def estimate_fully_bayesian(inputs: dict, params: dict) -> dict:
    """Fully Bayesian TVP-VAR (Gibbs sampling with SV)."""
    from tvp_var_framework.models.fully_bayesian import FullyBayesianTVPVAR
    from tvp_var_framework.core.model_result import ModelResult

    Y = inputs.get("Y_diff", inputs.get("Y"))
    n = Y.shape[1]
    names = inputs["var_names"]
    X_exog = inputs.get("X_exog")
    exog_names = inputs.get("exog_names", [])
    n_exog = 0 if X_exog is None else X_exog.shape[1]

    logger.info("=" * 60)
    logger.info("Fully Bayesian TVP-VAR (Gibbs 采样)")
    logger.info("=" * 60)

    fb = FullyBayesianTVPVAR(
        n_vars=n,
        n_iter=params.get("n_iter", 2000),
        burnin=params.get("burnin", 800),
        thin=params.get("thin", 2),
        sv_n_iter=params.get("sv_n_iter", 300),
        sv_burnin=params.get("sv_burnin", 100),
    )
    fb.fit(Y, use_sv=params.get("sv_enabled", True), verbose=inputs.get("debug", False))
    fb.posterior_summary(var_names=names)

    fc_steps = params.get("forecast_steps", 4)
    fc_samples = params.get("forecast_samples", 500)
    mean = inputs.get("mean")
    std = inputs.get("std")
    normalize = inputs.get("normalize", True)

    pred_mean, pred_lower, pred_upper = fb.predict(
        steps=fc_steps, n_samples=fc_samples, Y_last=Y[-1],
        mean=mean if normalize else None,
        std=std if normalize else None,
    )

    irf_periods = params.get("irf_periods", 6)
    logger.info(f"IRF: Fully Bayesian ({irf_periods} 期)")
    irf_data = {}
    for shock_var in range(n):
        irf_m, irf_l, irf_u = fb.impulse_response(
            shock_var=shock_var, shock_size=1.0,
            periods=irf_periods, var_names=names,
        )
        irf_data[names[shock_var]] = (irf_m, irf_l, irf_u)

    result = ModelResult(
        model_name="fully_bayesian",
        pred_mean=pred_mean, pred_lower=pred_lower, pred_upper=pred_upper,
        log_likelihood=float(np.mean(fb.result["chain_log_lik"]))
            if "chain_log_lik" in fb.result else None,
        chains={k: v for k, v in fb.result.items() if k.startswith("chain_")},
    )
    if irf_data:
        result.irf_mean = np.stack([v[0] for v in irf_data.values()], axis=-1)
        result.irf_lower = np.stack([v[1] for v in irf_data.values()], axis=-1)
        result.irf_upper = np.stack([v[2] for v in irf_data.values()], axis=-1)

    joint_chain = _build_joint_chain_from_model(fb, "fully_bayesian")

    return {"_model": fb, "_result": result, "_joint_chain": joint_chain}


def estimate_bayesian(inputs: dict, params: dict) -> dict:
    """Bayesian model comparison + BMA."""
    from tvp_var_framework.models.bayesian import (
        BayesianTVP_VAR, BayesianModelComparison,
        BayesianModelAveraging,
    )
    from tvp_var_framework.core.model_result import ModelResult

    Y = inputs.get("Y_diff", inputs.get("Y"))
    n = Y.shape[1]
    names = inputs["var_names"]
    X_exog = inputs.get("X_exog")
    exog_names = inputs.get("exog_names", [])
    n_exog = 0 if X_exog is None else X_exog.shape[1]

    logger.info("=" * 60)
    logger.info("贝叶斯模型比较")
    logger.info("=" * 60)

    comparison = BayesianModelComparison()
    q_grid = params.get("q_grid", [1e-7, 1e-5, 1e-3, 1e-1])
    gamma_1 = params.get("gamma_1", 0.1)
    gamma_2 = params.get("gamma_2", 0.1)
    intercept_var = params.get("intercept_var", 10.0)
    for q in q_grid:
        comparison.add_model(
            f"q={q:.0e}",
            BayesianTVP_VAR(
                n_vars=n,
                n_exog=n_exog,
                q=q,
                r=1e-3,
                gamma_1=gamma_1,
                gamma_2=gamma_2,
                intercept_var=intercept_var,
            ),
        )
    comparison.fit_all(Y, X_exog=X_exog)
    if n_exog:
        logger.info(f"VARX 外生变量: {exog_names}")

    best_name = comparison.best_model("bic")
    best_model = comparison.models[best_name]
    probs = comparison.posterior_model_probs()

    for name in sorted(comparison.results.keys()):
        r = comparison.results[name]
        p = probs[name]
        logger.info(f"  {name:<12} BIC={r['bic']:>12.1f}  后验概率={p:>10.4f}")
    logger.info(f"最优: {best_name}")

    summary = best_model.posterior_summary(var_names=names)
    for param, st in summary.items():
        ci = st["ci_95"]
        sig = "*" if ci[0] > 0 or ci[1] < 0 else ""
        logger.info(f"  {param:<15} {st['mean']:>8.4f}  [{ci[0]:>7.4f}, {ci[1]:>7.4f}] {sig}")

    ql = inputs.get("time_index", [])
    breaks = best_model.detect_structural_breaks(
        threshold=params.get("change_point_threshold", 1.5)
    )
    if breaks:
        logger.info(f"结构突变检测: {len(breaks)} 个")
        for t, param, change, z in breaks[:8]:
            label = ql[t] if t < len(ql) else f"t={t}"
            logger.info(f"  {label:<10} {param:<10} {change:>10.4f} {z:>10.2f}")

    fc_steps = params.get("forecast_steps", 4)
    fc_samples = params.get("forecast_samples", 500)
    stability_max_radius = params.get("stability_radius_threshold", 1.0)
    mean = inputs.get("mean")
    std = inputs.get("std")
    normalize = inputs.get("normalize", True)

    bma = BayesianModelAveraging(comparison)
    means, lowers, uppers = bma.predict_bma_interval(
        Y, steps=fc_steps, n_samples=fc_samples,
        X_exog_history=X_exog,
        enforce_stability=params.get("enforce_stability", False),
        stability_max_radius=stability_max_radius,
    )
    if normalize:
        s = std if std is not None else 1
        m = mean if mean is not None else 0
        means = means * s + m
        lowers = lowers * s + m
        uppers = uppers * s + m
    means, lowers, uppers = _apply_forecast_bounds(
        means, lowers, uppers, params.get("forecast_bounds")
    )

    irf_periods = params.get("irf_periods", 6)
    logger.info(f"IRF: Bayesian ({irf_periods} 期)")
    irf_mean = irf_lower = irf_upper = None
    for sv_idx, sn in enumerate(names):
        shock_irf_mean, shock_irf_lower, shock_irf_upper = best_model.impulse_response(
            shock_var=sv_idx, shock_size=1.0, periods=irf_periods, n_samples=300,
            enforce_stability=params.get("enforce_stability", False),
            stability_max_radius=stability_max_radius,
        )
        if irf_mean is None:
            irf_mean = np.zeros((irf_periods, n, n))
            irf_lower = np.zeros((irf_periods, n, n))
            irf_upper = np.zeros((irf_periods, n, n))
        irf_mean[:, :, sv_idx] = shock_irf_mean
        irf_lower[:, :, sv_idx] = shock_irf_lower
        irf_upper[:, :, sv_idx] = shock_irf_upper

    result = ModelResult(
        model_name="bayesian",
        pred_mean=means, pred_lower=lowers, pred_upper=uppers,
        log_likelihood=float(best_model.log_marginal_likelihood),
        bic=float(best_model.bic),
    )
    result.irf_mean = irf_mean
    result.irf_lower = irf_lower
    result.irf_upper = irf_upper

    posterior_samples = params.get("posterior_samples")
    if posterior_samples is None:
        posterior_samples = max(500, params.get("n_iter", 1000) - params.get("burnin", 500))
    joint_chain = _build_joint_chain_from_model(
        best_model,
        "bayesian",
        n_samples=posterior_samples,
    )

    return {"_model": best_model, "_result": result, "_joint_chain": joint_chain}


def estimate_v2(inputs: dict, params: dict) -> dict:
    """TVP-VAR v2: FFBS + Markov switching + SV."""
    from tvp_var_framework.models.ffbs import (
        FFBS_Sampler, MarkovSwitchingTVP, PathwiseMCPredictor, BayesianSV,
    )
    from tvp_var_framework.core.model_result import ModelResult

    Y = inputs.get("Y_diff", inputs.get("Y"))
    n = Y.shape[1]

    logger.info("=" * 60)
    logger.info("TVP-VAR v2 研究级模块")
    logger.info("=" * 60)

    ffbs = FFBS_Sampler(
        n_vars=n,
        n_iter=params.get("n_iter", 2000),
        burnin=params.get("burnin", 800),
        thin=params.get("thin", 2),
    )
    ffbs.fit(Y, verbose=inputs.get("debug", False))
    logger.info(f"FFBS 后验均值 A 对角: {np.diag(ffbs.chain_A.mean(axis=0)).round(4)}")

    ms = MarkovSwitchingTVP(n_vars=n, n_regimes=2)
    ms_results = ms.fit(Y, n_iter=800, burnin=300, verbose=inputs.get("debug", False))
    final_states = ms_results["final_states"]
    regime_counts = [np.sum(final_states == s) for s in range(2)]
    logger.info(f"区制分布: {', '.join(f'区制{i}={c}期' for i, c in enumerate(regime_counts))}")

    fc_samples = params.get("forecast_samples", 500)
    mc_pred = PathwiseMCPredictor(n_vars=n)
    forecast_steps = params.get("forecast_steps", 4)
    preds = mc_pred.predict(
        ffbs.chain_theta, ffbs.chain_Q, ffbs.chain_R,
        Y[-1], steps=forecast_steps, n_samples=min(300, fc_samples),
    )
    mean_pred, lower_pred, upper_pred = mc_pred.interval(preds)
    normalize = inputs.get("normalize", True)
    if normalize:
        s = inputs.get("std") if inputs.get("std") is not None else 1
        m = inputs.get("mean") if inputs.get("mean") is not None else 0
        mean_pred, lower_pred, upper_pred = mean_pred * s + m, lower_pred * s + m, upper_pred * s + m

    # SV
    if params.get("sv_enabled", True):
        T = len(Y)
        k = n + n * n
        residuals = np.zeros((T - 1, n))
        states_f, _, _, _, _ = ffbs._forward_filter(Y, ffbs.chain_Q[-1], ffbs.chain_R[-1])
        for t in range(1, T):
            Z = np.zeros((n, k))
            for i in range(n):
                Z[i, i] = 1.0
                Z[i, n + i * n:n + (i + 1) * n] = Y[t - 1]
            residuals[t - 1] = Y[t] - Z @ states_f[t]

        sv = BayesianSV(n_vars=n)
        sv_results = sv.fit(residuals, n_iter=500, burnin=150)
        for i, name in enumerate(inputs["var_names"]):
            logger.info(f"  SV {name}: mu={sv_results['mu'].mean(axis=0)[i]:.3f}, phi={sv_results['phi'].mean(axis=0)[i]:.3f}")

    result = ModelResult(
        model_name="v2",
        pred_mean=mean_pred, pred_lower=lower_pred, pred_upper=upper_pred,
        log_likelihood=float(np.mean(ffbs.chain_log_lik)),
        chains=ffbs.get_chains(),
    )

    irf_mean = irf_lower = irf_upper = None
    for shock_var in range(n):
        shock_irf_mean, shock_irf_lower, shock_irf_upper = ffbs.compute_irf(
            shock_var=shock_var, periods=forecast_steps, shock_size=1.0
        )
        if irf_mean is None:
            irf_mean = np.zeros((forecast_steps, n, n))
            irf_lower = np.zeros((forecast_steps, n, n))
            irf_upper = np.zeros((forecast_steps, n, n))
        irf_mean[:, :, shock_var] = shock_irf_mean
        irf_lower[:, :, shock_var] = shock_irf_lower
        irf_upper[:, :, shock_var] = shock_irf_upper

    result.irf_mean = irf_mean
    result.irf_lower = irf_lower
    result.irf_upper = irf_upper

    joint_chain = _build_joint_chain_from_model(ffbs, "ffbs")

    return {"_model": ffbs, "_result": result, "_joint_chain": joint_chain}


def estimate_research(inputs: dict, params: dict) -> dict:
    """Research-grade: Cholesky Kalman + change point + structural IRF."""
    from tvp_var_framework.models.research_grade import (
        CholeskyKalmanFilter, BayesianChangePoint, StructuralAnalysis,
    )
    from tvp_var_framework.core.model_result import ModelResult

    Y = inputs.get("Y_diff", inputs.get("Y"))
    n = Y.shape[1]

    logger.info("=" * 60)
    logger.info("Research-grade 分析")
    logger.info("=" * 60)

    ckf = CholeskyKalmanFilter(n_vars=n, q=0.01, r=0.1)
    filtered, P_filt, log_lik = ckf.filter(Y)
    smoothed, P_smooth = ckf.smooth(filtered, P_filt)

    k = n + n * n
    bic_val = -2 * log_lik + k * np.log(len(Y))
    logger.info(f"对数似然: {log_lik:.2f}, BIC: {bic_val:.2f}")

    bcp = BayesianChangePoint(n_vars=n, max_changepoints=5)
    A_history = np.array([smoothed[t][n:].reshape(n, n) for t in range(len(smoothed))])
    cps = bcp.detect(A_history, method="marginal_lik")
    if cps:
        ql = inputs.get("time_index", [])
        logger.info(f"变点检测: {len(cps)} 个")
        for t, score in cps[:5]:
            label = ql[t] if t < len(ql) else f"t={t}"
            logger.info(f"  变点: {label}, 得分={score:.2f}")

    # IRF
    irf_periods = params.get("irf_periods", 6)
    logger.info(f"IRF: Research-grade ({irf_periods} 期)")
    sa = StructuralAnalysis(n_vars=n)
    A_smooth = smoothed[-1][n:].reshape(n, n)
    residuals_smooth = np.zeros((len(Y) - 1, n))
    for t in range(1, len(Y)):
        y_pred = smoothed[t][:n] + smoothed[t][n:].reshape(n, n) @ Y[t - 1]
        residuals_smooth[t - 1] = Y[t] - y_pred
    Sigma_hat = np.cov(residuals_smooth.T) + np.eye(n) * 0.01
    A_post = np.array([A_smooth + np.random.randn(n, n) * 0.05 for _ in range(500)])
    Sigma_post = np.array([Sigma_hat for _ in range(500)])
    irf_m, irf_l, irf_u = sa.orthogonalized_irf(A_post, periods=irf_periods, Sigma_samples=Sigma_post)

    result = ModelResult(model_name="research", bic=bic_val, log_likelihood=log_lik)
    result.irf_mean = irf_m
    result.irf_lower = irf_l
    result.irf_upper = irf_u

    # Build joint_chain from posterior samples
    joint_chain = []
    for i in range(len(A_post)):
        joint_chain.append({
            "sample_index": i,
            "theta": np.copy(A_post[i].flatten()),
            "sigma": np.copy(Sigma_post[i]),
            "sv_state": None,
            "log_likelihood": None,
        })

    return {"_model": ckf, "_result": result, "_joint_chain": joint_chain}

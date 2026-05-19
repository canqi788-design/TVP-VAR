"""
Stability diagnostics for VAR-style recursive outputs.

The checks here are intentionally lightweight: they flag unstable
posterior dynamics and visibly expanding IRF/forecast intervals without
changing model output by default.
"""

import numpy as np
from tvp_var_framework.core.theta_layout import extract_transition


def spectral_radius(A):
    """Return max(abs(eigenvalue)) for a square transition matrix."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.size == 0:
        return np.nan
    if not np.all(np.isfinite(A)):
        return np.inf
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def stabilize_transition(A, max_radius=0.98):
    """
    Scale A when its spectral radius exceeds max_radius.

    This keeps eigen-directions intact and only shrinks explosive dynamics.
    """
    A = np.asarray(A, dtype=float)
    radius = spectral_radius(A)
    if not np.isfinite(radius) or radius <= max_radius:
        return A.copy(), radius, False
    return A * (max_radius / radius), radius, True


def transition_samples_from_joint_chain(joint_chain, n):
    """Extract A matrices from joint_chain snapshots."""
    samples = []
    for sample in joint_chain or []:
        A_avg = sample.get("A_avg")
        if A_avg is not None:
            A = np.asarray(A_avg, dtype=float)
            if A.shape == (n, n):
                samples.append(A)
                continue

        theta = sample.get("theta")
        layout = sample.get("theta_layout")
        if theta is not None and layout is not None:
            A = extract_transition(theta, layout)
            if A.shape == (n, n):
                samples.append(A)
        elif theta is not None and len(theta) >= n + n * n:
            samples.append(np.asarray(theta[n:n + n * n], dtype=float).reshape(n, n))

    return np.asarray(samples, dtype=float) if samples else np.empty((0, n, n))


def diagnose_transition_stability(A_samples, threshold=1.0):
    """Summarize posterior transition stability from A samples."""
    A_samples = np.asarray(A_samples, dtype=float)
    if A_samples.ndim == 2:
        A_samples = A_samples[None, :, :]
    if A_samples.size == 0:
        return {
            "available": False,
            "warnings": ["未找到可用于稳定性诊断的 VAR 系数样本"],
        }

    radii = np.array([spectral_radius(A) for A in A_samples], dtype=float)
    finite = radii[np.isfinite(radii)]
    if finite.size == 0:
        return {
            "available": False,
            "warnings": ["VAR 系数样本包含非有限值，无法计算谱半径"],
        }

    unstable_mask = radii >= threshold
    near_unit_mask = (radii >= 0.95) & (radii < threshold)
    diagnostics = {
        "available": True,
        "threshold": float(threshold),
        "n_samples": int(radii.size),
        "radius_mean": float(np.nanmean(radii)),
        "radius_median": float(np.nanmedian(radii)),
        "radius_p95": float(np.nanpercentile(radii, 95)),
        "radius_max": float(np.nanmax(radii)),
        "unstable_share": float(np.mean(unstable_mask)),
        "near_unit_share": float(np.mean(near_unit_mask)),
        "warnings": [],
    }

    if diagnostics["unstable_share"] > 0:
        diagnostics["warnings"].append(
            "后验 VAR 系数存在谱半径 >= 1 的样本，多步 IRF/forecast 可能在后几期放大"
        )
    elif diagnostics["near_unit_share"] > 0:
        diagnostics["warnings"].append(
            "后验 VAR 系数接近单位根，多步 IRF/forecast 对后验尾部较敏感"
        )
    return diagnostics


def diagnose_recursive_output_growth(mean, lower=None, upper=None, name="output",
                                     growth_threshold=4.0, interval_threshold=4.0):
    """Flag fast expansion in recursive output levels or intervals."""
    mean = np.asarray(mean, dtype=float)
    diagnostics = {
        "name": name,
        "available": mean.size > 0,
        "growth_ratio": np.nan,
        "interval_growth_ratio": np.nan,
        "max_abs": np.nan,
        "warnings": [],
    }
    if mean.size == 0:
        diagnostics["warnings"].append(f"{name} 为空，无法诊断递推增长")
        return diagnostics

    axes = tuple(range(1, mean.ndim))
    magnitudes = np.max(np.abs(mean), axis=axes) if axes else np.abs(mean)
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    if finite_magnitudes.size:
        diagnostics["max_abs"] = float(np.max(finite_magnitudes))
        baseline_candidates = finite_magnitudes[finite_magnitudes > 1e-12]
        baseline = float(baseline_candidates[0] if baseline_candidates.size else finite_magnitudes[0])
        baseline = max(baseline, 1e-12)
        last_value = float(finite_magnitudes[-1])
        diagnostics["growth_ratio"] = float(last_value / baseline)
        if diagnostics["growth_ratio"] >= growth_threshold:
            diagnostics["warnings"].append(
                f"{name} 后期均值幅度较首期放大 {diagnostics['growth_ratio']:.1f} 倍"
            )

    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        widths = np.max(np.abs(upper - lower), axis=axes) if axes else np.abs(upper - lower)
        finite_widths = widths[np.isfinite(widths)]
        if finite_widths.size:
            baseline_candidates = finite_widths[finite_widths > 1e-12]
            baseline = float(baseline_candidates[0] if baseline_candidates.size else finite_widths[0])
            baseline = max(baseline, 1e-12)
            diagnostics["interval_growth_ratio"] = float(float(finite_widths[-1]) / baseline)
            if diagnostics["interval_growth_ratio"] >= interval_threshold:
                diagnostics["warnings"].append(
                    f"{name} 后期可信区间宽度较首期放大 {diagnostics['interval_growth_ratio']:.1f} 倍"
                )

    if not np.all(np.isfinite(mean)):
        diagnostics["warnings"].append(f"{name} 包含非有限数值")
    return diagnostics


def build_stability_report(A_samples=None, irf=None, forecast=None, threshold=1.0):
    """Build a combined diagnostic dict for report rendering."""
    transition = diagnose_transition_stability(A_samples, threshold=threshold)
    outputs = []
    if irf is not None:
        outputs.append(diagnose_recursive_output_growth(
            irf.get("mean"), irf.get("lower"), irf.get("upper"), name="IRF",
        ))
    if forecast is not None:
        outputs.append(diagnose_recursive_output_growth(
            forecast.get("mean"), forecast.get("lower"), forecast.get("upper"), name="Forecast",
        ))

    warnings = list(transition.get("warnings", []))
    for item in outputs:
        warnings.extend(item.get("warnings", []))

    return {
        "transition": transition,
        "outputs": outputs,
        "warnings": warnings,
        "has_warning": bool(warnings),
    }

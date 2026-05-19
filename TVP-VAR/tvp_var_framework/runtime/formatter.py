"""
Reporting Formatter Layer

Converts raw numeric artifacts (ndarrays, ModelResult) into
display-ready dicts. The report generator only consumes these
pre-formatted structures — no ndarray formatting inside reporting.
"""

import numpy as np


def prepare_data(data_source, vars_to_use, flow_transform="auto",
                 seasonal_transform="none", normalize_data=True,
                 handle_missing="interpolate"):
    """
    Compatibility data-prep entrypoint for compact runtime engines.

    Returns the same tuple as load_company_data:
    Y, quarter_labels, var_names, mean, std.
    """
    from tvp_var_framework.utils.data_loader import load_company_data

    return load_company_data(
        data_source,
        vars_to_use,
        normalize_data=normalize_data,
        handle_missing=handle_missing,
        flow_transform=flow_transform,
        seasonal_transform=seasonal_transform,
    )


def format_irf_for_report(irf_mean, irf_lower, irf_upper, var_names, shock_name):
    """
    Convert IRF arrays to display-ready dict.

    Parameters
    ----------
    irf_mean : ndarray(periods, n) or ndarray(periods, n, n_shocks)
    irf_lower : same shape as irf_mean
    irf_upper : same shape as irf_mean
    var_names : list[str]
    shock_name : str

    Returns
    -------
    list[dict]
        One dict per shock variable (for 3D) or single dict (for 2D).
        Each dict has keys: type, shock_name, var_names, rows (list of dicts).
    """
    sections = []

    if irf_mean.ndim == 3:
        for j, name in enumerate(var_names):
            sections.append(_format_single_irf(
                irf_mean[:, :, j], irf_lower[:, :, j], irf_upper[:, :, j],
                var_names, f"{name} 冲击 ({shock_name})",
            ))
    else:
        sections.append(_format_single_irf(
            irf_mean, irf_lower, irf_upper, var_names, shock_name,
        ))

    return sections


def _format_single_irf(mean, lower, upper, var_names, shock_name):
    """Format a single 2D IRF array into a display dict."""
    periods = mean.shape[0]
    rows = []
    for t in range(periods):
        row = {"period": f"t+{t}"}
        for i, vname in enumerate(var_names):
            row[vname] = {
                "mean": float(mean[t, i]),
                "lower": float(lower[t, i]),
                "upper": float(upper[t, i]),
                "display": f"{mean[t, i]:.4f}[{lower[t, i]:.4f},{upper[t, i]:.4f}]",
            }
        rows.append(row)
    return {
        "type": "irf",
        "shock_name": shock_name,
        "var_names": var_names,
        "rows": rows,
    }


def format_forecast_for_report(pred_mean, pred_lower, pred_upper, labels, var_names):
    """
    Convert forecast arrays to display-ready dict.

    Returns
    -------
    dict
        With keys: type, var_names, rows (list of dicts).
    """
    n_steps = pred_mean.shape[0]
    rows = []
    for s in range(n_steps):
        label = labels[s] if s < len(labels) else f"t+{s+1}"
        row = {"label": label}
        for i, vname in enumerate(var_names):
            row[vname] = {
                "mean": float(pred_mean[s, i]),
                "lower": float(pred_lower[s, i]),
                "upper": float(pred_upper[s, i]),
                "display": f"{pred_mean[s, i]:.4f}[{pred_lower[s, i]:.4f},{pred_upper[s, i]:.4f}]",
            }
        rows.append(row)
    return {
        "type": "forecast",
        "var_names": var_names,
        "rows": rows,
    }


def format_posterior_for_report(summary_dict, var_names):
    """
    Convert posterior summary dict to display-ready format.

    Returns
    -------
    dict
        With keys: type, rows (list of dicts).
    """
    rows = []
    for param, st in summary_dict.items():
        ci = st.get("ci_95", (None, None))
        rows.append({
            "param": param,
            "mean": st.get("mean"),
            "std": st.get("std"),
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "display_mean": f"{st['mean']:.4f}" if st.get("mean") is not None else "",
            "display_ci": f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "",
        })
    return {
        "type": "posterior_summary",
        "rows": rows,
    }

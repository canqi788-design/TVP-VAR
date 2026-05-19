"""
TVP-VAR 平稳性检验

ADF 和 KPSS 检验 + 自动差分处理。
纯 numpy 实现，不依赖 statsmodels。

参考:
  - Dickey & Fuller (1979) "Distribution of the Estimators for AR Time Series"
  - Kwiatkowski et al. (1992) "Testing the Null Hypothesis of Stationarity"
  - MacKinnon (1994) "Approximate Asymptotic Distribution Functions for Unit Root Tests"
"""

import numpy as np
import logging

logger = logging.getLogger("tvp_var")


def _clean_series(series):
    """
    Clean NaN/Inf values in a 1D series with linear interpolation.

    np.interp also extends edge gaps with the nearest finite endpoint, which
    prevents missing values from leaking into ADF/KPSS regressions.
    """
    s = np.asarray(series, dtype=float).copy()
    n = len(s)
    if n == 0:
        return s

    s[~np.isfinite(s)] = np.nan
    nans = np.isnan(s)
    if not np.any(nans):
        return s
    if np.all(nans):
        return np.zeros_like(s)

    x = np.arange(n)
    valid = ~nans
    s[nans] = np.interp(x[nans], x[valid], s[valid])
    return s


def adf_test(series, max_lag=None, significance=0.05):
    series = _clean_series(series)
    T = len(series)
    if T < 5:
        return {"statistic": np.nan, "critical_values": {}, "p_value": 1.0, "is_stationary": False}

    if max_lag is None:
        max_lag = int(np.ceil(12 * (T / 100) ** 0.25))
    max_lag = min(max_lag, T // 3)

    # The ADF regression has k = 2 + max_lag regressors and
    # N = T - 1 - max_lag observations. Cap max_lag so OLS keeps
    # at least one residual degree of freedom on short quarterly samples.
    max_lag_allowed = min(T // 3, (T - 4) // 2, T - 5)
    if max_lag_allowed < 0:
        max_lag = 0
    else:
        max_lag = max(0, min(int(max_lag), int(max_lag_allowed)))

    dy = np.diff(series)
    y_lag = series[:-1]
    N = len(dy) - max_lag
    if N < 3:
        max_lag = 0
        N = len(dy)

    X = np.zeros((N, 2 + max_lag))
    X[:, 0] = 1.0
    X[:, 1] = y_lag[max_lag:]
    for lag in range(1, max_lag + 1):
        X[:, 1 + lag] = dy[max_lag - lag:-lag] if lag < max_lag else dy[max_lag - lag:N + max_lag - lag]

    y_reg = dy[max_lag:]
    if len(y_reg) != N:
        y_reg = y_reg[:N]
    X = X[:len(y_reg)]

    try:
        beta_hat, residuals = _ols(X, y_reg)
    except Exception:
        return {"statistic": np.nan, "critical_values": {}, "p_value": 1.0, "is_stationary": False}

    se_beta = _se_beta(X, residuals)
    if se_beta < 1e-30:
        return {"statistic": 0.0, "critical_values": {}, "p_value": 1.0, "is_stationary": False}

    adf_stat = beta_hat[1] / se_beta

    critical_values = _macKinnon_critical_values(T, regression="c")
    p_value = _macKinnon_pvalue(adf_stat, T, regression="c")

    is_stationary = adf_stat < critical_values.get(f"{significance*100:.0f}%", -2.86)

    return {
        "statistic": float(adf_stat) if np.isfinite(adf_stat) else np.nan,
        "critical_values": critical_values,
        "p_value": float(p_value) if np.isfinite(p_value) else 1.0,
        "is_stationary": bool(is_stationary),
    }


def kpss_test(series, regression='c', significance=0.05):
    series = _clean_series(series)
    T = len(series)
    if T < 5:
        return {"statistic": np.nan, "critical_values": {}, "p_value": 1.0, "is_stationary": True}

    if regression == 'ct':
        t = np.arange(T)
        X = np.column_stack([np.ones(T), t])
        beta = np.linalg.lstsq(X, series, rcond=None)[0]
        residuals = series - X @ beta
    else:
        residuals = series - np.mean(series)

    cumsum = np.cumsum(residuals)

    gamma0 = np.sum(residuals ** 2) / T
    bandwidth = int(np.ceil(4 * (T / 100) ** 0.25))
    bandwidth = min(bandwidth, T - 2)
    if bandwidth < 1:
        bandwidth = 1
    gamma_sum = 0.0
    for j in range(1, bandwidth + 1):
        gamma_j = np.sum(residuals[j:] * residuals[:-j]) / T
        weight = 1.0 - j / (bandwidth + 1)
        gamma_sum += weight * gamma_j

    sigma2 = gamma0 + 2 * gamma_sum

    if sigma2 < 1e-30:
        return {"statistic": 0.0, "critical_values": {}, "p_value": 1.0, "is_stationary": True}

    eta = np.sum(cumsum ** 2) / (T ** 2 * sigma2)

    if regression == 'ct':
        critical_values = {"1%": 0.216, "2.5%": 0.176, "5%": 0.146, "10%": 0.119}
    else:
        critical_values = {"1%": 0.739, "2.5%": 0.574, "5%": 0.463, "10%": 0.347}

    cv_key = f"{significance*100:.0f}%"
    cv = critical_values.get(cv_key, 0.463)
    if eta > critical_values["1%"]:
        p_value = 0.001
    elif eta < critical_values["10%"]:
        p_value = 0.1
    else:
        cvs = sorted(critical_values.items(), key=lambda x: x[1])
        for i in range(len(cvs) - 1):
            if cvs[i][1] <= eta <= cvs[i + 1][1]:
                p_vals = [0.1, 0.05, 0.025, 0.01]
                idx = [k for k, v in cvs].index(cvs[i][0])
                p_value = p_vals[idx] if idx < len(p_vals) else 0.05
                break
        else:
            p_value = 0.05

    is_stationary = eta < cv

    return {
        "statistic": float(eta) if np.isfinite(eta) else np.nan,
        "critical_values": critical_values,
        "p_value": float(p_value) if np.isfinite(p_value) else 1.0,
        "is_stationary": bool(is_stationary),
    }


def _evaluate_stationarity(series, test='adf', significance=0.05):
    """Run one or two tests and return a conservative stationarity decision."""
    if test == 'kpss':
        kpss_result = kpss_test(series, significance=significance)
        return {
            "decision": kpss_result["is_stationary"],
            "primary_result": kpss_result,
            "combined_result": {"kpss": kpss_result},
        }

    if test == 'both':
        adf_result = adf_test(series, significance=significance)
        kpss_result = kpss_test(series, significance=significance)
        decision = adf_result["is_stationary"] and kpss_result["is_stationary"]
        return {
            "decision": decision,
            "primary_result": {
                "statistic": adf_result["statistic"],
                "critical_values": adf_result["critical_values"],
                "p_value": adf_result["p_value"],
                "is_stationary": decision,
                "adf_is_stationary": adf_result["is_stationary"],
                "kpss_is_stationary": kpss_result["is_stationary"],
            },
            "combined_result": {"adf": adf_result, "kpss": kpss_result},
        }

    adf_result = adf_test(series, significance=significance)
    return {
        "decision": adf_result["is_stationary"],
        "primary_result": adf_result,
        "combined_result": {"adf": adf_result},
    }


def _difference_series(series, method="regular", period=1):
    """Apply one configured differencing step to a 1D series."""
    if method == "seasonal":
        return series[period:] - series[:-period]
    return np.diff(series)


def _lag_aligned_series(series, method="regular", period=1):
    """Drop the observations lost by one differencing step without differencing."""
    if method == "seasonal":
        return series[period:]
    return series[1:]


def auto_difference(Y, var_names=None, max_d=2, test='adf', significance=0.05,
                    log_transform=False, method="regular", period=1):
    """
    自动差分处理。

    Parameters
    ----------
    log_transform : bool
        若为 True，先对数据取 ln，再做一阶差分（增长率）。
        适用于金额类指标，可避免二阶差分过度压缩信号。
    method : {"regular", "seasonal"}
        差分算子。regular 为连续一阶差分；seasonal 为周期同比差分。
    period : int
        seasonal 差分周期，例如季度数据为 4。
    """
    Y = np.asarray(Y, dtype=float).copy()
    T, n = Y.shape
    if var_names is None:
        var_names = [f"x{i}" for i in range(n)]
    method = str(method or "regular").lower()
    if method in {"first", "diff", "difference", "continuous"}:
        method = "regular"
    if method not in {"regular", "seasonal"}:
        raise ValueError(f"未知差分方法: {method}")
    period = int(period or 1)
    if period < 1:
        raise ValueError("period 必须 >= 1")
    if method == "seasonal" and T <= period:
        raise ValueError(f"seasonal 差分需要样本长度大于 period: T={T}, period={period}")
    if method == "seasonal":
        max_d = min(int(max_d), 1)

    for j in range(n):
        Y[:, j] = _clean_series(Y[:, j])

    # Log 变换: ln(Y) → 一阶差分 = 增长率
    if log_transform:
        # 处理非正值: 平移使最小值 > 0
        for j in range(n):
            col_min = np.nanmin(Y[:, j])
            if col_min <= 0:
                Y[:, j] = Y[:, j] - col_min + 1.0
        Y = np.log(Y)
        max_d = 1  # log 后只需一阶差分

    d_orders = np.zeros(n, dtype=int)
    test_results = {}
    for j in range(n):
        col = Y[:, j]
        d = 0
        for attempt in range(max_d + 1):
            assessment = _evaluate_stationarity(col, test=test, significance=significance)
            is_stationary = assessment["decision"]
            result = assessment["primary_result"]
            result["tests"] = assessment["combined_result"]
            test_results[f"{var_names[j]}_d{d}"] = result

            if is_stationary or d >= max_d:
                break
            step_loss = period if method == "seasonal" else 1
            if len(col) <= step_loss:
                break
            col = _difference_series(col, method=method, period=period)
            d += 1

        d_orders[j] = d

    max_d_applied = int(np.max(d_orders))
    if max_d_applied > 0:
        Y_diff = Y.copy()
        for d_step in range(max_d_applied):
            step_loss = period if method == "seasonal" else 1
            if Y_diff.shape[0] <= step_loss:
                break
            new_Y = np.zeros((Y_diff.shape[0] - step_loss, Y_diff.shape[1]))
            for j in range(n):
                if d_orders[j] > d_step:
                    new_Y[:, j] = _difference_series(Y_diff[:, j], method=method, period=period)
                else:
                    new_Y[:, j] = _lag_aligned_series(Y_diff[:, j], method=method, period=period)
            Y_diff = new_Y
    else:
        Y_diff = Y.copy()

    warnings = []
    if T < 10:
        warning = "样本量过短，ADF/KPSS 结果可能不稳定，报告中的平稳性结论仅供参考"
        logger.warning(warning)
        warnings.append(warning)
    if test == 'both':
        warnings.append("已启用 ADF+KPSS 联合判定，仅当两者同时支持平稳时才停止差分")
    if method == "seasonal":
        warnings.append(f"已启用 seasonal 周期差分，period={period}")

    return {
        "Y_diff": Y_diff,
        "d_orders": d_orders,
        "test_results": test_results,
        "original_Y": Y,
        "warnings": warnings,
    }


class StationarityAnalyzer:
    def __init__(self, significance=0.05, max_d=2, test='both', log_transform=False,
                 method="regular", period=1):
        self.significance = significance
        self.max_d = max_d
        self.test = test
        self.log_transform = log_transform
        self.method = method
        self.period = period
        self.results = {}
        self._d_orders = None
        self._Y_diff = None
        self._original_Y = None
        self._var_names = None
        self.warnings = []

    def analyze(self, Y, var_names=None):
        Y = np.asarray(Y, dtype=float)
        self._original_Y = Y
        self._var_names = var_names or [f"x{i}" for i in range(Y.shape[1])]

        result = auto_difference(
            Y, var_names=self._var_names,
            max_d=self.max_d, test=self.test,
            significance=self.significance,
            log_transform=self.log_transform,
            method=self.method,
            period=self.period,
        )
        self._Y_diff = result["Y_diff"]
        self._d_orders = result["d_orders"]
        self.results = result["test_results"]
        self.warnings = result.get("warnings", [])

        logger.info(f"平稳性检验完成: {self.test.upper()}, 显著性={self.significance}")
        for j, (name, d) in enumerate(zip(self._var_names, self._d_orders)):
            logger.info(f"  {name}: 差分阶数 d={d}")

        return self

    def fit(self, Y, var_names=None):
        """Backward-compatible alias used by newer compact runtime drafts."""
        return self.analyze(Y, var_names=var_names)

    def get_differenced_data(self):
        if self._Y_diff is None:
            raise ValueError("请先调用 analyze()")
        return self._Y_diff

    @property
    def transformed_data(self):
        return self.get_differenced_data()

    def get_d_orders(self):
        return self._d_orders

    @property
    def d_orders(self):
        return self._d_orders

    def report(self, var_names=None):
        names = var_names or self._var_names
        if names is None:
            names = [f"x{i}" for i in range(len(self._d_orders))]

        lines = []
        lines.append("平稳性检验报告")
        lines.append("-" * 50)
        lines.append(f"检验方法: {self.test.upper()}")
        lines.append(f"显著性水平: {self.significance}")
        lines.append(f"最大差分阶数: {self.max_d}")
        lines.append(f"差分算子: {self.method} (period={self.period})")
        lines.append("")

        for j, name in enumerate(names):
            d = self._d_orders[j]
            if d == 0:
                lines.append(f"  {name}: 平稳 (无需差分)")
            else:
                lines.append(f"  {name}: 非平稳, 差分 {d} 阶后平稳")

        return "\n".join(lines)


def _ols(X, y):
    ridge = 1e-8 * np.eye(X.shape[1])
    cov_inv = np.linalg.pinv(X.T @ X + ridge)
    beta = cov_inv @ X.T @ y
    residuals = y - X @ beta
    return beta, residuals


def _se_beta(X, residuals):
    n, k = X.shape
    if n <= k:
        return 1e10
    sigma2 = np.sum(residuals ** 2) / (n - k)
    try:
        cov_beta = sigma2 * np.linalg.pinv(X.T @ X + 1e-8 * np.eye(k))
        return np.sqrt(max(cov_beta[1, 1], 1e-30))
    except Exception:
        return 1e10


def _macKinnon_critical_values(T, regression="c"):
    t_inf = 1.0 / T if T > 0 else 0.01
    if regression == "c":
        cv_1 = -3.43 - 5.99 * t_inf - 28.5 * t_inf ** 2
        cv_5 = -2.86 - 2.77 * t_inf - 8.36 * t_inf ** 2
        cv_10 = -2.57 - 1.52 * t_inf - 4.05 * t_inf ** 2
    else:
        cv_1 = -3.96 - 8.35 * t_inf - 47.4 * t_inf ** 2
        cv_5 = -3.41 - 4.98 * t_inf - 21.0 * t_inf ** 2
        cv_10 = -3.13 - 3.57 * t_inf - 13.0 * t_inf ** 2

    return {"1%": cv_1, "5%": cv_5, "10%": cv_10}


def _macKinnon_pvalue(adf_stat, T, regression="c"):
    """MacKinnon p-value — 在临界值之间线性插值 (log-p 尺度)"""
    if not np.isfinite(adf_stat):
        return 1.0
    cv = _macKinnon_critical_values(T, regression)
    # 从极端到宽松: 1%, 5%, 10%
    points = [
        (cv["1%"], 0.01),
        (cv["5%"], 0.05),
        (cv["10%"], 0.10),
    ]
    if adf_stat < cv["1%"]:
        # 比 1% 更极端，用对数外推
        slope = (np.log(0.01) - np.log(0.05)) / (cv["1%"] - cv["5%"])
        log_p = np.log(0.01) + slope * (adf_stat - cv["1%"])
        return float(np.clip(np.exp(log_p), 1e-6, 0.01))
    elif adf_stat < cv["5%"]:
        frac = (adf_stat - cv["1%"]) / (cv["5%"] - cv["1%"])
        log_p = np.log(0.01) + frac * (np.log(0.05) - np.log(0.01))
        return float(np.exp(log_p))
    elif adf_stat < cv["10%"]:
        frac = (adf_stat - cv["5%"]) / (cv["10%"] - cv["5%"])
        log_p = np.log(0.05) + frac * (np.log(0.10) - np.log(0.05))
        return float(np.exp(log_p))
    else:
        # 超过 10% 临界值，用对数外推
        slope = (np.log(0.10) - np.log(0.05)) / (cv["10%"] - cv["5%"])
        log_p = np.log(0.10) + slope * (adf_stat - cv["10%"])
        return float(np.clip(np.exp(log_p), 0.10, 0.99))

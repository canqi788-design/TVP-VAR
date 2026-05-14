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


def adf_test(series, max_lag=None, significance=0.05):
    series = np.asarray(series, dtype=float)
    T = len(series)
    if T < 10:
        return {"statistic": np.nan, "critical_values": {}, "p_value": 1.0, "is_stationary": False}

    if max_lag is None:
        max_lag = int(np.ceil(12 * (T / 100) ** 0.25))
    max_lag = min(max_lag, T // 3)

    dy = np.diff(series)
    y_lag = series[:-1]
    N = len(dy) - max_lag
    if N < 5:
        return {"statistic": np.nan, "critical_values": {}, "p_value": 1.0, "is_stationary": False}

    X = np.zeros((N, 2 + max_lag))
    X[:, 0] = 1.0
    X[:, 1] = y_lag[max_lag:]
    for lag in range(1, max_lag + 1):
        X[:, 1 + lag] = dy[max_lag - lag:-lag] if lag < max_lag + 1 else dy[max_lag - lag:N + max_lag - lag]

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
        "statistic": float(adf_stat),
        "critical_values": critical_values,
        "p_value": float(p_value),
        "is_stationary": bool(is_stationary),
    }


def kpss_test(series, regression='c', significance=0.05):
    series = np.asarray(series, dtype=float)
    T = len(series)
    if T < 10:
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
        "statistic": float(eta),
        "critical_values": critical_values,
        "p_value": float(p_value),
        "is_stationary": bool(is_stationary),
    }


def auto_difference(Y, var_names=None, max_d=2, test='adf', significance=0.05, log_transform=False):
    """
    自动差分处理。

    Parameters
    ----------
    log_transform : bool
        若为 True，先对数据取 ln，再做一阶差分（增长率）。
        适用于金额类指标，可避免二阶差分过度压缩信号。
    """
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    if var_names is None:
        var_names = [f"x{i}" for i in range(n)]

    # Log 变换: ln(Y) → 一阶差分 = 增长率
    if log_transform:
        # 处理非正值: 平移使最小值 > 0
        for j in range(n):
            col_min = Y[:, j].min()
            if col_min <= 0:
                Y[:, j] = Y[:, j] - col_min + 1.0
        Y = np.log(Y)
        max_d = 1  # log 后只需一阶差分

    d_orders = np.zeros(n, dtype=int)
    test_results = {}
    Y_current = Y.copy()

    for j in range(n):
        col = Y[:, j]
        d = 0
        for attempt in range(max_d + 1):
            if test == 'kpss':
                result = kpss_test(col, significance=significance)
            else:
                result = adf_test(col, significance=significance)

            is_stationary = result["is_stationary"]
            test_results[f"{var_names[j]}_d{d}"] = result

            if is_stationary or d >= max_d:
                break
            col = np.diff(col)
            d += 1

        d_orders[j] = d

    max_d_applied = int(np.max(d_orders))
    if max_d_applied > 0:
        Y_diff = Y.copy()
        for d_step in range(max_d_applied):
            new_Y = np.zeros((Y_diff.shape[0] - 1, Y_diff.shape[1]))
            for j in range(n):
                if d_orders[j] > d_step:
                    new_Y[:, j] = np.diff(Y_diff[:, j])
                else:
                    new_Y[:, j] = Y_diff[1:, j]
            Y_diff = new_Y
    else:
        Y_diff = Y.copy()

    return {
        "Y_diff": Y_diff,
        "d_orders": d_orders,
        "test_results": test_results,
        "original_Y": Y,
    }


class StationarityAnalyzer:
    def __init__(self, significance=0.05, max_d=2, test='adf', log_transform=False):
        self.significance = significance
        self.max_d = max_d
        self.test = test
        self.log_transform = log_transform
        self.results = {}
        self._d_orders = None
        self._Y_diff = None
        self._original_Y = None
        self._var_names = None

    def analyze(self, Y, var_names=None):
        Y = np.asarray(Y, dtype=float)
        self._original_Y = Y
        self._var_names = var_names or [f"x{i}" for i in range(Y.shape[1])]

        result = auto_difference(
            Y, var_names=self._var_names,
            max_d=self.max_d, test=self.test,
            significance=self.significance,
            log_transform=self.log_transform,
        )
        self._Y_diff = result["Y_diff"]
        self._d_orders = result["d_orders"]
        self.results = result["test_results"]

        logger.info(f"平稳性检验完成: {self.test.upper()}, 显著性={self.significance}")
        for j, (name, d) in enumerate(zip(self._var_names, self._d_orders)):
            logger.info(f"  {name}: 差分阶数 d={d}")

        return self

    def get_differenced_data(self):
        if self._Y_diff is None:
            raise ValueError("请先调用 analyze()")
        return self._Y_diff

    def get_d_orders(self):
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
        lines.append("")

        for j, name in enumerate(names):
            d = self._d_orders[j]
            if d == 0:
                lines.append(f"  {name}: 平稳 (无需差分)")
            else:
                lines.append(f"  {name}: 非平稳, 差分 {d} 阶后平稳")

        return "\n".join(lines)


def _ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    return beta, residuals


def _se_beta(X, residuals):
    n, k = X.shape
    if n <= k:
        return 1e10
    sigma2 = np.sum(residuals ** 2) / (n - k)
    try:
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
        return np.sqrt(max(cov_beta[1, 1], 1e-30))
    except np.linalg.LinAlgError:
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

"""
贝叶斯 TVP-VAR 扩展模块
包含：边际似然计算、贝叶斯模型比较(BIC)、后验采样、贝叶斯模型平均(BMA)、
先验敏感性分析、MCMC 采样 (FFBS + Metropolis-Hastings)
"""

import logging
import numpy as np
from .analyst import TVP_VAR_Analyst
from tvp_var_framework.core.theta_layout import split_theta, extract_transition
from tvp_var_framework.utils.stability import stabilize_transition

logger = logging.getLogger("tvp_var")


def _ar1_residual_scales(Y):
    """Estimate per-equation innovation scales from single-variable AR(1) fits."""
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[1]
    scales = np.zeros(n, dtype=float)
    for j in range(n):
        y = Y[:, j].astype(float).copy()
        y[~np.isfinite(y)] = np.nan
        if np.all(np.isnan(y)):
            scales[j] = 1.0
            continue
        if np.any(np.isnan(y)):
            x = np.arange(len(y))
            valid = ~np.isnan(y)
            y[~valid] = np.interp(x[~valid], x[valid], y[valid])

        if len(y) < 3:
            scales[j] = np.std(y) if np.std(y) > 1e-12 else 1.0
            continue

        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        target = y[1:]
        try:
            beta = np.linalg.lstsq(X, target, rcond=None)[0]
            resid = target - X @ beta
            scale = np.sqrt(np.mean(resid ** 2))
        except np.linalg.LinAlgError:
            scale = np.std(np.diff(y))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.std(y)
        scales[j] = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    return scales


def minnesota_prior_cov_diag(n_vars, n_exog=0, gamma_1=0.1, gamma_2=0.1,
                             intercept_var=10.0, exog_var=1.0, lag_order=1,
                             equation_scales=None):
    """
    Build equation-by-equation Minnesota prior covariance diagonal.

    Own lag terms use gamma_1. Cross-variable lag terms use gamma_2 scaled by
    Litterman's sigma_i / sigma_j ratio, where sigma_i is the AR(1) residual
    scale of the response equation and sigma_j is the residual scale of the
    predictor variable. The current framework is VAR(1), so lag_order defaults
    to 1 while keeping the formula lag-aware for later generalized layouts.
    """
    n = int(n_vars)
    m = int(n_exog)
    lag = max(int(lag_order), 1)
    components_per_equation = 1 + n + m
    diag = np.zeros(n * components_per_equation, dtype=float)

    intercept_var = float(intercept_var)
    if intercept_var <= 0:
        raise ValueError("intercept_var must be positive")
    if equation_scales is None:
        scales = np.ones(n, dtype=float)
    else:
        scales = np.asarray(equation_scales, dtype=float).reshape(-1)
        if len(scales) != n:
            raise ValueError(f"equation_scales length mismatch: expected {n}, got {len(scales)}")
        scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)

    for i in range(n):
        row_start = i * components_per_equation
        diag[row_start] = intercept_var
        for j in range(n):
            idx = row_start + 1 + j
            if i == j:
                diag[idx] = (float(gamma_1) / lag) ** 2
            else:
                scale_ratio = scales[i] / max(scales[j], 1e-12)
                diag[idx] = ((float(gamma_2) * scale_ratio) / lag) ** 2
        if m:
            diag[row_start + 1 + n:row_start + components_per_equation] = float(exog_var)

    return diag


class BayesianTVP_VAR:
    """贝叶斯 TVP-VAR 分析器，封装 TVP_VAR_Analyst 并添加贝叶斯功能"""

    def __init__(self, n_vars=3, n_exog=0, q=1e-5, r=1e-3, prior_mean=None, prior_cov=None,
                 gamma_1=0.1, gamma_2=0.1, intercept_var=10.0):
        """
        参数:
            n_vars: 变量数
            q: 过程噪声方差 (状态转移噪声)
            r: 观测噪声方差
            prior_mean: theta 的先验均值 (默认截距0, 系数0.5)
            prior_cov: theta 的先验协方差 (默认单位阵)
        """
        self.n_vars = n_vars
        self.n_exog = n_exog
        self.q = q
        self.r = r
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.intercept_var = intercept_var
        self.analyst = TVP_VAR_Analyst(n_vars=n_vars, n_exog=n_exog, q=q, r=r)
        self.meta = self.analyst.meta
        if prior_cov is None:
            V_prior_diag = minnesota_prior_cov_diag(
                n_vars=n_vars,
                n_exog=n_exog,
                gamma_1=gamma_1,
                gamma_2=gamma_2,
                intercept_var=intercept_var,
            )
            self.analyst.P = np.diag(V_prior_diag)
            logger.info(f"Minnesota prior shrinkage: gamma_1={gamma_1}, gamma_2={gamma_2}")
        logger.info(f"V_prior_diag.shape={self.analyst.P.diagonal().shape}")
        logger.info(f"theta_layout={self.meta}")

        # 覆盖先验均值
        if prior_mean is not None:
            self.analyst.theta = prior_mean.copy()

        # 覆盖先验协方差
        if prior_cov is not None:
            self.analyst.P = prior_cov.copy()

        # 存储每步的边际似然
        self.log_likelihoods = []

    def fit(self, Y, X_exog=None):
        """
        拟合模型，Y 为 (T, n) 的数据矩阵，每行是一个时间点的观测
        返回: 模型实例 (方便链式调用)
        """
        Y = np.asarray(Y, dtype=float)
        self.log_likelihoods = []
        self.analyst = TVP_VAR_Analyst(n_vars=self.n_vars, n_exog=self.n_exog, q=self.q, r=self.r)
        self.meta = self.analyst.meta
        equation_scales = _ar1_residual_scales(Y)
        V_prior_diag = minnesota_prior_cov_diag(
            n_vars=self.n_vars,
            n_exog=self.n_exog,
            gamma_1=self.gamma_1,
            gamma_2=self.gamma_2,
            intercept_var=self.intercept_var,
            equation_scales=equation_scales,
        )
        self.analyst.P = np.diag(V_prior_diag)
        logger.info(f"Minnesota AR(1) residual scales={equation_scales}")
        logger.info(f"V_prior_diag.shape={self.analyst.P.diagonal().shape}")
        if X_exog is None and self.n_exog:
            raise ValueError("VARX 模型需要显式传入 X_exog 历史数据")
        if X_exog is not None:
            X_exog = np.asarray(X_exog, dtype=float)

        for t in range(1, len(Y)):
            y_prev = Y[t - 1]
            y_true = Y[t]
            x_t = None if X_exog is None else X_exog[t]

            Z = self.analyst._build_Z(y_prev, x_t)

            # 预测
            theta_pred = self.analyst.theta.copy()
            P_pred = self.analyst.P + self.analyst.Q
            y_pred = Z @ theta_pred

            # 创新
            v = y_true - y_pred
            S = Z @ P_pred @ Z.T + self.analyst.R

            # 计算对数边际似然: log p(y_t | y_{1:t-1})
            sign, log_det = np.linalg.slogdet(S)
            log_lik = -0.5 * (self.n_vars * np.log(2 * np.pi) + log_det + v @ np.linalg.solve(S, v))
            self.log_likelihoods.append(log_lik)

            # Kalman 更新
            self.analyst.update(y_prev, y_true, x_t)

        return self

    @property
    def log_marginal_likelihood(self):
        """对数边际似然 = 各步对数预测似然之和"""
        return sum(self.log_likelihoods)

    @property
    def bic(self):
        """贝叶斯信息准则 (越小越好)"""
        T = len(self.log_likelihoods)
        k = self.analyst.k  # 自由参数个数
        return -2 * self.log_marginal_likelihood + k * np.log(T)

    @property
    def aic(self):
        """赤池信息准则 (越小越好)"""
        k = self.analyst.k
        return -2 * self.log_marginal_likelihood + 2 * k

    def sample_posterior(self, n_samples=1000):
        """
        从最终时刻的参数后验分布中采样
        后验为高斯: theta_T | Y ~ N(theta_T, P_T)
        返回: (n_samples, k) 的采样矩阵
        """
        theta_mean = self.analyst.theta.copy()
        theta_cov = self.analyst.P.copy()
        # 确保对称
        theta_cov = (theta_cov + theta_cov.T) / 2

        samples = np.random.multivariate_normal(theta_mean, theta_cov, size=n_samples)
        return samples

    def fit_and_sample(self, Y, X_exog=None, n_samples=1000, n_burn=0, params=None):
        """
        Compatibility helper for compact runtime engines.

        Fits the Kalman Bayesian TVP-VAR path, then draws an analytical
        posterior joint_chain with theta, sigma, A_avg, and theta_layout.
        n_burn is accepted for API compatibility; analytical posterior draws
        are iid and do not use a burn-in phase.
        """
        params = params or {}
        if "gamma_1" in params:
            self.gamma_1 = params["gamma_1"]
        if "gamma_2" in params:
            self.gamma_2 = params["gamma_2"]
        if "intercept_var" in params:
            self.intercept_var = params["intercept_var"]

        self.fit(Y, X_exog=X_exog)
        theta_samples = self.sample_posterior(n_samples=int(n_samples))

        innovations = [
            np.asarray(record.get("innovation"), dtype=float)
            for record in self.analyst.history
            if record.get("innovation") is not None
        ]
        if len(innovations) > 1:
            innov_arr = np.vstack(innovations)
            sigma = np.cov(innov_arr.T) + np.eye(self.n_vars) * 1e-8
        else:
            sigma = np.copy(self.analyst.R)

        if self.analyst.history:
            A_avg = np.mean([record["A"] for record in self.analyst.history], axis=0)
        else:
            _, A_avg, _ = self._theta_parts(self.analyst.theta)

        joint_chain = []
        for idx, theta in enumerate(theta_samples):
            _, A, _ = self._theta_parts(theta)
            joint_chain.append({
                "sample_index": idx,
                "theta": np.copy(theta),
                "sigma": np.copy(sigma),
                "sv_state": None,
                "A_avg": np.copy(A),
                "A_path_mean": np.copy(A_avg),
                "theta_layout": dict(self.analyst.meta),
                "log_likelihood": None,
                "chain_source": "analytic_iid",
            })
        return joint_chain

    def _theta_parts(self, theta):
        return split_theta(theta, self.analyst.meta)

    def _attach_layout(self, theta):
        return {"theta": np.copy(theta), "theta_layout": self.analyst.meta}

    def sample_trajectory(self, n_samples=100):
        """
        从各时间步的后验中采样参数轨迹
        返回: list of (n_samples, k) 数组，每个时间步一个
        """
        trajectories = []
        for record in self.analyst.history:
            c = record["c"]
            A = record["A"]
            B = record.get("B")
            pieces = [c.reshape(self.n_vars, 1), A, B if B is not None else np.zeros((self.n_vars, self.n_exog))]
            theta = np.hstack(pieces).reshape(-1)
            # 使用最终的 P 作为近似 (简化)
            samples = np.random.multivariate_normal(theta, self.analyst.P, size=n_samples)
            trajectories.append(samples)
        return trajectories

    def posterior_summary(self, var_names=None):
        """
        后验参数摘要统计
        返回: dict，包含每个参数的均值、标准差、95% 置信区间
        """
        samples = self.sample_posterior(n_samples=5000)
        n = self.n_vars

        if var_names is None:
            var_names = [f"y{i+1}" for i in range(n)]

        summary = {}

        # 截距
        components_per_equation = 1 + n + self.n_exog
        for i, name in enumerate(var_names):
            idx = i * components_per_equation
            s = samples[:, idx]
            summary[f"c_{name}"] = {
                "mean": np.mean(s),
                "std": np.std(s),
                "ci_95": (np.percentile(s, 2.5), np.percentile(s, 97.5))
            }

        # 系数矩阵
        for i, name_i in enumerate(var_names):
            for j, name_j in enumerate(var_names):
                idx = i * components_per_equation + 1 + j
                s = samples[:, idx]
                summary[f"A_{name_i}<-{name_j}"] = {
                    "mean": np.mean(s),
                    "std": np.std(s),
                    "ci_95": (np.percentile(s, 2.5), np.percentile(s, 97.5))
                }

        for i, name_i in enumerate(var_names):
            for j in range(self.n_exog):
                idx = i * components_per_equation + 1 + n + j
                s = samples[:, idx]
                summary[f"B_{name_i}<-x{j+1}"] = {
                    "mean": np.mean(s),
                    "std": np.std(s),
                    "ci_95": (np.percentile(s, 2.5), np.percentile(s, 97.5))
                }

        return summary

    def _future_exog_path(self, steps, X_exog_history=None, X_exog_future=None):
        if self.n_exog == 0:
            return None
        if X_exog_future is not None:
            future = np.asarray(X_exog_future, dtype=float)
            if future.ndim == 1:
                future = future.reshape(-1, self.n_exog)
            if future.shape[1] != self.n_exog:
                raise ValueError(f"X_exog_future 维度不匹配: expected {self.n_exog}, got {future.shape[1]}")
            if len(future) < steps:
                pad = np.tile(future[-1], (steps - len(future), 1))
                future = np.vstack([future, pad])
            return future[:steps]
        if X_exog_history is None:
            raise ValueError("VARX 预测需要 X_exog_history 或 X_exog_future，禁止从内生 Y 推断外生变量")
        hist = np.asarray(X_exog_history, dtype=float)
        if hist.ndim == 1:
            hist = hist.reshape(-1, self.n_exog)
        if hist.shape[1] != self.n_exog:
            raise ValueError(f"X_exog_history 维度不匹配: expected {self.n_exog}, got {hist.shape[1]}")
        return np.tile(hist[-1], (steps, 1))

    def predict_samples(self, Y, steps=1, n_samples=1000,
                        X_exog_history=None, X_exog_future=None,
                        enforce_stability=False, stability_max_radius=0.98):
        """
        从后验预测分布中采样
        返回: (n_samples, steps, n) 的预测样本
        """
        theta_samples = self.sample_posterior(n_samples)
        n = self.n_vars
        m = self.n_exog
        R = self.analyst.R

        all_preds = np.zeros((n_samples, steps, n))
        X_future = self._future_exog_path(
            steps, X_exog_history=X_exog_history, X_exog_future=X_exog_future
        )

        for i, theta in enumerate(theta_samples):
            y_current = Y[-1].copy()
            c, A, B = self._theta_parts(theta)
            if enforce_stability:
                A, _, _ = stabilize_transition(A, max_radius=stability_max_radius)
            for s in range(steps):
                x_s = None
                if m:
                    x_s = X_future[s]
                y_pred = c + A @ y_current
                if m:
                    y_pred = y_pred + B @ x_s
                # 加入观测噪声
                noise = np.random.multivariate_normal(np.zeros(n), R)
                all_preds[i, s] = y_pred + noise
                y_current = y_pred

        return all_preds

    def predict_interval(self, Y, steps=1, n_samples=1000,
                         X_exog_history=None, X_exog_future=None,
                         enforce_stability=False, stability_max_radius=0.98):
        """
        带置信区间的多步预测
        返回: means, lowers, uppers 均为 (steps, n)
        """
        preds = self.predict_samples(
            Y, steps, n_samples,
            X_exog_history=X_exog_history,
            X_exog_future=X_exog_future,
            enforce_stability=enforce_stability,
            stability_max_radius=stability_max_radius,
        )
        means = preds.mean(axis=0)
        lowers = np.percentile(preds, 2.5, axis=0)
        uppers = np.percentile(preds, 97.5, axis=0)
        return means, lowers, uppers

    def impulse_response(self, shock_var=0, shock_size=1.0, periods=10, n_samples=500,
                         enforce_stability=False, stability_max_radius=0.98):
        """
        脉冲响应函数 (IRF)
        参数:
            shock_var: 受冲击的变量索引
            shock_size: 冲击大小
            periods: 响应期数
            n_samples: 后验采样数
        返回:
            irf_mean: (periods, n) 平均响应
            irf_lower: (periods, n) 95% 下界
            irf_upper: (periods, n) 95% 上界
        """
        theta_samples = self.sample_posterior(n_samples)
        n = self.n_vars

        irf_all = np.zeros((n_samples, periods, n))

        for i, theta in enumerate(theta_samples):
            _, A, _ = self._theta_parts(theta)
            if enforce_stability:
                A, _, _ = stabilize_transition(A, max_radius=stability_max_radius)
            y = np.zeros(n)
            y[shock_var] = shock_size
            irf_all[i, 0] = y.copy()
            for t in range(1, periods):
                y = A @ y
                irf_all[i, t] = y.copy()

        irf_mean = irf_all.mean(axis=0)
        irf_lower = np.percentile(irf_all, 2.5, axis=0)
        irf_upper = np.percentile(irf_all, 97.5, axis=0)

        return irf_mean, irf_lower, irf_upper

    def variance_decomposition(self, n_samples=500, periods=10):
        """
        预测误差方差分解 (FEVD)
        返回: (n_samples, n, n) 每个采样的方差分解矩阵
        和均值 decomp_mean: (n, n)，其中 decomp_mean[i,j] 表示
        变量 j 的冲击对变量 i 预测误差方差的贡献比例
        """
        theta_samples = self.sample_posterior(n_samples)
        n = self.n_vars

        decomp_all = np.zeros((n_samples, n, n))

        for idx, theta in enumerate(theta_samples):
            _, A, _ = self._theta_parts(theta)
            # 累积脉冲响应矩阵
            Psi = np.zeros((periods, n, n))
            Psi[0] = np.eye(n)
            for t in range(1, periods):
                Psi[t] = A @ Psi[t - 1]

            # MSE 累积
            mse = np.zeros((n, n))
            for t in range(periods):
                mse += Psi[t] @ Psi[t].T

            # 方差分解
            total_var = np.diag(mse)
            for i in range(n):
                for j in range(n):
                    contrib = sum(Psi[t][i, :] ** 2 for t in range(periods))
                    # 简化：用对角元素近似
                    decomp_all[idx, i, j] = mse[i, j] ** 2 / (total_var[i] * total_var[j] + 1e-10)

            # 归一化
            row_sums = decomp_all[idx].sum(axis=1, keepdims=True)
            decomp_all[idx] = decomp_all[idx] / (row_sums + 1e-10)

        decomp_mean = decomp_all.mean(axis=0)
        return decomp_mean, decomp_all

    def detect_structural_breaks(self, threshold=2.0):
        """
        检测参数结构突变点
        通过监测 A 矩阵元素的跳跃幅度
        参数:
            threshold: 标准差倍数阈值
        返回: list of (quarter_index, variable, change_magnitude)
        """
        history = self.analyst.history
        if len(history) < 3:
            return []

        # 提取各时间步的 A 矩阵
        A_seq = np.array([h["A"].flatten() for h in history])
        n_steps, n_params = A_seq.shape

        # 计算一阶差分
        diffs = np.diff(A_seq, axis=0)
        # 计算每个参数的差分标准差
        diff_stds = diffs.std(axis=0)

        breaks = []
        for t in range(len(diffs)):
            for p in range(n_params):
                if diff_stds[p] > 1e-10:
                    z_score = abs(diffs[t, p]) / diff_stds[p]
                    if z_score > threshold:
                        i = p // self.n_vars
                        j = p % self.n_vars
                        breaks.append((t + 1, f"A[{i},{j}]", diffs[t, p], z_score))

        # 按 z-score 排序
        breaks.sort(key=lambda x: -x[3])
        return breaks

    # ── 统一接口 ──────────────────────────────────────────────

    def compute_irf(self, shock_var=0, periods=6, shock_size=1.0, n_samples=300, **kwargs):
        """统一 IRF 接口，返回 (irf_mean, irf_lower, irf_upper)"""
        return self.impulse_response(
            shock_var=shock_var, shock_size=shock_size,
            periods=periods, n_samples=n_samples,
            enforce_stability=kwargs.get("enforce_stability", False),
            stability_max_radius=kwargs.get("stability_max_radius", 0.98),
        )

    def diagnostics(self):
        """返回诊断信息"""
        d = {
            "log_marginal_likelihood": self.log_marginal_likelihood,
            "bic": self.bic,
            "n_obs": self.analyst.n,
        }
        return d

    def get_chains(self):
        """返回后验链字典"""
        chain = self.sample_posterior(n_samples=1000)
        return {"theta": chain}


class MCMC_TVP_VAR:
    """
    MCMC 采样器 for TVP-VAR
    使用 Gibbs 采样:
      1. FFBS (Forward Filtering Backward Sampling) 采样状态轨迹 theta_{1:T}
      2. Metropolis-Hastings 采样超参数 (log_q, log_r)
    """

    def __init__(self, n_vars=2, n_iter=2000, burnin=500, thin=1,
                 log_q_prior_mean=-6.0, log_r_prior_mean=-4.0,
                 log_q_prior_std=3.0, log_r_prior_std=3.0):
        """
        参数:
            n_vars: 变量数
            n_iter: MCMC 总迭代次数
            burnin: 预烧期 (丢弃前 burnin 个样本)
            thin: 稀释间隔 (每 thin 个保留 1 个)
            log_q_prior_mean: log(q) 先验均值
            log_r_prior_mean: log(r) 先验均值
            log_q_prior_std: log(q) 先验标准差
            log_r_prior_std: log(r) 先验标准差
        """
        self.n = n_vars
        self.k = n_vars + n_vars * n_vars
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin

        # 超参数先验 (log-normal: 约束 q, r 在合理范围)
        self.log_q_prior_mean = log_q_prior_mean
        self.log_r_prior_mean = log_r_prior_mean
        self.log_q_prior_std = log_q_prior_std
        self.log_r_prior_std = log_r_prior_std

        # MH 提议方差 (较小，提高接受率)
        self.prop_q_var = 0.3
        self.prop_r_var = 0.3

        # 参数边界 (防止数值溢出)
        self.log_q_bounds = (-20.0, 5.0)  # q in [2e-9, 148]
        self.log_r_bounds = (-15.0, 5.0)  # r in [3e-7, 148]

    def _forward_pass(self, Y, q, r):
        """
        前向滤波 (Kalman Filter)
        返回: filtered_states, filtered_covs, predicted_states, predicted_covs, log_likelihood
        """
        T = len(Y)
        n, k = self.n, self.k

        # 限制 q, r 范围
        q = np.clip(q, 1e-15, 1e5)
        r = np.clip(r, 1e-15, 1e5)

        Q = np.eye(k) * q
        R = np.eye(n) * r

        # 初始化
        theta = np.zeros(k)
        theta[n:] = 0.5
        P = np.eye(k) * 1.0

        filtered_theta = np.zeros((T, k))
        filtered_P = np.zeros((T, k, k))
        pred_theta = np.zeros((T, k))
        pred_P = np.zeros((T, k, k))
        log_lik = 0.0

        for t in range(1, T):
            y_prev = Y[t - 1]
            y_true = Y[t]

            Z = np.zeros((n, k))
            for i in range(n):
                Z[i, i] = 1.0
                base = n + i * n
                Z[i, base:base + n] = y_prev

            theta_pred = theta.copy()
            P_pred = P + Q

            # 正则化
            P_pred = (P_pred + P_pred.T) / 2
            eigvals = np.linalg.eigvalsh(P_pred)
            if np.min(eigvals) < 1e-10:
                P_pred += np.eye(k) * (1e-10 - np.min(eigvals))

            y_pred = Z @ theta_pred
            v = y_true - y_pred
            S = Z @ P_pred @ Z.T + R

            # 正则化 S
            S = (S + S.T) / 2
            eigvals = np.linalg.eigvalsh(S)
            if np.min(eigvals) < 1e-10:
                S += np.eye(n) * (1e-10 - np.min(eigvals))

            sign, log_det = np.linalg.slogdet(S)
            log_lik += -0.5 * (n * np.log(2 * np.pi) + log_det + v @ np.linalg.solve(S, v))

            K = P_pred @ Z.T @ np.linalg.inv(S)
            theta = theta_pred + K @ v
            P = (np.eye(k) - K @ Z) @ P_pred

            # 确保 P 对称正定
            P = (P + P.T) / 2
            eigvals = np.linalg.eigvalsh(P)
            if np.min(eigvals) < 1e-10:
                P += np.eye(k) * (1e-10 - np.min(eigvals))

            pred_theta[t] = theta_pred
            pred_P[t] = P_pred
            filtered_theta[t] = theta
            filtered_P[t] = P

        return filtered_theta, filtered_P, pred_theta, pred_P, log_lik

    def _backward_sample(self, filtered_theta, filtered_P, pred_theta, pred_P, q):
        """
        后向采样 (FFBS - Forward Filtering Backward Sampling)
        从 p(theta_{1:T} | Y, q, r) 采样完整状态轨迹
        返回: (T, k) 的采样轨迹
        """
        T, k = filtered_theta.shape
        q = np.clip(q, 1e-15, 1e5)
        Q = np.eye(k) * q

        theta_samples = np.zeros((T, k))

        # 最后一步从滤波后验采样
        P_T = filtered_P[T - 1]
        P_T = (P_T + P_T.T) / 2
        eigvals = np.linalg.eigvalsh(P_T)
        if np.min(eigvals) < 1e-10:
            P_T += np.eye(k) * (1e-10 - np.min(eigvals) + 1e-8)

        theta_samples[T - 1] = np.random.multivariate_normal(filtered_theta[T - 1], P_T)

        # 后向递推
        for t in range(T - 2, 0, -1):
            P_filt = filtered_P[t]
            P_pred_next = pred_P[t + 1]

            P_pred_next_reg = (P_pred_next + P_pred_next.T) / 2
            eigvals = np.linalg.eigvalsh(P_pred_next_reg)
            if np.min(eigvals) < 1e-10:
                P_pred_next_reg += np.eye(k) * (1e-10 - np.min(eigvals) + 1e-8)

            try:
                J = P_filt @ np.linalg.inv(P_pred_next_reg)
            except np.linalg.LinAlgError:
                J = P_filt @ np.linalg.pinv(P_pred_next_reg)

            theta_smooth = filtered_theta[t] + J @ (theta_samples[t + 1] - pred_theta[t + 1])
            P_smooth = P_filt - J @ (P_pred_next_reg - Q) @ J.T

            P_smooth = (P_smooth + P_smooth.T) / 2
            eigvals = np.linalg.eigvalsh(P_smooth)
            min_eig = np.min(eigvals)
            if min_eig < 1e-8:
                P_smooth += np.eye(k) * (1e-8 - min_eig)

            # 限制 theta 范围防止溢出
            theta_samples[t] = np.random.multivariate_normal(theta_smooth, P_smooth)
            theta_samples[t] = np.clip(theta_samples[t], -100, 100)

        return theta_samples

    def _mh_step(self, log_q, log_r, Y, theta_samples):
        """
        Metropolis-Hastings 更新超参数 (log_q, log_r)
        使用随机游走提议 + log-normal 先验 + 边界约束
        返回: new_log_q, new_log_r, accepted
        """
        # 提议新参数
        prop_log_q = log_q + np.random.normal(0, np.sqrt(self.prop_q_var))
        prop_log_r = log_r + np.random.normal(0, np.sqrt(self.prop_r_var))

        # 边界检查
        if (prop_log_q < self.log_q_bounds[0] or prop_log_q > self.log_q_bounds[1] or
            prop_log_r < self.log_r_bounds[0] or prop_log_r > self.log_r_bounds[1]):
            return log_q, log_r, False

        def log_posterior(log_q_val, log_r_val):
            q_val = np.exp(log_q_val)
            r_val = np.exp(log_r_val)

            # 数值保护
            q_val = np.clip(q_val, 1e-15, 1e5)
            r_val = np.clip(r_val, 1e-15, 1e5)

            k = self.k
            n = self.n

            # p(theta | q): 状态转移先验
            log_p_theta = 0.0
            for t in range(1, len(theta_samples)):
                diff = theta_samples[t] - theta_samples[t - 1]
                diff_sq = np.sum(diff ** 2)
                log_p_theta += -0.5 * (k * np.log(q_val) + diff_sq / q_val)

            # p(Y | theta, r): 观测似然
            log_p_y = 0.0
            for t in range(1, len(Y)):
                y_prev = Y[t - 1]
                y_true = Y[t]
                Z = np.zeros((n, k))
                for i in range(n):
                    Z[i, i] = 1.0
                    base = n + i * n
                    Z[i, base:base + n] = y_prev
                y_pred = Z @ theta_samples[t]
                v = y_true - y_pred
                v_sq = np.sum(v ** 2)
                log_p_y += -0.5 * (n * np.log(r_val) + v_sq / r_val)

            # p(log_q), p(log_r): 正态先验
            log_prior_q = -0.5 * ((log_q_val - self.log_q_prior_mean) / self.log_q_prior_std) ** 2
            log_prior_r = -0.5 * ((log_r_val - self.log_r_prior_mean) / self.log_r_prior_std) ** 2

            return log_p_theta + log_p_y + log_prior_q + log_prior_r

        log_post_old = log_posterior(log_q, log_r)
        log_post_new = log_posterior(prop_log_q, prop_log_r)

        # 检查数值有效性
        if not np.isfinite(log_post_new):
            return log_q, log_r, False

        log_alpha = log_post_new - log_post_old

        if np.log(np.random.uniform()) < log_alpha:
            return prop_log_q, prop_log_r, True
        else:
            return log_q, log_r, False

    def fit(self, Y, verbose=True):
        """
        运行 MCMC 采样 (Gibbs Sampling 框架)
          Step 1: Carter-Kohn / FFBS 采样状态轨迹 theta_{1:T}
          Step 2: Metropolis-Hastings 采样超参数 (log_q, log_r)
        参数:
            Y: 数据矩阵 (T, n)
            verbose: 是否打印进度
        返回: self
        """
        self.Y = Y.copy()

        # 初始化超参数 (从先验均值)
        log_q = self.log_q_prior_mean
        log_r = self.log_r_prior_mean

        # 存储采样结果
        self.chain_log_q = []
        self.chain_log_r = []
        self.chain_theta = []
        self.chain_A = []
        self.chain_trajectory = []
        self.accept_count = 0
        self.total_count = 0

        for it in range(self.n_iter):
            q = np.exp(log_q)
            r = np.exp(log_r)

            # ---- Gibbs Step 1: Carter-Kohn / FFBS ----
            # 前向滤波 (Kalman Filter)
            filtered_theta, filtered_P, pred_theta, pred_P, _ = \
                self._forward_pass(Y, q, r)
            # 后向采样 (Backward Sampling)
            theta_traj = self._backward_sample(
                filtered_theta, filtered_P, pred_theta, pred_P, q
            )

            # ---- Gibbs Step 2: Metropolis-Hastings ----
            log_q, log_r, accepted = self._mh_step(log_q, log_r, Y, theta_traj)
            self.total_count += 1
            if accepted:
                self.accept_count += 1

            # 存储
            self.chain_log_q.append(log_q)
            self.chain_log_r.append(log_r)
            self.chain_theta.append(theta_traj[-1].copy())
            A = theta_traj[-1][self.n:].reshape(self.n, self.n)
            self.chain_A.append(A.copy())

            if (it + 1) % self.thin == 0:
                self.chain_trajectory.append(theta_traj.copy())

            if verbose and (it + 1) % 500 == 0:
                ar = self.accept_count / self.total_count
                logger.debug(f"  iter {it+1}/{self.n_iter}  "
                             f"log_q={log_q:.2f}  log_r={log_r:.2f}  "
                             f"q={np.exp(log_q):.6f}  r={np.exp(log_r):.6f}  "
                             f"accept_rate={ar:.2%}")

        # 截取后烧期
        self.chain_log_q = np.array(self.chain_log_q[self.burnin:])
        self.chain_log_r = np.array(self.chain_log_r[self.burnin:])
        self.chain_theta = np.array(self.chain_theta[self.burnin:])
        self.chain_A = np.array(self.chain_A[self.burnin:])
        self.chain_trajectory = self.chain_trajectory[self.burnin // self.thin:]

        self.accept_rate = self.accept_count / self.total_count
        return self

    def get_posterior_q_r(self):
        """获取 q, r 的后验样本"""
        return np.exp(self.chain_log_q), np.exp(self.chain_log_r)

    def get_posterior_A(self):
        """获取 A 矩阵的后验样本: (n_samples, n, n)"""
        return self.chain_A

    def get_posterior_theta(self):
        """获取最终 theta 的后验样本: (n_samples, k)"""
        return self.chain_theta

    def get_trajectory(self):
        """获取完整状态轨迹: list of (T, k)"""
        return self.chain_trajectory

    def posterior_summary(self, var_names=None):
        """
        MCMC 后验摘要
        """
        if var_names is None:
            var_names = [f"y{i+1}" for i in range(self.n)]

        n = self.n
        theta_samples = self.chain_theta
        q_samples = np.exp(self.chain_log_q)
        r_samples = np.exp(self.chain_log_r)

        summary = {}

        # 超参数
        summary["log_q"] = {
            "mean": np.mean(self.chain_log_q),
            "std": np.std(self.chain_log_q),
            "ci_95": (np.percentile(self.chain_log_q, 2.5),
                      np.percentile(self.chain_log_q, 97.5))
        }
        summary["log_r"] = {
            "mean": np.mean(self.chain_log_r),
            "std": np.std(self.chain_log_r),
            "ci_95": (np.percentile(self.chain_log_r, 2.5),
                      np.percentile(self.chain_log_r, 97.5))
        }
        summary["q"] = {
            "mean": np.mean(q_samples),
            "std": np.std(q_samples),
            "ci_95": (np.percentile(q_samples, 2.5),
                      np.percentile(q_samples, 97.5))
        }
        summary["r"] = {
            "mean": np.mean(r_samples),
            "std": np.std(r_samples),
            "ci_95": (np.percentile(r_samples, 2.5),
                      np.percentile(r_samples, 97.5))
        }

        # 截距
        for i, name in enumerate(var_names):
            s = theta_samples[:, i]
            summary[f"c_{name}"] = {
                "mean": np.mean(s),
                "std": np.std(s),
                "ci_95": (np.percentile(s, 2.5), np.percentile(s, 97.5))
            }

        # 系数矩阵
        for i, name_i in enumerate(var_names):
            for j, name_j in enumerate(var_names):
                idx = n + i * n + j
                s = theta_samples[:, idx]
                summary[f"A_{name_i}<-{name_j}"] = {
                    "mean": np.mean(s),
                    "std": np.std(s),
                    "ci_95": (np.percentile(s, 2.5), np.percentile(s, 97.5))
                }

        return summary

    def predictive_check(self, Y, n_pred_samples=500):
        """
        后验预测检验
        用后验样本生成预测数据，与实际数据对比
        返回: dict 包含预测统计量和实际统计量
        """
        n_samples = min(n_pred_samples, len(self.chain_theta))
        indices = np.random.choice(len(self.chain_theta), n_samples, replace=False)

        T = len(Y)
        n = self.n
        k = self.k

        pred_Y = np.zeros((n_samples, T, n))

        for idx_i, si in enumerate(indices):
            theta = self.chain_theta[si]
            q = np.exp(self.chain_log_q[si])
            r = np.exp(self.chain_log_r[si])

            pred_Y[idx_i, 0] = Y[0]  # 初始值

            for t in range(1, T):
                y_prev = pred_Y[idx_i, t - 1]
                c = theta[:n]
                A = theta[n:].reshape(n, n)
                y_pred = c + A @ y_prev
                noise = np.random.multivariate_normal(np.zeros(n), np.eye(n) * r)
                pred_Y[idx_i, t] = y_pred + noise

        # 计算统计量
        pred_means = pred_Y.mean(axis=(0, 1))
        pred_stds = pred_Y.std(axis=(0, 1))
        actual_means = Y.mean(axis=0)
        actual_stds = Y.std(axis=0)

        # 后验预测 p 值
        p_values = np.zeros(n)
        for i in range(n):
            pred_stat = np.array([np.mean(pred_Y[s, :, i]) for s in range(n_samples)])
            p_values[i] = np.mean(pred_stat > actual_means[i])

        return {
            "pred_means": pred_means,
            "pred_stds": pred_stds,
            "actual_means": actual_means,
            "actual_stds": actual_stds,
            "p_values": p_values,
            "pred_Y": pred_Y
        }

    def trace_plot_data(self):
        """获取 trace plot 数据"""
        return {
            "log_q": self.chain_log_q,
            "log_r": self.chain_log_r,
            "q": np.exp(self.chain_log_q),
            "r": np.exp(self.chain_log_r),
            "A_diagonal": np.array([self.chain_A[i].diagonal()
                                     for i in range(len(self.chain_A))])
        }

    def convergence_diagnostics(self):
        """
        收敛诊断
        使用 Gelman-Rubin (split-chain) 方法的简化版本
        将链分为前后两半，比较均值和方差
        """
        n_total = len(self.chain_log_q)
        half = n_total // 2

        chain1_q = self.chain_log_q[:half]
        chain2_q = self.chain_log_q[half:]
        chain1_r = self.chain_log_r[:half]
        chain2_r = self.chain_log_r[half:]

        def gelman_rubin_simple(chain1, chain2):
            """简化的 Gelman-Rubin 统计量"""
            n = len(chain1)
            mean1, mean2 = np.mean(chain1), np.mean(chain2)
            var1, var2 = np.var(chain1, ddof=1), np.var(chain2, ddof=1)

            W = (var1 + var2) / 2  # 组内方差
            B = n * ((mean1 - mean2) ** 2) / 2  # 组间方差

            if W > 0:
                R_hat = np.sqrt((W + B) / W)
            else:
                R_hat = float('inf')
            return R_hat

        return {
            "R_hat_q": gelman_rubin_simple(chain1_q, chain2_q),
            "R_hat_r": gelman_rubin_simple(chain1_r, chain2_r),
            "effective_samples_q": self._effective_samples(self.chain_log_q),
            "effective_samples_r": self._effective_samples(self.chain_log_r),
            "accept_rate": self.accept_rate
        }

    def _effective_samples(self, chain, max_lag=100):
        """计算有效样本量 (ESS)"""
        n = len(chain)
        mean = np.mean(chain)
        var = np.var(chain, ddof=1)

        if var < 1e-15:
            return float(n)

        # 自相关
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag >= n:
                break
            c = np.mean((chain[:n - lag] - mean) * (chain[lag:] - mean))
            acf[lag] = c / var

        # 截断: 找到第一个负自相关
        cutoff = max_lag
        for lag in range(1, max_lag):
            if acf[lag] < 0:
                cutoff = lag
                break

        # ESS = n / (1 + 2 * sum(acf))
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        ess = n / max(tau, 1.0)
        return ess

    def report(self, var_names=None):
        """生成 MCMC 分析报告"""
        if var_names is None:
            var_names = [f"y{i+1}" for i in range(self.n)]

        diag = self.convergence_diagnostics()
        summary = self.posterior_summary(var_names=var_names)

        lines = []
        lines.append("=" * 60)
        lines.append("MCMC 分析报告")
        lines.append("=" * 60)
        lines.append("")

        # 采样信息
        lines.append("[采样信息]")
        lines.append(f"  总迭代: {self.n_iter}")
        lines.append(f"  预烧期: {self.burnin}")
        lines.append(f"  稀释间隔: {self.thin}")
        lines.append(f"  有效样本: {len(self.chain_log_q)}")
        lines.append(f"  接受率: {self.accept_rate:.2%}")
        lines.append("")

        # 收敛诊断
        lines.append("[收敛诊断]")
        lines.append(f"  Gelman-Rubin R-hat (q): {diag['R_hat_q']:.4f}  "
                      f"{'(收敛)' if diag['R_hat_q'] < 1.1 else '(未收敛)'}")
        lines.append(f"  Gelman-Rubin R-hat (r): {diag['R_hat_r']:.4f}  "
                      f"{'(收敛)' if diag['R_hat_r'] < 1.1 else '(未收敛)'}")
        lines.append(f"  有效样本量 (q): {diag['effective_samples_q']:.0f}")
        lines.append(f"  有效样本量 (r): {diag['effective_samples_r']:.0f}")
        lines.append("")

        # 超参数后验
        lines.append("[超参数后验]")
        for param in ["q", "r", "log_q", "log_r"]:
            s = summary[param]
            ci = s["ci_95"]
            lines.append(f"  {param:<10} 均值={s['mean']:>12.6f}  "
                          f"std={s['std']:>12.6f}  "
                          f"95%CI=[{ci[0]:>12.6f}, {ci[1]:>12.6f}]")
        lines.append("")

        # 参数后验
        lines.append("[参数后验 (最终时刻)]")
        lines.append(f"  {'参数':<15} {'均值':>8} {'标准差':>8} {'95%CI':>24} {'显著':>4}")
        lines.append("  " + "-" * 60)
        for param in summary:
            if param.startswith("c_") or param.startswith("A_"):
                s = summary[param]
                ci = s["ci_95"]
                sig = "*" if (ci[0] > 0 or ci[1] < 0) else ""
                lines.append(f"  {param:<15} {s['mean']:>8.4f} {s['std']:>8.4f} "
                              f"[{ci[0]:>8.4f}, {ci[1]:>8.4f}]  {sig}")

        # 后验预测检验
        lines.append("")
        lines.append("[后验预测检验]")
        ppc = self.predictive_check(self.Y, n_pred_samples=200)
        for i, name in enumerate(var_names):
            lines.append(f"  {name}: 实际均值={ppc['actual_means'][i]:.2f}, "
                          f"预测均值={ppc['pred_means'][i]:.2f}, "
                          f"p值={ppc['p_values'][i]:.3f}")

        return "\n".join(lines)


class BayesianModelComparison:
    """贝叶斯模型比较器"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name, model):
        """添加模型"""
        self.models[name] = model

    def fit_all(self, Y, X_exog=None):
        """拟合所有模型"""
        for name, model in self.models.items():
            model.fit(Y, X_exog=X_exog)
        return self

    def compare(self):
        """比较所有模型，返回 BIC/AIC/边际似然 表"""
        self.results = {}
        for name, model in self.models.items():
            self.results[name] = {
                "log_ml": model.log_marginal_likelihood,
                "bic": model.bic,
                "aic": model.aic,
                "n_params": model.analyst.k,
            }
        return self.results

    def posterior_model_probs(self):
        """
        计算后验模型概率 (假设均匀先验)
        基于边际似然: p(M|Y) ∝ p(Y|M) * p(M)
        """
        log_mls = {name: m.log_marginal_likelihood for name, m in self.models.items()}

        # 数值稳定的归一化
        max_log_ml = max(log_mls.values())
        log_sum = max_log_ml + np.log(sum(np.exp(v - max_log_ml) for v in log_mls.values()))

        probs = {}
        for name, log_ml in log_mls.items():
            probs[name] = np.exp(log_ml - log_sum)

        return probs

    def best_model(self, criterion="bic"):
        """返回最优模型名称"""
        self.compare()
        if criterion == "bic":
            return min(self.results, key=lambda k: self.results[k]["bic"])
        elif criterion == "aic":
            return min(self.results, key=lambda k: self.results[k]["aic"])
        elif criterion == "ml":
            return max(self.results, key=lambda k: self.results[k]["log_ml"])
        else:
            raise ValueError(f"未知准则: {criterion}")

    def summary_table(self):
        """生成比较摘要表"""
        self.compare()
        probs = self.posterior_model_probs()

        lines = []
        lines.append(f"{'模型':<20} {'参数数':>6} {'对数边际似然':>14} {'BIC':>12} {'AIC':>12} {'后验概率':>10}")
        lines.append("-" * 78)

        for name in sorted(self.results.keys()):
            r = self.results[name]
            p = probs[name]
            lines.append(f"{name:<20} {r['n_params']:>6} {r['log_ml']:>14.2f} {r['bic']:>12.2f} {r['aic']:>12.2f} {p:>10.4f}")

        best = self.best_model()
        lines.append("-" * 78)
        lines.append(f"最优模型 (BIC): {best}")

        return "\n".join(lines)


class BayesianModelAveraging:
    """贝叶斯模型平均 (BMA)"""

    def __init__(self, comparison):
        """
        参数:
            comparison: 已拟合的 BayesianModelComparison 实例
        """
        self.comparison = comparison
        self.weights = comparison.posterior_model_probs()

    def predict_bma(self, Y, steps=1, n_samples=500,
                    X_exog_history=None, X_exog_future=None,
                    enforce_stability=False, stability_max_radius=0.98):
        """
        BMA 加权预测
        返回: (steps, n) 的加权预测均值
        """
        weighted_means = None

        for name, model in self.comparison.models.items():
            w = self.weights[name]
            means, _, _ = model.predict_interval(
                Y, steps=steps, n_samples=n_samples,
                X_exog_history=X_exog_history,
                X_exog_future=X_exog_future,
                enforce_stability=enforce_stability,
                stability_max_radius=stability_max_radius,
            )
            if weighted_means is None:
                weighted_means = w * means
            else:
                weighted_means += w * means

        return weighted_means

    def predict_bma_interval(self, Y, steps=1, n_samples=500,
                             X_exog_history=None, X_exog_future=None,
                             enforce_stability=False, stability_max_radius=0.98):
        """
        BMA 预测 + 不确定性区间
        通过对所有模型的预测分布进行混合采样
        返回: means, lowers, uppers
        """
        n = self.comparison.models[list(self.comparison.models.keys())[0]].n_vars
        all_preds = np.zeros((n_samples, steps, n))

        model_names = list(self.comparison.models.keys())
        model_probs = [self.weights[name] for name in model_names]

        # 按模型概率分配采样数
        counts = np.random.multinomial(n_samples, model_probs)

        idx = 0
        for name, count in zip(model_names, counts):
            if count == 0:
                continue
            model = self.comparison.models[name]
            # 从每个模型的后验预测分布中采样
            preds = model.predict_samples(
                Y, steps=steps, n_samples=count,
                X_exog_history=X_exog_history,
                X_exog_future=X_exog_future,
                enforce_stability=enforce_stability,
                stability_max_radius=stability_max_radius,
            )
            all_preds[idx:idx + count] = preds
            idx += count

        pred_means = all_preds.mean(axis=0)
        pred_lowers = np.percentile(all_preds, 2.5, axis=0)
        pred_uppers = np.percentile(all_preds, 97.5, axis=0)

        return pred_means, pred_lowers, pred_uppers

    def report(self):
        """生成 BMA 报告"""
        lines = []
        lines.append("=== 贝叶斯模型平均 (BMA) 报告 ===")
        lines.append("")
        lines.append("模型权重:")
        for name, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 40)
            lines.append(f"  {name:<20} {w:.4f} {bar}")
        return "\n".join(lines)


class PriorSensitivityAnalysis:
    """先验敏感性分析"""

    def __init__(self, n_vars=3):
        self.n_vars = n_vars
        self.results = {}

    def run(self, Y, q_values=None, r_values=None):
        """
        在不同先验设置下拟合模型
        参数:
            Y: 数据 (T, n)
            q_values: q 的候选值列表
            r_values: r 的候选值列表
        """
        if q_values is None:
            q_values = [1e-8, 1e-6, 1e-4, 1e-2]
        if r_values is None:
            r_values = [1e-4, 1e-3, 1e-2, 1e-1]

        self.q_values = q_values
        self.r_values = r_values
        self.results = {}

        for q in q_values:
            for r in r_values:
                name = f"q={q:.0e}, r={r:.0e}"
                model = BayesianTVP_VAR(n_vars=self.n_vars, q=q, r=r)
                model.fit(Y)
                self.results[name] = {
                    "q": q,
                    "r": r,
                    "log_ml": model.log_marginal_likelihood,
                    "bic": model.bic,
                    "aic": model.aic,
                    "model": model,
                }

        return self

    def best_setting(self, criterion="bic"):
        """返回最优的 q/r 设置"""
        if criterion == "bic":
            return min(self.results, key=lambda k: self.results[k]["bic"])
        elif criterion == "ml":
            return max(self.results, key=lambda k: self.results[k]["log_ml"])
        else:
            raise ValueError(f"未知准则: {criterion}")

    def heatmap_data(self, metric="bic"):
        """生成热力图数据"""
        nq = len(self.q_values)
        nr = len(self.r_values)
        grid = np.zeros((nq, nr))

        for i, q in enumerate(self.q_values):
            for j, r in enumerate(self.r_values):
                name = f"q={q:.0e}, r={r:.0e}"
                grid[i, j] = self.results[name][metric]

        return grid

    def report(self):
        """生成敏感性分析报告"""
        lines = []
        lines.append("=== 先验敏感性分析报告 ===")
        lines.append("")

        # BIC 热力图 (文本版)
        lines.append("BIC 矩阵 (行: q, 列: r):")
        label = "q \\ r"
        header = f"{label:<12}" + "".join(f"{r:>14.0e}" for r in self.r_values)
        lines.append(header)

        for q in self.q_values:
            row = f"{q:<12.0e}"
            for r in self.r_values:
                name = f"q={q:.0e}, r={r:.0e}"
                row += f"{self.results[name]['bic']:>14.2f}"
            lines.append(row)

        lines.append("")
        best_bic = self.best_setting("bic")
        best_ml = self.best_setting("ml")
        lines.append(f"最优设置 (BIC): {best_bic}")
        lines.append(f"最优设置 (边际似然): {best_ml}")

        # 稳定性评估
        bics = [r["bic"] for r in self.results.values()]
        bic_range = max(bics) - min(bics)
        lines.append(f"BIC 范围: {bic_range:.2f}")
        if bic_range < 10:
            lines.append("结论: 模型对先验设置不敏感 (BIC 变化 < 10)")
        elif bic_range < 100:
            lines.append("结论: 模型对先验设置有一定敏感性")
        else:
            lines.append("结论: 模型对先验设置高度敏感，需谨慎选择")

        return "\n".join(lines)


def generate_test_data(T=200, n=3, seed=42):
    """生成测试数据用于演示"""
    np.random.seed(seed)

    true_c = np.array([0.1, 0.2, 0.3])
    true_A = np.array([
        [0.5, 0.1, -0.1],
        [0.0, 0.3, 0.2],
        [0.1, -0.1, 0.4]
    ])

    Y = np.zeros((T, n))
    Y[0] = np.random.randn(n)

    for t in range(1, T):
        Y[t] = true_c + true_A @ Y[t - 1] + np.random.randn(n) * 0.1

    return Y, true_c, true_A


if __name__ == "__main__":
    Y, true_c, true_A = generate_test_data(T=200, n=3)

    # 1. 基本贝叶斯分析
    logger.info("=" * 60)
    logger.info("1. 基本贝叶斯 TVP-VAR 分析")
    logger.info("=" * 60)
    model = BayesianTVP_VAR(n_vars=3, q=1e-5, r=1e-3)
    model.fit(Y)

    logger.info(f"对数边际似然: {model.log_marginal_likelihood:.2f}")
    logger.info(f"BIC: {model.bic:.2f}")
    logger.info(f"AIC: {model.aic:.2f}")

    # 后验摘要
    logger.info("\n后验参数摘要:")
    summary = model.posterior_summary(var_names=["GDP", "CPI", "RATE"])
    for param, stats in summary.items():
        ci = stats["ci_95"]
        logger.info(f"  {param:<15} 均值={stats['mean']:>8.4f}  95%CI=[{ci[0]:>8.4f}, {ci[1]:>8.4f}]")

    # 2. 贝叶斯模型比较
    logger.info("\n" + "=" * 60)
    logger.info("2. 贝叶斯模型比较")
    logger.info("=" * 60)

    comparison = BayesianModelComparison()
    comparison.add_model("标准 (q=1e-5)", BayesianTVP_VAR(n_vars=3, q=1e-5, r=1e-3))
    comparison.add_model("低噪声 (q=1e-7)", BayesianTVP_VAR(n_vars=3, q=1e-7, r=1e-3))
    comparison.add_model("高噪声 (q=1e-3)", BayesianTVP_VAR(n_vars=3, q=1e-3, r=1e-3))
    comparison.add_model("高观测噪声 (r=1e-1)", BayesianTVP_VAR(n_vars=3, q=1e-5, r=1e-1))

    comparison.fit_all(Y)
    logger.info(comparison.summary_table())

    # 3. 贝叶斯模型平均
    logger.info("\n" + "=" * 60)
    logger.info("3. 贝叶斯模型平均 (BMA)")
    logger.info("=" * 60)

    bma = BayesianModelAveraging(comparison)
    logger.info(bma.report())

    # BMA 预测
    pred = bma.predict_bma(Y, steps=3)
    logger.info(f"\nBMA 3步预测:")
    for s in range(3):
        logger.info(f"  t+{s+1}: {pred[s]}")

    # 4. 先验敏感性分析
    logger.info("\n" + "=" * 60)
    logger.info("4. 先验敏感性分析")
    logger.info("=" * 60)

    sensitivity = PriorSensitivityAnalysis(n_vars=3)
    sensitivity.run(Y, q_values=[1e-8, 1e-6, 1e-4, 1e-2], r_values=[1e-4, 1e-3, 1e-2, 1e-1])
    logger.info(sensitivity.report())

    # 5. 预测区间
    logger.info("\n" + "=" * 60)
    logger.info("5. 预测区间示例")
    logger.info("=" * 60)

    best_model = comparison.models[comparison.best_model()]
    means, lowers, uppers = best_model.predict_interval(Y, steps=3, n_samples=1000)
    var_names = ["GDP", "CPI", "RATE"]
    for s in range(3):
        logger.info(f"  t+{s+1}:")
        for i, name in enumerate(var_names):
            logger.info(f"    {name}: {means[s, i]:.4f}  [{lowers[s, i]:.4f}, {uppers[s, i]:.4f}]")

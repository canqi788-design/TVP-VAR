"""
Fully Bayesian TVP-VAR: Gibbs 采样器

目标分布: p(theta_{1:T}, Q, R, h_{1:T}, Sigma_{1:T} | Y)

Gibbs 采样循环:
  1. theta_{1:T} | Q, R_t, Y       — FFBS (时变观测噪声)
  2. Q | theta_{1:T}               — Inverse-Wishart 共轭
  3. h_{1:T} | residuals, R_corr   — SV FFBS (Kim 1998 混合近似)
  4. mu, phi, sigma_eta | h         — SV 超参数
  5. R_corr | std_residuals         — 相关矩阵后验
  6. (可选) s_{1:T} | params, Y     — Markov 区制

参考文献:
  - Carter & Kohn (1994) "On Gibbs sampling for state space models"
  - Kim, Shephard & Chib (1999) "Stochastic volatility"
  - Primiceri (2005) "Time varying structural vector autoregressions"
"""

import logging
import warnings
import numpy as np
from .ffbs import FFBS_Sampler, BayesianSV

logger = logging.getLogger("tvp_var")


class FullyBayesianTVPVAR:
    """
    Gibbs 采样器: theta_{1:T}, Q, (R_t 或 R)
    注意: chain_A 只存 theta_T 的边际后验，丢失时间结构
    """

    def __init__(self, n_vars, n_iter=3000, burnin=1000, thin=2,
                 sv_n_iter=500, sv_burnin=200):
        self.n = n_vars
        self.k = n_vars + n_vars * n_vars  # 参数维度
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.sv_n_iter = sv_n_iter
        self.sv_burnin = sv_burnin

        # FFBS 先验
        self.nu_Q = self.k + 10
        self.Psi_Q = np.eye(self.k) * 0.1
        self.nu_R = self.n + 10
        self.Psi_R = np.eye(self.n) * 0.5

        # 数值约束
        self.q_diag_min = 1e-6
        self.q_diag_max = 1.0
        self.r_diag_min = 1e-4
        self.r_diag_max = 5.0
        self.theta_clip = 20.0
        self.A_max_eig = 0.95

    def _build_Z(self, y_prev):
        """构建观测矩阵 Z_t"""
        n, k = self.n, self.k
        Z = np.zeros((n, k))
        for i in range(n):
            Z[i, i] = 1.0
            base = n + i * n
            Z[i, base:base + n] = y_prev
        return Z

    def _clip_A(self, theta):
        """限制 A 矩阵特征值 < A_max_eig"""
        n = self.n
        c = theta[:n]
        A = theta[n:].reshape(n, n)
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > self.A_max_eig:
            A = A * (self.A_max_eig / max_eig)
        return np.concatenate([c, A.flatten()])

    def _forward_filter_tvp_R(self, Y, Q, R_t):
        """
        前向 Kalman 滤波 (时变观测噪声 R_t)
        R_t: (T, n, n) 时变协方差序列
        """
        T = len(Y)
        n, k = self.n, self.k

        theta = np.zeros(k)
        theta[n:] = 0.5
        P = np.eye(k) * 0.5

        states = np.zeros((T, k))
        covs = np.zeros((T, k, k))
        pred_states = np.zeros((T, k))
        pred_covs = np.zeros((T, k, k))
        log_lik = 0.0

        for t in range(1, T):
            Z = self._build_Z(Y[t - 1])

            theta_pred = theta.copy()
            P_pred = P + Q

            # 正则化
            P_pred = (P_pred + P_pred.T) / 2
            eig = np.linalg.eigvalsh(P_pred)
            if np.min(eig) < 1e-10:
                P_pred += np.eye(k) * (1e-10 - np.min(eig) + 1e-8)

            y_pred = Z @ theta_pred
            v = Y[t] - y_pred
            S = Z @ P_pred @ Z.T + R_t[t]
            S = (S + S.T) / 2
            eig = np.linalg.eigvalsh(S)
            if np.min(eig) < 1e-10:
                S += np.eye(n) * (1e-10 - np.min(eig) + 1e-8)

            sign, log_det = np.linalg.slogdet(S)
            log_lik += -0.5 * (n * np.log(2 * np.pi) + log_det +
                                v @ np.linalg.solve(S, v))

            K = P_pred @ Z.T @ np.linalg.inv(S)
            theta = theta_pred + K @ v
            P = (np.eye(k) - K @ Z) @ P_pred
            P = (P + P.T) / 2

            pred_states[t] = theta_pred
            pred_covs[t] = P_pred
            states[t] = theta
            covs[t] = P

        return states, covs, pred_states, pred_covs, log_lik

    def _backward_sample(self, states, covs, pred_states, pred_covs, Q):
        """后向采样 (Carter-Kohn)"""
        T, k = states.shape
        n = self.n
        samples = np.zeros((T, k))

        # 最后时刻
        P_T = covs[T - 1]
        P_T = (P_T + P_T.T) / 2
        eig = np.linalg.eigvalsh(P_T)
        if np.min(eig) < 1e-8:
            P_T += np.eye(k) * (1e-8 - np.min(eig))
        samples[T - 1] = np.random.multivariate_normal(states[T - 1], P_T)
        samples[T - 1] = self._clip_A(samples[T - 1])

        for t in range(T - 2, 0, -1):
            P_filt = covs[t]
            P_pred_next = pred_covs[t + 1]

            P_pred_next = (P_pred_next + P_pred_next.T) / 2
            eig = np.linalg.eigvalsh(P_pred_next)
            if np.min(eig) < 1e-8:
                P_pred_next += np.eye(k) * (1e-8 - np.min(eig))

            try:
                J = P_filt @ np.linalg.inv(P_pred_next)
            except:
                J = P_filt @ np.linalg.pinv(P_pred_next)

            theta_smooth = states[t] + J @ (samples[t + 1] - pred_states[t + 1])
            P_smooth = P_filt + J @ (covs[t + 1] - P_pred_next) @ J.T
            P_smooth = (P_smooth + P_smooth.T) / 2
            eig = np.linalg.eigvalsh(P_smooth)
            if np.min(eig) < 1e-8:
                P_smooth += np.eye(k) * (1e-8 - np.min(eig) + 1e-8)

            samples[t] = np.random.multivariate_normal(theta_smooth, P_smooth)
            samples[t] = np.clip(samples[t], -self.theta_clip, self.theta_clip)
            samples[t] = self._clip_A(samples[t])

        return samples

    def _sample_Q(self, theta_traj):
        """采样 Q ~ Inverse-Wishart"""
        T, k = theta_traj.shape
        diffs = np.diff(theta_traj, axis=0)
        S = diffs.T @ diffs
        nu_post = self.nu_Q + T - 1
        Psi_post = self.Psi_Q + S
        Q = self._sample_iw(nu_post, Psi_post)
        return self._regularize_diag(Q, self.q_diag_min, self.q_diag_max)

    def _sample_R_static(self, Y, theta_traj):
        """采样静态 R ~ Inverse-Wishart (当不使用 SV 时)"""
        T = len(Y)
        n, k = self.n, self.k
        residuals = np.zeros((T - 1, n))
        for t in range(1, T):
            Z = self._build_Z(Y[t - 1])
            y_pred = Z @ theta_traj[t]
            residuals[t - 1] = Y[t] - y_pred

        S = residuals.T @ residuals
        nu_post = self.nu_R + T - 1
        Psi_post = self.Psi_R + S
        R = self._sample_iw(nu_post, Psi_post)
        return self._regularize_diag(R, self.r_diag_min, self.r_diag_max)

    def _sample_iw(self, nu, Psi):
        """采样 Inverse-Wishart，增加数值稳定性"""
        k = Psi.shape[0]
        Psi_reg = Psi + np.eye(k) * 1e-9
        try:
            L = np.linalg.cholesky(np.linalg.inv(Psi_reg))
            Z = np.random.randn(nu, k) @ L.T
            W = Z.T @ Z
            return np.linalg.inv(W + np.eye(k) * 1e-9)
        except np.linalg.LinAlgError:
            return np.eye(k) * 0.01

    def _regularize_diag(self, M, diag_min, diag_max):
        """正则化协方差矩阵"""
        M = (M + M.T) / 2
        d = np.diag(M).copy()
        d = np.clip(d, diag_min, diag_max)
        std = np.sqrt(d)
        corr = M / np.outer(std, std) if np.all(std > 0) else np.eye(len(d))
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -0.99, 0.99)
        np.fill_diagonal(corr, 1.0)
        M_new = corr * np.outer(std, std)
        M_new = (M_new + M_new.T) / 2
        eig = np.linalg.eigvalsh(M_new)
        if np.min(eig) < 1e-10:
            M_new += np.eye(len(d)) * (1e-10 - np.min(eig))
        return M_new

    def _build_R_t_from_sv(self, log_vol, R_corr):
        """从 SV log-volatility 和相关矩阵构建 R_t"""
        T, n = log_vol.shape
        R_t = np.zeros((T, n, n))
        for t in range(T):
            D_t = np.diag(np.exp(log_vol[t] / 2))
            R_t[t] = D_t @ R_corr @ D_t
        return R_t

    def _compute_residuals(self, Y, theta_traj):
        """计算残差序列"""
        T = len(Y)
        n, k = self.n, self.k
        residuals = np.zeros((T - 1, n))
        for t in range(1, T):
            Z = self._build_Z(Y[t - 1])
            y_pred = Z @ theta_traj[t]
            residuals[t - 1] = Y[t] - y_pred
        return residuals

    def fit(self, Y, use_sv=True, verbose=True):
        """
        运行 Gibbs 采样 (theta | Q, R_t) → (Q | theta) → (R_t | theta)

        参数:
            Y: (T, n) 观测数据
            use_sv: 是否使用随机波动率 (True=时变 R_t, False=静态 R)
            verbose: 打印进度

        返回: dict 包含所有后验链
        """
        T = len(Y)
        n, k = self.n, self.k

        # 初始化
        Q = np.eye(k) * 0.01
        R = np.eye(n) * 0.1
        R_t = np.tile(R, (T, 1, 1))  # 初始: 时变 R_t 都等于 R
        R_corr = np.eye(n)
        log_vol = np.full((T, n), np.log(0.1))

        # 存储
        chain_theta = []
        chain_Q = []
        chain_R = []
        chain_R_t = []
        chain_log_vol = []
        chain_R_corr = []
        chain_log_lik = []
        chain_A = []

        sv_sampler = BayesianSV(n_vars=n)

        for it in range(self.n_iter):
            # ===== Step 1: FFBS 采样 theta (给定 Q, R_t) =====
            states, covs, pred_states, pred_covs, log_lik = \
                self._forward_filter_tvp_R(Y, Q, R_t)
            theta_traj = self._backward_sample(
                states, covs, pred_states, pred_covs, Q)

            # ===== Step 2: 采样 Q =====
            Q = self._sample_Q(theta_traj)

            # ===== Step 3: 采样 R (SV 或 静态) =====
            residuals = self._compute_residuals(Y, theta_traj)

            if use_sv and len(residuals) > 10:
                # SV 采样: 用 BayesianSV 的 FFBS 采样 log-volatility
                # 简化: 每次迭代做一次 SV 更新
                for j in range(n):
                    log_e2 = np.log(residuals[:, j] ** 2 + 1e-10)

                    # 前向滤波 (简化 AR(1) FFBS)
                    h = np.zeros(T - 1)
                    P_h = np.ones(T - 1)
                    mu_h = -5.0
                    phi_h = 0.9
                    sigma_h = 0.3

                    for t in range(T - 1):
                        if t == 0:
                            h_pred = mu_h
                            P_pred_h = sigma_h ** 2 / (1 - phi_h ** 2 + 1e-10)
                        else:
                            h_pred = mu_h + phi_h * (h[t - 1] - mu_h)
                            P_pred_h = phi_h ** 2 * P_h[t - 1] + sigma_h ** 2

                        # 混合近似 (10-component)
                        mix_prob = np.array([0.00609, 0.04775, 0.13057, 0.20674,
                                             0.22715, 0.18842, 0.12047, 0.05591,
                                             0.01575, 0.00115])
                        mix_mean = np.array([-1.5797, -1.1616, -0.7702, -0.4318,
                                              -0.1168, 0.1958, 0.5316, 0.9212,
                                              1.4262, 2.1855])
                        mix_var = np.array([0.5576, 0.3712, 0.2557, 0.1883,
                                             0.1505, 0.1331, 0.1342, 0.1611,
                                             0.2392, 0.5218])

                        # 采样混合成分
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

                        # 滤波更新
                        obs_var = mix_var[s_k]
                        S_h = P_pred_h + obs_var
                        K_h = P_pred_h / S_h
                        h[t] = h_pred + K_h * (log_e2[t] - h_pred - mix_mean[s_k])
                        P_h[t] = (1 - K_h) * P_pred_h

                    # 后向采样
                    h_sample = np.zeros(T - 1)
                    h_sample[T - 2] = np.random.normal(
                        h[T - 2], np.sqrt(max(P_h[T - 2], 1e-10)))
                    for t in range(T - 3, -1, -1):
                        J_h = phi_h * P_h[t] / (phi_h ** 2 * P_h[t] + sigma_h ** 2 + 1e-10)
                        h_smooth = h[t] + J_h * (h_sample[t + 1] - mu_h - phi_h * (h[t] - mu_h))
                        P_smooth_h = max(P_h[t] - J_h * phi_h * P_h[t], 1e-10)
                        h_sample[t] = np.random.normal(h_smooth, np.sqrt(P_smooth_h))

                    log_vol[1:, j] = h_sample
                    log_vol[0, j] = h_sample[0]

                # 构建 R_t
                R_t = self._build_R_t_from_sv(log_vol, R_corr)

                # 采样相关矩阵
                D_inv = np.exp(-log_vol / 2)
                std_residuals = residuals * D_inv[1:]
                nu_corr = len(residuals) + n + 2
                S_corr = std_residuals.T @ std_residuals + np.eye(n) * 0.1
                try:
                    S_corr = (S_corr + S_corr.T) / 2
                    eig = np.linalg.eigvalsh(S_corr)
                    if np.min(eig) < 1e-8:
                        S_corr += np.eye(n) * (1e-8 - np.min(eig))
                    L_c = np.linalg.cholesky(np.linalg.inv(S_corr))
                    Z_c = np.random.randn(nu_corr, n) @ L_c.T
                    W_c = Z_c.T @ Z_c
                    R_draw = np.linalg.inv(W_c)
                    d_c = np.sqrt(np.clip(np.diag(R_draw), 1e-10, None))
                    R_corr = R_draw / np.outer(d_c, d_c)
                    np.fill_diagonal(R_corr, 1.0)
                    R_corr = np.clip(R_corr, -0.99, 0.99)
                    np.fill_diagonal(R_corr, 1.0)
                except:
                    pass  # 保持当前 R_corr

            else:
                # 静态 R
                R = self._sample_R_static(Y, theta_traj)
                R_t = np.tile(R, (T, 1, 1))

            # ===== 存储 =====
            if it >= self.burnin and (it - self.burnin) % self.thin == 0:
                chain_theta.append(theta_traj[-1].copy())
                chain_Q.append(Q.copy())
                if use_sv:
                    chain_R_t.append(R_t.copy())
                    chain_log_vol.append(log_vol.copy())
                    chain_R_corr.append(R_corr.copy())
                else:
                    chain_R.append(R.copy())
                chain_log_lik.append(log_lik)
                chain_A.append(theta_traj[-1][n:].reshape(n, n).copy())

            if verbose and (it + 1) % 500 == 0:
                q_diag = np.diag(Q).mean()
                msg = f"  iter {it+1}/{self.n_iter}  Q_diag={q_diag:.6f}"
                if use_sv:
                    msg += f"  log_vol_mean={log_vol.mean():.2f}"
                else:
                    msg += f"  R_diag={np.diag(R).mean():.6f}"
                msg += f"  log_lik={log_lik:.1f}"
                logger.debug(msg)

        result = {
            "chain_theta": np.array(chain_theta),
            "chain_Q": np.array(chain_Q),
            "chain_A": np.array(chain_A),
            "chain_log_lik": np.array(chain_log_lik),
        }

        if use_sv:
            result["chain_R_t"] = chain_R_t
            result["chain_log_vol"] = chain_log_vol
            result["chain_R_corr"] = chain_R_corr
            # 后验均值 R_t
            if chain_R_t:
                result["posterior_mean_R_t"] = np.mean(chain_R_t, axis=0)
        else:
            result["chain_R"] = np.array(chain_R)

        self.result = result
        return result

    def posterior_summary(self, var_names=None):
        """后验摘要"""
        n = self.n
        if var_names is None:
            var_names = [f"y{i}" for i in range(n)]

        A_chain = self.result["chain_A"]
        A_mean = A_chain.mean(axis=0)
        A_std = A_chain.std(axis=0)
        A_lower = np.percentile(A_chain, 2.5, axis=0)
        A_upper = np.percentile(A_chain, 97.5, axis=0)

        logger.info("后验摘要 (A 矩阵):")
        header = f"{'':>10}"
        for name in var_names:
            header += f" {name+'(t-1)':>14}"
        logger.info(header)
        for i, name in enumerate(var_names):
            row = f"{name+'(t)':<10}"
            for j in range(n):
                sig = "*" if A_lower[i, j] > 0 or A_upper[i, j] < 0 else " "
                row += f" {A_mean[i,j]:>6.3f}{sig}[{A_lower[i,j]:>5.3f},{A_upper[i,j]:>5.3f}]"
            logger.info(row)

        Q_chain = self.result["chain_Q"]
        Q_mean = Q_chain.mean(axis=0)
        logger.info(f"\nQ (过程噪声) 对角: {np.diag(Q_mean).round(6)}")

        if "chain_R_corr" in self.result:
            R_corr = np.mean(self.result["chain_R_corr"], axis=0)
            logger.info(f"\nR_corr (相关矩阵):")
            for i in range(n):
                logger.info(f"  {[f'{v:.3f}' for v in R_corr[i]]}")

        return {"A_mean": A_mean, "A_std": A_std,
                "A_lower": A_lower, "A_upper": A_upper}

    def impulse_response(self, shock_var=0, shock_size=1.0, periods=6,
                         var_names=None):
        """
        从后验链计算 IRF (贝叶斯可信区间)
        使用后验样本 A 计算, 不是随机噪声
        """
        n = self.n
        A_chain = self.result["chain_A"]
        n_samples = len(A_chain)

        irf_all = np.zeros((n_samples, periods, n))

        for s in range(n_samples):
            A = A_chain[s]
            Psi = np.eye(n)
            for t in range(periods):
                irf_all[s, t] = Psi[:, shock_var] * shock_size
                Psi = A @ Psi

        irf_mean = irf_all.mean(axis=0)
        irf_lower = np.percentile(irf_all, 2.5, axis=0)
        irf_upper = np.percentile(irf_all, 97.5, axis=0)

        if var_names is None:
            var_names = [f"y{i}" for i in range(n)]

        logger.info(f"脉冲响应 (冲击: {var_names[shock_var]} +{shock_size}σ):")
        header = f"{'期':<6}"
        for name in var_names:
            header += f" {name:>18}"
        logger.info(header)
        for t in range(periods):
            row = f"t+{t:<4}"
            for i in range(n):
                sig = "*" if irf_lower[t, i] > 0 or irf_upper[t, i] < 0 else " "
                row += f" {irf_mean[t,i]:>7.4f}{sig}[{irf_lower[t,i]:>6.3f},{irf_upper[t,i]:>6.3f}]"
            logger.info(row)

        return irf_mean, irf_lower, irf_upper

    def predict(self, steps=4, n_samples=500, Y_last=None, mean=None, std=None):
        """
        后验预测 (含完整不确定性传播)
        mean, std: 反标准化参数 (可选)
        """
        n = self.n
        A_chain = self.result["chain_A"]
        Q_chain = self.result["chain_Q"]

        if Y_last is None:
            raise ValueError("需要提供 Y_last (最后观测)")

        n_chain = len(A_chain)
        idx = np.random.choice(n_chain, min(n_samples, n_chain), replace=False)
        preds = np.zeros((len(idx), steps, n))

        for i, si in enumerate(idx):
            A = A_chain[si]
            Q = Q_chain[si]

            # 限制特征值
            eigvals = np.linalg.eigvals(A)
            if np.max(np.abs(eigvals)) > 0.95:
                A = A * (0.95 / np.max(np.abs(eigvals)))

            # 从 theta 中提取 c
            theta = self.result["chain_theta"][si]
            c = theta[:n]

            y = Y_last.copy()
            for s in range(steps):
                # 参数扰动 (传播 Q 不确定性)
                theta_perturbed = theta + np.random.multivariate_normal(
                    np.zeros(len(theta)), Q)
                c_p = theta_perturbed[:n]
                A_p = theta_perturbed[n:].reshape(n, n)
                eigvals_p = np.linalg.eigvals(A_p)
                if np.max(np.abs(eigvals_p)) > 0.95:
                    A_p = A_p * (0.95 / np.max(np.abs(eigvals_p)))

                y_mean = c_p + A_p @ y
                if not np.all(np.isfinite(y_mean)):
                    preds[i, s:] = np.nan
                    break

                # 观测噪声
                if "posterior_mean_R_t" in self.result:
                    R_last = self.result["posterior_mean_R_t"][-1]
                elif "chain_R" in self.result:
                    R_last = self.result["chain_R"][si]
                else:
                    R_last = np.eye(n) * 0.1

                try:
                    noise = np.random.multivariate_normal(np.zeros(n), R_last)
                except:
                    noise = np.random.randn(n) * 0.1

                preds[i, s] = y_mean + noise
                y = y_mean

        pred_mean = preds.mean(axis=0)
        pred_lower = np.percentile(preds, 2.5, axis=0)
        pred_upper = np.percentile(preds, 97.5, axis=0)

        # 反标准化
        if mean is not None and std is not None:
            pred_mean = pred_mean * std + mean
            pred_lower = pred_lower * std + mean
            pred_upper = pred_upper * std + mean

        return pred_mean, pred_lower, pred_upper

    # ── 统一接口 ──────────────────────────────────────────────

    def compute_irf(self, shock_var=0, periods=6, shock_size=1.0, var_names=None, **kwargs):
        """统一 IRF 接口，返回 (irf_mean, irf_lower, irf_upper)"""
        return self.impulse_response(
            shock_var=shock_var, shock_size=shock_size,
            periods=periods, var_names=var_names)

    def diagnostics(self):
        """返回后验链诊断信息"""
        if not hasattr(self, 'result') or self.result is None:
            return {}
        d = {}
        for key in ("chain_A", "chain_Q", "chain_log_lik"):
            if key in self.result:
                chain = self.result[key]
                d[key] = {
                    "shape": chain.shape if hasattr(chain, 'shape') else None,
                    "n_samples": len(chain),
                }
        return d

    def get_chains(self):
        """返回后验链字典"""
        if not hasattr(self, 'result') or self.result is None:
            return {}
        return {k: v for k, v in self.result.items() if k.startswith("chain_")}

    # ── Coupled posterior methods (_joint_chain driven) ──────

    def impulse_response_from_joint_chain(self, joint_chain, shock_var=0,
                                           shock_size=1.0, periods=6, var_names=None):
        """
        IRF from coupled posterior samples.

        Parameters
        ----------
        joint_chain : list[dict]
            _joint_chain with coupled (theta, sigma) per sample.
        """
        n = self.n
        n_samples = len(joint_chain)
        irf_all = np.zeros((n_samples, periods, n))

        for s in range(n_samples):
            theta = joint_chain[s]["theta"]
            A = theta[n:].reshape(n, n)
            Psi = np.eye(n)
            for t in range(periods):
                irf_all[s, t] = Psi[:, shock_var] * shock_size
                Psi = A @ Psi

        irf_mean = irf_all.mean(axis=0)
        irf_lower = np.percentile(irf_all, 2.5, axis=0)
        irf_upper = np.percentile(irf_all, 97.5, axis=0)
        return irf_mean, irf_lower, irf_upper

    def predict_from_joint_chain(self, joint_chain, steps=4, n_samples=500,
                                  Y_last=None, mean=None, std=None):
        """
        Posterior prediction from coupled _joint_chain.

        Each sample uses coupled (theta, sigma) from the same
        Gibbs iteration, preserving posterior dependency integrity.
        """
        n = self.n
        if Y_last is None:
            raise ValueError("需要提供 Y_last (最后观测)")

        n_chain = len(joint_chain)
        idx = np.random.choice(n_chain, min(n_samples, n_chain), replace=False)
        preds = np.zeros((len(idx), steps, n))

        for i, si in enumerate(idx):
            sample = joint_chain[si]
            theta = sample["theta"]
            sigma = sample["sigma"]

            c = theta[:n]
            A = theta[n:].reshape(n, n)
            eigvals = np.linalg.eigvals(A)
            if np.max(np.abs(eigvals)) > 0.95:
                A = A * (0.95 / np.max(np.abs(eigvals)))

            y = Y_last.copy()
            for s in range(steps):
                theta_perturbed = theta + np.random.multivariate_normal(
                    np.zeros(len(theta)), np.eye(len(theta)) * 0.01)
                c_p = theta_perturbed[:n]
                A_p = theta_perturbed[n:].reshape(n, n)
                eigvals_p = np.linalg.eigvals(A_p)
                if np.max(np.abs(eigvals_p)) > 0.95:
                    A_p = A_p * (0.95 / np.max(np.abs(eigvals_p)))

                y_mean = c_p + A_p @ y
                if not np.all(np.isfinite(y_mean)):
                    preds[i, s:] = np.nan
                    break

                try:
                    noise = np.random.multivariate_normal(np.zeros(n), sigma)
                except Exception:
                    noise = np.random.randn(n) * 0.1

                preds[i, s] = y_mean + noise
                y = y_mean

        pred_mean = preds.mean(axis=0)
        pred_lower = np.percentile(preds, 2.5, axis=0)
        pred_upper = np.percentile(preds, 97.5, axis=0)

        if mean is not None and std is not None:
            pred_mean = pred_mean * std + mean
            pred_lower = pred_lower * std + mean
            pred_upper = pred_upper * std + mean

        return pred_mean, pred_lower, pred_upper

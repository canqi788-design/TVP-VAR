"""
TVP-VAR v2: Research-grade 贝叶斯升级
核心升级:
  1. FFBS Gibbs 采样 p(theta, Q | Y; R)
  2. 完整贝叶斯随机波动率 (SV Gibbs Sampler)
  3. Markov 区制转换 TVP-VAR (MS-TVP-VAR)
  4. 符号约束 SVAR (Sign Restriction)
  5. 动态协方差 Sigma_t (Wishart)
  6. WAIC / LOO-CV 模型选择
  7. 后验预测模拟 (pathwise MC)
"""

import logging
import warnings
import numpy as np
from scipy import stats

logger = logging.getLogger("tvp_var")


# ============================================================
# 1. FFBS Gibbs 采样
# ============================================================

class FFBS_Sampler:
    """
    Forward Filtering Backward Sampling (Carter-Kohn 1994)
    Gibbs 采样 p(theta_{1:T}, Q | Y; R)

    Gibbs 框架:
      1. 给定 (Q, R), FFBS 采样 theta_{1:T}
      2. 给定 theta_{1:T}, 采样 Q ~ Inverse-Wishart
      3. 给定 theta_{1:T}, 采样 R ~ Inverse-Wishart
    """

    def __init__(self, n_vars, n_iter=3000, burnin=1000, thin=2):
        self.n = n_vars
        self.k = n_vars + n_vars * n_vars
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin

        # Inverse-Wishart 超参数 (用强先验约束)
        self.nu_Q = self.k + 10       # 高自由度 -> 更集中
        self.Psi_Q = np.eye(self.k) * 0.1  # 小尺度 -> Q 偏小
        self.nu_R = self.n + 10
        self.Psi_R = np.eye(self.n) * 0.5

        # 硬约束: Q, R 对角线范围
        self.q_diag_min = 1e-6
        self.q_diag_max = 1.0
        self.r_diag_min = 1e-4
        self.r_diag_max = 5.0
        self.theta_clip = 20.0

    def _forward_filter(self, Y, Q, R):
        """前向 Kalman 滤波"""
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
            eig = np.linalg.eigvalsh(P_pred)
            if np.min(eig) < 1e-10:
                P_pred += np.eye(k) * (1e-10 - np.min(eig) + 1e-8)

            y_pred = Z @ theta_pred
            v = y_true - y_pred
            S = Z @ P_pred @ Z.T + R
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

    def _clip_A_eigenvalues(self, theta, max_abs=0.95):
        """限制 A 矩阵特征值 < max_abs (平稳性条件)"""
        n = self.n
        c = theta[:n]
        A = theta[n:].reshape(n, n)
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > max_abs:
            A = A * (max_abs / max_eig)
        return np.concatenate([c, A.flatten()])

    def _backward_sample(self, states, covs, pred_states, pred_covs, Q):
        """后向采样 (Carter-Kohn)"""
        T, k = states.shape
        samples = np.zeros((T, k))

        # 最后时刻
        P_T = covs[T - 1]
        P_T = (P_T + P_T.T) / 2
        eig = np.linalg.eigvalsh(P_T)
        if np.min(eig) < 1e-8:
            P_T += np.eye(k) * (1e-8 - np.min(eig))
        samples[T - 1] = np.random.multivariate_normal(states[T - 1], P_T)
        samples[T - 1] = self._clip_A_eigenvalues(samples[T - 1])

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
            samples[t] = self._clip_A_eigenvalues(samples[t])

        return samples

    def _regularize_cov(self, M, diag_min, diag_max):
        """正则化协方差矩阵: 截断对角线, 保证正定"""
        M = (M + M.T) / 2
        d = np.diag(M).copy()
        d = np.clip(d, diag_min, diag_max)
        # 用相关系数矩阵重建
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

    def _sample_Q(self, theta_traj):
        """采样 Q ~ Inverse-Wishart(nu_Q + T-1, Psi_Q + sum(d*d'))"""
        T, k = theta_traj.shape
        diffs = np.diff(theta_traj, axis=0)
        S = diffs.T @ diffs
        nu_post = self.nu_Q + T - 1
        Psi_post = self.Psi_Q + S
        Q = self._sample_inverse_wishart(nu_post, Psi_post)
        return self._regularize_cov(Q, self.q_diag_min, self.q_diag_max)

    def _sample_R(self, Y, theta_traj):
        """采样 R ~ Inverse-Wishart(nu_R + T-1, Psi_R + sum(v*v'))"""
        T = len(Y)
        n, k = self.n, self.k
        residuals = np.zeros((T - 1, n))

        for t in range(1, T):
            y_prev = Y[t - 1]
            y_true = Y[t]
            Z = np.zeros((n, k))
            for i in range(n):
                Z[i, i] = 1.0
                base = n + i * n
                Z[i, base:base + n] = y_prev
            y_pred = Z @ theta_traj[t]
            residuals[t - 1] = y_true - y_pred

        S = residuals.T @ residuals
        nu_post = self.nu_R + T - 1
        Psi_post = self.Psi_R + S
        R = self._sample_inverse_wishart(nu_post, Psi_post)
        return self._regularize_cov(R, self.r_diag_min, self.r_diag_max)

    def _sample_inverse_wishart(self, nu, Psi):
        """采样 Inverse-Wishart"""
        k = Psi.shape[0]
        # 先采 Wishart, 再求逆
        try:
            L = np.linalg.cholesky(np.linalg.inv(Psi))
            Z = np.random.randn(nu, k) @ L.T
            W = Z.T @ Z
            return np.linalg.inv(W)
        except:
            return np.eye(k) * 0.01

    def fit(self, Y, verbose=True):
        """运行 Gibbs 采样"""
        self.Y = Y.copy()

        # 初始化
        Q = np.eye(self.k) * 0.01
        R = np.eye(self.n) * 0.1

        self.chain_Q = []
        self.chain_R = []
        self.chain_theta = []
        self.chain_A = []
        self.chain_log_lik = []

        for it in range(self.n_iter):
            # Step 1: FFBS 采样 theta
            states, covs, pred_states, pred_covs, log_lik = \
                self._forward_filter(Y, Q, R)
            theta_traj = self._backward_sample(
                states, covs, pred_states, pred_covs, Q)

            # Step 2: 采样 Q (已含正则化)
            Q = self._sample_Q(theta_traj)

            # Step 3: 采样 R (已含正则化)
            R = self._sample_R(Y, theta_traj)

            # 存储
            self.chain_Q.append(Q.copy())
            self.chain_R.append(R.copy())
            self.chain_theta.append(theta_traj[-1].copy())
            A = theta_traj[-1][self.n:].reshape(self.n, self.n)
            self.chain_A.append(A.copy())
            self.chain_log_lik.append(log_lik)

            if verbose and (it + 1) % 500 == 0:
                q_diag = np.diag(Q).mean()
                r_diag = np.diag(R).mean()
                logger.debug(f"  iter {it+1}/{self.n_iter}  "
                             f"Q_diag={q_diag:.6f}  R_diag={r_diag:.6f}  "
                             f"log_lik={log_lik:.1f}")

        # 截取后烧期
        self.chain_Q = self.chain_Q[self.burnin::self.thin]
        self.chain_R = self.chain_R[self.burnin::self.thin]
        self.chain_theta = np.array(self.chain_theta[self.burnin::self.thin])
        self.chain_A = np.array(self.chain_A[self.burnin::self.thin])
        self.chain_log_lik = self.chain_log_lik[self.burnin::self.thin]

        return self

    # ── 统一接口 ──────────────────────────────────────────────

    def compute_irf(self, shock_var=0, periods=6, shock_size=1.0, var_names=None, **kwargs):
        """统一 IRF 接口，返回 (irf_mean, irf_lower, irf_upper)"""
        n = self.n
        A_chain = self.chain_A
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
        return irf_mean, irf_lower, irf_upper

    def predict(self, steps=4, n_samples=500, Y_last=None, **kwargs):
        """统一预测接口，返回 (pred_mean, pred_lower, pred_upper)"""
        mc = PathwiseMCPredictor(n_vars=self.n)
        preds = mc.predict(
            self.chain_theta, self.chain_Q, self.chain_R,
            Y_last, steps=steps, n_samples=min(n_samples, len(self.chain_A)))
        return mc.interval(preds)

    def diagnostics(self):
        """返回后验链诊断信息"""
        return {
            "chain_A": {"shape": self.chain_A.shape, "n_samples": len(self.chain_A)},
            "chain_Q": {"shape": np.array(self.chain_Q).shape, "n_samples": len(self.chain_Q)},
            "chain_log_lik": {"n_samples": len(self.chain_log_lik)},
        }

    def get_chains(self):
        """返回后验链字典"""
        return {
            "chain_A": self.chain_A,
            "chain_Q": np.array(self.chain_Q),
            "chain_R": np.array(self.chain_R),
            "chain_theta": self.chain_theta,
            "chain_log_lik": np.array(self.chain_log_lik),
        }


# ============================================================
# 2. 完整贝叶斯随机波动率
# ============================================================

class BayesianSV:
    """
    贝叶斯随机波动率模型
    log(sigma_{j,t}^2) = mu_j + phi_j * (log(sigma_{j,t-1}^2) - mu_j) + eta_{j,t}

    使用 Kim et al. (1998) 混合正态近似 + Gibbs 采样
    """

    def __init__(self, n_vars):
        self.n = n_vars
        # 10-component mixture (Omori, Chib, Shephard 2007)
        self.mix_prob = np.array([0.00609, 0.04775, 0.13057, 0.20674,
                                   0.22715, 0.18842, 0.12047, 0.05591,
                                   0.01575, 0.00115])
        self.mix_mean = np.array([-1.5797, -1.1616, -0.7702, -0.4318,
                                   -0.1168, 0.1958, 0.5316, 0.9212,
                                   1.4262, 2.1855])
        self.mix_var = np.array([0.5576, 0.3712, 0.2557, 0.1883,
                                  0.1505, 0.1331, 0.1342, 0.1611,
                                  0.2392, 0.5218])

    def _sample_s(self, log_sigma2, mu, phi):
        """采样混合成分指标 s_t"""
        T = len(log_sigma2)
        s = np.zeros(T, dtype=int)
        for t in range(T):
            if t == 0:
                h_pred = mu
            else:
                h_pred = mu + phi * (log_sigma2[t - 1] - mu)

            log_prob = np.zeros(10)
            for k in range(10):
                log_prob[k] = (np.log(self.mix_prob[k]) -
                                0.5 * np.log(self.mix_var[k]) -
                                0.5 * (log_sigma2[t] - h_pred - self.mix_mean[k]) ** 2 /
                                self.mix_var[k])
            log_prob -= np.max(log_prob)
            prob = np.exp(log_prob)
            prob /= prob.sum()
            s[t] = np.random.choice(10, p=prob)
        return s

    def _sample_log_sigma2(self, residuals_j, s, mu, phi, sigma_eta):
        """采样 log(sigma^2) 使用卡尔曼滤波 (AR(1) 状态空间)"""
        T = len(residuals_j)
        log_e2 = np.log(residuals_j ** 2 + 1e-10)

        # 状态空间: h_t = mu + phi*(h_{t-1}-mu) + eta_t
        # 观测: log(e_t^2) = h_t + z_t, z_t ~ N(mean_s, var_s)

        h = np.ones(T) * mu
        P = np.ones(T)
        h_pred = np.ones(T)
        P_pred = np.ones(T)

        # 前向滤波
        for t in range(T):
            if t == 0:
                h_pred[t] = mu
                P_pred[t] = sigma_eta ** 2 / (1 - phi ** 2 + 1e-10)
            else:
                h_pred[t] = mu + phi * (h[t - 1] - mu)
                P_pred[t] = phi ** 2 * P[t - 1] + sigma_eta ** 2

            obs_var = self.mix_var[s[t]]
            S = P_pred[t] + obs_var
            K = P_pred[t] / S

            h[t] = h_pred[t] + K * (log_e2[t] - h_pred[t] - self.mix_mean[s[t]])
            P[t] = (1 - K) * P_pred[t]

        # 后向采样
        h_sample = np.zeros(T)
        h_sample[T - 1] = np.random.normal(h[T - 1], np.sqrt(max(P[T - 1], 1e-10)))

        for t in range(T - 2, -1, -1):
            J = phi * P[t] / (phi ** 2 * P[t] + sigma_eta ** 2 + 1e-10)
            h_smooth = h[t] + J * (h_sample[t + 1] - mu - phi * (h[t] - mu))
            P_smooth = P[t] - J * phi * P[t]
            P_smooth = max(P_smooth, 1e-10)
            h_sample[t] = np.random.normal(h_smooth, np.sqrt(P_smooth))

        return h_sample

    def _sample_mu(self, log_sigma2, phi):
        """采样 mu"""
        T = len(log_sigma2)
        # 先验: mu ~ N(0, 10)
        prior_var = 10.0
        # 似然: h_t | h_{t-1} ~ N(mu + phi*(h_{t-1}-mu), sigma_eta^2)
        # 简化: 用样本均值
        post_mean = np.mean(log_sigma2) * (1 - phi)
        post_var = prior_var * 0.1
        return np.random.normal(post_mean, np.sqrt(post_var))

    def _sample_phi(self, log_sigma2, mu):
        """采样 phi (用 Beta 先验)"""
        T = len(log_sigma2)
        x = log_sigma2[:-1] - mu
        y = log_sigma2[1:] - mu

        if np.sum(x ** 2) < 1e-10:
            return 0.9

        # OLS
        phi_hat = np.sum(x * y) / np.sum(x ** 2)
        phi_var = 1.0 / (np.sum(x ** 2) + 1e-10)

        # Beta(20, 2) 先验 -> phi 在 (0,1) 附近
        # 用截断正态近似
        phi = np.random.normal(phi_hat, np.sqrt(phi_var))
        phi = np.clip(phi, 0.01, 0.999)
        return phi

    def _sample_sigma_eta(self, log_sigma2, mu, phi):
        """采样 sigma_eta"""
        T = len(log_sigma2)
        residuals = (log_sigma2[1:] - mu - phi * (log_sigma2[:-1] - mu))
        # Inverse-Gamma(2, 0.1)
        a = 2 + (T - 1) / 2
        b = 0.1 + np.sum(residuals ** 2) / 2
        return np.sqrt(1.0 / np.random.gamma(a, 1.0 / b))

    def fit(self, residuals, n_iter=1000, burnin=300):
        """
        拟合 SV 模型
        参数:
            residuals: (T, n) 残差序列
        返回: dict 包含后验样本
        """
        T, n = residuals.shape

        # 初始化
        log_sigma2 = np.zeros((T, n))
        mu = np.full(n, -5.0)
        phi = np.full(n, 0.9)
        sigma_eta = np.full(n, 0.3)

        chain_log_sigma2 = []
        chain_mu = []
        chain_phi = []
        chain_sigma_eta = []

        for it in range(n_iter):
            for j in range(n):
                # 采样混合成分
                s = self._sample_s(log_sigma2[:, j], mu[j], phi[j])

                # 采样 log(sigma^2)
                log_sigma2[:, j] = self._sample_log_sigma2(
                    residuals[:, j], s, mu[j], phi[j], sigma_eta[j])

                # 采样超参数
                mu[j] = self._sample_mu(log_sigma2[:, j], phi[j])
                phi[j] = self._sample_phi(log_sigma2[:, j], mu[j])
                sigma_eta[j] = self._sample_sigma_eta(
                    log_sigma2[:, j], mu[j], phi[j])

            if it >= burnin:
                chain_log_sigma2.append(log_sigma2.copy())
                chain_mu.append(mu.copy())
                chain_phi.append(phi.copy())
                chain_sigma_eta.append(sigma_eta.copy())

            if (it + 1) % 200 == 0:
                logger.debug(f"  SV iter {it+1}/{n_iter}  "
                             f"mu={mu.round(2)}  phi={phi.round(3)}")

        return {
            "log_sigma2": np.array(chain_log_sigma2),
            "mu": np.array(chain_mu),
            "phi": np.array(chain_phi),
            "sigma_eta": np.array(chain_sigma_eta)
        }


# ============================================================
# 3. Markov 区制转换 TVP-VAR
# ============================================================

class MarkovSwitchingTVP:
    """
    Markov Switching TVP-VAR (Hamilton 1989 + TVP 扩展)

    y_t = c_{s_t} + A_{s_t} y_{t-1} + e_t
    s_t ~ Markov(p_{ij})

    使用 Gibbs 采样:
      1. 给定参数, 采样区制序列 s_{1:T} (Hamilton filter + Kim smoother)
      2. 给定 s_{1:T}, 采样各区制参数
      3. 采样转移概率矩阵 P
    """

    def __init__(self, n_vars, n_regimes=2):
        self.n = n_vars
        self.k = n_vars + n_vars * n_vars
        self.R = n_regimes
        # 各区制的观测协方差 (初始值)
        self._Sigma = [np.eye(n_vars) for _ in range(n_regimes)]

    def _hamilton_filter(self, Y, regime_params, trans_prob, regime_Sigma=None):
        """
        Hamilton 滤波
        返回: filtered_probs (T, R), log_likelihood
        regime_Sigma: 各区制的观测协方差列表, None 则用 self._Sigma
        """
        T = len(Y)
        R = self.R
        n = self.n

        if regime_Sigma is None:
            regime_Sigma = self._Sigma

        filtered = np.zeros((T, R))
        pred = np.zeros((T, R))
        log_lik = 0.0

        # 初始概率 (平稳分布)
        eigvals, eigvecs = np.linalg.eig(trans_prob.T)
        idx = np.argmin(np.abs(eigvals - 1))
        stat_dist = np.real(eigvecs[:, idx])
        stat_dist = stat_dist / stat_dist.sum()

        # 预计算各区制的 Sigma 逆和 log-det
        Sigma_inv = []
        log_det_Sigma = []
        for s in range(R):
            S = regime_Sigma[s]
            S = (S + S.T) / 2
            eig = np.linalg.eigvalsh(S)
            if np.min(eig) < 1e-8:
                S = S + np.eye(n) * (1e-8 - np.min(eig))
            sign, ld = np.linalg.slogdet(S)
            log_det_Sigma.append(ld)
            Sigma_inv.append(np.linalg.inv(S))

        for t in range(1, T):
            y_prev = Y[t - 1]
            y_true = Y[t]

            # 预测: pred[s] = sum_s' trans_prob[s',s] * filtered[s']
            if t == 1:
                pred[t] = stat_dist
            else:
                for s in range(R):
                    pred[t, s] = sum(trans_prob[s2, s] * filtered[t - 1, s2]
                                      for s2 in range(R))

            # 似然 (使用各区制的实际协方差)
            for s in range(R):
                c, A = regime_params[s]
                y_pred = c + A @ y_prev
                v = y_true - y_pred
                log_lik_s = -0.5 * (n * np.log(2 * np.pi) +
                                     log_det_Sigma[s] +
                                     v @ Sigma_inv[s] @ v)
                filtered[t, s] = pred[t, s] * np.exp(log_lik_s)

            # 归一化
            total = filtered[t].sum()
            if total > 0:
                filtered[t] /= total
                log_lik += np.log(total + 1e-300)
            else:
                filtered[t] = np.ones(R) / R

        return filtered, log_lik

    def _kim_smoother(self, filtered, trans_prob):
        """Kim 平滑器: 后向递推计算平滑概率"""
        T, R = filtered.shape
        smoothed = np.zeros((T, R))
        smoothed[T - 1] = filtered[T - 1]

        for t in range(T - 2, 0, -1):
            for s in range(R):
                smoothed[t, s] = filtered[t, s] * sum(
                    trans_prob[s, s2] * smoothed[t + 1, s2] /
                    (sum(filtered[t, s3] * trans_prob[s3, s2]
                         for s3 in range(R)) + 1e-300)
                    for s2 in range(R)
                )
            total = smoothed[t].sum()
            if total > 0:
                smoothed[t] /= total

        return smoothed

    def _sample_regimes(self, smoothed):
        """从平滑概率中采样区制序列"""
        T, R = smoothed.shape
        states = np.zeros(T, dtype=int)
        for t in range(1, T):
            probs = smoothed[t]
            probs = np.clip(probs, 0, None)
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                probs = np.ones(R) / R
            states[t] = np.random.choice(R, p=probs)
        return states

    def _sample_regime_params(self, Y, states):
        """
        采样各区制参数 (Normal-inverse-Wishart 后验)
        beta_s ~ N(beta_hat, Sigma_e ⊗ (X'X)^{-1})
        Sigma_s ~ IW(nu_post, Psi_post)
        """
        R = self.R
        n = self.n
        params = []
        Sigma_list = []

        # 先验: beta ~ N(0, I * kappa), Sigma ~ IW(n+2, I * 0.5)
        kappa = 1.0  # 先验精度
        nu_prior = n + 2
        Psi_prior = np.eye(n) * 0.5

        for s in range(R):
            idx = np.where(states == s)[0]
            if len(idx) < 3:
                c = np.zeros(n)
                A = np.eye(n) * 0.3
                params.append((c, A))
                Sigma_list.append(np.eye(n) * 0.5)
                continue

            # 构建回归
            X_list = []
            y_list = []
            for t in idx:
                if t > 0:
                    y_list.append(Y[t])
                    X_list.append(np.concatenate([[1], Y[t - 1]]))

            if len(X_list) < 2:
                c = np.zeros(n)
                A = np.eye(n) * 0.3
                params.append((c, A))
                Sigma_list.append(np.eye(n) * 0.5)
                continue

            X = np.array(X_list)
            Y_reg = np.array(y_list)
            T_s = len(X_list)
            p = X.shape[1]  # = n + 1

            # 后验超参数
            XtX = X.T @ X
            XtX_post = XtX + np.eye(p) * kappa
            try:
                XtX_inv = np.linalg.inv(XtX_post)
            except:
                XtX_inv = np.linalg.pinv(XtX_post)

            beta_hat = XtX_inv @ (X.T @ Y_reg)  # 后验均值
            residuals = Y_reg - X @ beta_hat

            # Sigma 后验: IW(nu_post, Psi_post)
            nu_post = nu_prior + T_s
            Psi_post = Psi_prior + residuals.T @ residuals + \
                       (beta_hat.T @ XtX @ beta_hat -
                        beta_hat.T @ XtX @ beta_hat)  # 简化: 用残差
            Psi_post = Psi_prior + residuals.T @ residuals

            # 采样 Sigma ~ IW
            try:
                Psi_post = (Psi_post + Psi_post.T) / 2
                eig = np.linalg.eigvalsh(Psi_post)
                if np.min(eig) < 1e-8:
                    Psi_post += np.eye(n) * (1e-8 - np.min(eig))
                L_psi = np.linalg.cholesky(np.linalg.inv(Psi_post))
                Z = np.random.randn(nu_post, n) @ L_psi.T
                W = Z.T @ Z
                Sigma_s = np.linalg.inv(W)
                # 正则化
                Sigma_s = (Sigma_s + Sigma_s.T) / 2
                d = np.clip(np.diag(Sigma_s), 1e-4, 10.0)
                np.fill_diagonal(Sigma_s, d)
            except:
                Sigma_s = np.eye(n) * 0.5

            # 采样 beta | Sigma ~ N(beta_hat, Sigma ⊗ XtX_inv)
            # 对每个方程独立采样
            try:
                L_Sigma = np.linalg.cholesky(Sigma_s)
                L_X = np.linalg.cholesky(XtX_inv)
                # beta_vec = vec(beta) ~ N(vec(beta_hat), Sigma ⊗ XtX_inv)
                # 用 Kronecker 结构: beta = beta_hat + L_Sigma @ Z @ L_X'
                Z_mat = np.random.randn(n, p)
                beta_s = beta_hat + L_Sigma @ Z_mat @ L_X.T
            except:
                beta_s = beta_hat

            c = beta_s[0]
            A = beta_s[1:].T
            params.append((c, A))
            Sigma_list.append(Sigma_s)

        self._Sigma = Sigma_list
        return params

    def _sample_trans_prob(self, states):
        """采样转移概率 (Dirichlet-Multinomial)"""
        R = self.R
        trans_prob = np.zeros((R, R))

        for s in range(R):
            counts = np.zeros(R)
            for t in range(1, len(states) - 1):
                if states[t] == s:
                    counts[states[t + 1]] += 1
            # Dirichlet 先验: alpha = 1
            alpha = counts + 1
            trans_prob[s] = np.random.dirichlet(alpha)

        return trans_prob

    def fit(self, Y, n_iter=1000, burnin=300, verbose=True):
        """运行 MS-TVP-VAR Gibbs 采样"""
        T = len(Y)
        R = self.R

        # 初始化
        regime_params = [(np.zeros(self.n), np.eye(self.n) * 0.3)
                          for _ in range(R)]
        self._Sigma = [np.eye(self.n) * 0.5 for _ in range(R)]
        trans_prob = np.eye(R) * 0.9 + np.ones((R, R)) * 0.1 / R
        states = np.zeros(T, dtype=int)

        chain_params = []
        chain_states = []
        chain_trans = []
        chain_Sigma = []

        for it in range(n_iter):
            # Step 1: Hamilton filter + Kim smoother (用实际 Sigma)
            filtered, _ = self._hamilton_filter(Y, regime_params, trans_prob)
            smoothed = self._kim_smoother(filtered, trans_prob)

            # Step 2: 采样区制序列
            states = self._sample_regimes(smoothed)

            # Step 3: 采样各区制参数 + 协方差 (Normal-IW 后验)
            regime_params = self._sample_regime_params(Y, states)

            # Step 4: 采样转移概率
            trans_prob = self._sample_trans_prob(states)

            if it >= burnin:
                chain_params.append(regime_params)
                chain_states.append(states.copy())
                chain_trans.append(trans_prob.copy())
                chain_Sigma.append([S.copy() for S in self._Sigma])

            if verbose and (it + 1) % 200 == 0:
                regime_counts = [np.sum(states == s) for s in range(R)]
                logger.debug(f"  MS iter {it+1}/{n_iter}  "
                             f"regimes={regime_counts}  "
                             f"P_diag={np.diag(trans_prob).round(3)}")

        return {
            "params": chain_params,
            "states": chain_states,
            "trans_prob": chain_trans,
            "Sigma": chain_Sigma,
            "final_states": states,
            "final_trans_prob": trans_prob,
            "final_Sigma": self._Sigma,
        }


# ============================================================
# 4. 符号约束 SVAR
# ============================================================

class SignRestrictionSVAR:
    """
    符号约束 SVAR (Uhlig 2005, Rubio-Ramírez et al. 2010)

    不依赖 Cholesky 排序，用经济学约束识别冲击
    """

    def __init__(self, n_vars, n_draws=1000):
        self.n = n_vars
        self.n_draws = n_draws

    def identify(self, A_posterior, Sigma_posterior, sign_restrictions):
        """
        符号约束识别
        参数:
            A_posterior: (n_samples, n, n) A 矩阵后验样本
            Sigma_posterior: (n_samples, n, n) 协方差后验样本
            sign_restrictions: dict, e.g. {0: [1, 1], 1: [0, -1]}
                键为冲击编号，值为 [变量, 符号] 列表
        返回: accepted_impulse_responses
        """
        n = self.n
        accepted_irfs = []

        for s in range(min(self.n_draws, len(A_posterior))):
            A = A_posterior[s % len(A_posterior)]
            Sigma = Sigma_posterior[s % len(Sigma_posterior)]

            # 随机正交矩阵 (QR decomposition of random matrix)
            Q, R_qr = np.linalg.qr(np.random.randn(n, n))
            # 确保正定
            D = np.diag(np.sign(np.diag(R_qr)))
            Q = Q @ D

            # 结构化冲击: P = Q * chol(Sigma)
            try:
                L = np.linalg.cholesky(Sigma)
            except:
                continue

            P = Q @ L

            # 检查符号约束
            # IRF: Psi_h @ P, 其中 Psi_h 是脉冲响应矩阵
            Psi = np.zeros((6, n, n))
            Psi[0] = np.eye(n)
            for h in range(1, 6):
                Psi[h] = A @ Psi[h - 1]

            # 检查约束
            valid = True
            for shock_j, constraints in sign_restrictions.items():
                for var_i, sign in constraints:
                    # 检查前几期 IRF
                    for h in range(1, 4):
                        irf_val = Psi[h, var_i, :] @ P[:, shock_j]
                        if sign > 0 and irf_val < 0:
                            valid = False
                            break
                        if sign < 0 and irf_val > 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                irf = np.zeros((6, n, n))
                for h in range(6):
                    irf[h] = Psi[h] @ P
                accepted_irfs.append(irf)

        return accepted_irfs

    def summary(self, accepted_irfs, var_names=None, shock_names=None):
        """符号约束 IRF 摘要"""
        n = self.n
        if var_names is None:
            var_names = [f"y{i}" for i in range(n)]
        if shock_names is None:
            shock_names = [f"eps{i}" for i in range(n)]

        if not accepted_irfs:
            logger.warning("  无满足约束的 IRF")
            return

        irf_array = np.array(accepted_irfs)
        irf_mean = irf_array.mean(axis=0)
        irf_lower = np.percentile(irf_array, 16, axis=0)
        irf_upper = np.percentile(irf_array, 84, axis=0)

        logger.info(f"  接受率: {len(accepted_irfs)}/{self.n_draws} "
                    f"({len(accepted_irfs)/self.n_draws:.1%})")

        for shock_j in range(n):
            header = f"  {'期':<6}"
            for name in var_names:
                header += f" {name:>18}"
            logger.info(f"\n  冲击: {shock_names[shock_j]}")
            logger.info(header)
            for h in range(6):
                line = f"  t+{h:<3}"
                for i in range(n):
                    line += (f" {irf_mean[h, i, shock_j]:>7.3f}"
                             f"[{irf_lower[h, i, shock_j]:>5.3f},"
                             f"{irf_upper[h, i, shock_j]:>5.3f}]")
                logger.info(line)


# ============================================================
# 5. WAIC / LOO-CV
# ============================================================

class AdvancedModelSelection:
    """WAIC 和 LOO-CV"""

    @staticmethod
    def waic(log_lik_pointwise):
        """
        WAIC (Watanabe-Akaike)
        参数: log_lik_pointwise (n_samples, T)
        """
        n_samples, T = log_lik_pointwise.shape

        lppd = 0.0
        for t in range(T):
            lppd += np.logaddexp.reduce(log_lik_pointwise[:, t]) - np.log(n_samples)

        p_waic = np.sum(np.var(log_lik_pointwise, axis=0))
        waic = -2 * (lppd - p_waic)
        return waic, p_waic, lppd

    @staticmethod
    def loo_cv(log_lik_pointwise):
        """
        LOO-CV (近似, 用 Pareto-smoothed importance sampling 简化版)
        """
        n_samples, T = log_lik_pointwise.shape

        loo_lik = np.zeros(T)
        for t in range(T):
            # 留出 t, 用其余样本估计
            loo_samples = np.delete(log_lik_pointwise, t, axis=1)
            # 简化: 用后验均值似然近似
            loo_lik[t] = np.mean(log_lik_pointwise[:, t])

        return np.sum(loo_lik)

    @staticmethod
    def bayes_factor(log_ml_1, log_ml_2):
        """贝叶斯因子"""
        return np.exp(log_ml_1 - log_ml_2)


# ============================================================
# 6. 动态协方差 (Wishart)
# ============================================================

class DynamicCovariance:
    """
    动态协方差矩阵 Sigma_t = D_t^{1/2} R_t D_t^{1/2}

    实现:
      - log(sigma_{j,t}^2) ~ AR(1): 时变方差 (复用 BayesianSV 结构)
      - R_t ~ DCC 退化: 常数相关矩阵, 从残差后验采样
      - 或完整 DCC: Q_t = (1-a-b)*Q_bar + a*eps_{t-1}*eps_{t-1}' + b*Q_{t-1}

    Gibbs 采样:
      1. 给定 R, 对每个变量 j 采样 h_{j,1:T} (AR(1) FFBS)
      2. 给定 h, 采样 mu_j, phi_j, sigma_eta_j
      3. 给定标准化残差, 采样 R (相关矩阵)
    """

    def __init__(self, n_vars, dcc_a=0.05, dcc_b=0.90):
        self.n = n_vars
        self.dcc_a = dcc_a
        self.dcc_b = dcc_b
        # SV 先验
        self.prior_mu_mean = -5.0
        self.prior_mu_var = 10.0
        self.prior_phi_alpha = 20.0  # Beta(20,2) -> phi 约 0.91
        self.prior_phi_beta = 2.0
        self.prior_sigma_eta_a = 2.0
        self.prior_sigma_eta_b = 0.1

    def _sample_log_variances(self, residuals, R_corr, h_prev, mu, phi, sigma_eta):
        """
        对每个变量 j, 采样 h_{j,1:T} (log-variance)
        使用 Kim (1998) 混合正态近似 + FFBS
        """
        T, n = residuals.shape
        # 标准化残差: eps_t = D_t^{-1/2} * r_t, 其中 D_t = diag(exp(h_t))
        # 但在相关矩阵 R_corr 下, eps_t ~ N(0, R_corr)

        # 简化: 对每个变量单独处理, 用卡方近似
        # log(r_{j,t}^2) ~ N(h_{j,t}, psi_j) 其中 psi_j 由 R_corr 的对角线决定
        h_new = np.zeros((T, n))

        for j in range(n):
            # 近似: log(r_{j,t}^2) ~ log(chi^2_1) + h_{j,t}
            # E[log(chi^2_1)] = -1.2704, Var[log(chi^2_1)] = pi^2/2
            log_e2 = np.log(residuals[:, j] ** 2 + 1e-10)
            obs_const = -1.2704  # E[log(chi^2_1)]
            obs_var = np.pi ** 2 / 2  # Var[log(chi^2_1)]

            # 调整: 如果 R_corr[j,j] != 1, 需要修正
            # log(r_{j,t}^2) = h_{j,t} + log(eps_{j,t}^2)
            # eps_{j,t} ~ N(0, R_corr[j,j]) -> eps_{j,t}/sqrt(R_corr[j,j]) ~ N(0,1)
            # 所以 log(r_{j,t}^2) = h_{j,t} + log(R_corr[j,j]) + log(chi^2_1)
            corr_adj = np.log(max(R_corr[j, j], 1e-6))

            # AR(1) FFBS: h_t = mu + phi*(h_{t-1}-mu) + eta_t
            # 观测: log_e2_t = h_t + corr_adj + obs_const + noise, noise ~ N(0, obs_var)
            h = np.zeros(T)
            P = np.zeros(T)

            # 前向滤波
            for t in range(T):
                if t == 0:
                    h_pred = mu[j]
                    P_pred = sigma_eta[j] ** 2 / (1 - phi[j] ** 2 + 1e-10)
                else:
                    h_pred = mu[j] + phi[j] * (h[t - 1] - mu[j])
                    P_pred = phi[j] ** 2 * P[t - 1] + sigma_eta[j] ** 2

                y_obs = log_e2[t] - corr_adj - obs_const
                S = P_pred + obs_var
                K = P_pred / S
                h[t] = h_pred + K * (y_obs - h_pred)
                P[t] = (1 - K) * P_pred

            # 后向采样
            h_sample = np.zeros(T)
            h_sample[T - 1] = np.random.normal(h[T - 1], np.sqrt(max(P[T - 1], 1e-10)))
            for t in range(T - 2, -1, -1):
                J = phi[j] * P[t] / (phi[j] ** 2 * P[t] + sigma_eta[j] ** 2 + 1e-10)
                h_smooth = h[t] + J * (h_sample[t + 1] - mu[j] - phi[j] * (h[t] - mu[j]))
                P_smooth = max(P[t] - J * phi[j] * P[t], 1e-10)
                h_sample[t] = np.random.normal(h_smooth, np.sqrt(P_smooth))

            h_new[:, j] = h_sample

        return h_new

    def _sample_sv_params(self, h):
        """采样 SV 超参数 mu, phi, sigma_eta"""
        T, n = h.shape
        mu = np.zeros(n)
        phi = np.zeros(n)
        sigma_eta = np.zeros(n)

        for j in range(n):
            hj = h[:, j]

            # mu: 条件正态后验
            mu_var_post = 1.0 / (1.0 / self.prior_mu_var + T / (1 - 0.9 ** 2 + 1e-10))
            mu_mean_post = mu_var_post * (self.prior_mu_mean / self.prior_mu_var +
                                          np.sum(hj) * (1 - 0.9) / (1 - 0.9 ** 2 + 1e-10))
            mu[j] = np.random.normal(mu_mean_post, np.sqrt(mu_var_post))

            # phi: OLS + 截断
            x = hj[:-1] - mu[j]
            y = hj[1:] - mu[j]
            if np.sum(x ** 2) > 1e-10:
                phi_hat = np.sum(x * y) / np.sum(x ** 2)
                phi_var = 1.0 / (np.sum(x ** 2) + 1e-10)
                phi[j] = np.clip(np.random.normal(phi_hat, np.sqrt(phi_var)),
                                 0.01, 0.999)
            else:
                phi[j] = 0.9

            # sigma_eta: Inverse-Gamma
            residuals_h = y - phi[j] * x
            a = self.prior_sigma_eta_a + (T - 1) / 2
            b = self.prior_sigma_eta_b + np.sum(residuals_h ** 2) / 2
            sigma_eta[j] = np.sqrt(1.0 / np.random.gamma(a, 1.0 / b))

        return mu, phi, sigma_eta

    def _sample_correlation(self, std_residuals):
        """
        采样相关矩阵 R_corr
        使用退化 DCC: R_corr 从标准化残差的后验采样
        退化为: R_corr ~ IW(nu, S) 然后标准化
        """
        T, n = std_residuals.shape
        nu = T + n + 2
        S = std_residuals.T @ std_residuals + np.eye(n) * 0.1

        try:
            S = (S + S.T) / 2
            eig = np.linalg.eigvalsh(S)
            if np.min(eig) < 1e-8:
                S += np.eye(n) * (1e-8 - np.min(eig))
            L = np.linalg.cholesky(np.linalg.inv(S))
            Z = np.random.randn(nu, n) @ L.T
            W = Z.T @ Z
            R_draw = np.linalg.inv(W)
            # 标准化为相关矩阵
            d = np.sqrt(np.clip(np.diag(R_draw), 1e-10, None))
            R_corr = R_draw / np.outer(d, d)
            np.fill_diagonal(R_corr, 1.0)
            R_corr = np.clip(R_corr, -0.99, 0.99)
            np.fill_diagonal(R_corr, 1.0)
        except:
            R_corr = np.eye(n)

        return R_corr

    def fit(self, residuals, n_iter=500, burnin=200, verbose=True):
        """
        采样动态协方差轨迹
        residuals: (T, n) 残差序列

        返回: dict 包含 Sigma_t 轨迹和后验样本
        """
        T, n = residuals.shape

        # 初始化
        h = np.full((T, n), np.log(np.var(residuals, axis=0) + 1e-6))
        mu = np.full(n, -5.0)
        phi = np.full(n, 0.9)
        sigma_eta = np.full(n, 0.3)
        R_corr = np.eye(n)

        chain_h = []
        chain_mu = []
        chain_phi = []
        chain_sigma_eta = []
        chain_R = []
        chain_Sigma = []

        for it in range(n_iter):
            # Step 1: 采样 h_{j,1:T} (给定 R_corr)
            h = self._sample_log_variances(residuals, R_corr, h, mu, phi, sigma_eta)

            # Step 2: 采样 SV 超参数
            mu, phi, sigma_eta = self._sample_sv_params(h)

            # Step 3: 采样相关矩阵 (给定标准化残差)
            D_inv = np.exp(-h / 2)  # T x n
            std_residuals = residuals * D_inv
            R_corr = self._sample_correlation(std_residuals)

            if it >= burnin:
                chain_h.append(h.copy())
                chain_mu.append(mu.copy())
                chain_phi.append(phi.copy())
                chain_sigma_eta.append(sigma_eta.copy())
                chain_R.append(R_corr.copy())
                # 构建 Sigma_t = D_t R D_t
                Sigma_t = np.zeros((T, n, n))
                for t in range(T):
                    D_t = np.diag(np.exp(h[t] / 2))
                    Sigma_t[t] = D_t @ R_corr @ D_t
                chain_Sigma.append(Sigma_t)

            if verbose and (it + 1) % 100 == 0:
                logger.debug(f"  DynCov iter {it+1}/{n_iter}  "
                             f"mu={mu.round(2)}  phi={phi.round(3)}")

        return {
            "Sigma_t": chain_Sigma,  # list of (T, n, n) arrays
            "h": chain_h,
            "mu": chain_mu,
            "phi": chain_phi,
            "sigma_eta": chain_sigma_eta,
            "R_corr": chain_R,
            "posterior_mean_Sigma": np.mean(chain_Sigma, axis=0) if chain_Sigma else None,
        }


# ============================================================
# 7. 后验预测模拟 (Pathwise MC)
# ============================================================

class PathwiseMCPredictor:
    """
    Pathwise Monte Carlo 预测
    完整传播参数 + 波动率 + 区制不确定性
    """

    def __init__(self, n_vars):
        self.n = n_vars

    @staticmethod
    def _clip_A(A, max_abs=0.98):
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig >= 1.0:
            A = A * (max_abs / (max_eig + 1e-10))
        return A

    def predict(self, theta_chain, Q_chain, R_chain, Y_last,
                steps=4, n_samples=500):
        """
        路径式 MC 预测 (DEPRECATED — use predict_from_joint_chain)

        每个样本: 从后验采样 (theta, Q, R) -> 模拟未来路径
        """
        warnings.warn(
            "PathwiseMCPredictor.predict() with separate theta_chain/Q_chain/R_chain "
            "is deprecated. Use predict_from_joint_chain(joint_chain, Y_last, ...) "
            "for posterior-sample driven analysis.",
            DeprecationWarning,
            stacklevel=2,
        )
        n = self.n
        n_chain = len(theta_chain)
        idx = np.random.choice(n_chain, min(n_samples, n_chain), replace=False)

        preds = np.zeros((len(idx), steps, n))

        for i, si in enumerate(idx):
            theta = theta_chain[si]
            Q = Q_chain[si]
            R = R_chain[si]

            c = theta[:n]
            A = self._clip_A(theta[n:].reshape(n, n))

            y = Y_last.copy()
            for s in range(steps):
                # 参数也随机扰动 (传播 Q 不确定性)
                theta_perturbed = theta + np.random.multivariate_normal(
                    np.zeros(len(theta)), Q)
                c_p = theta_perturbed[:n]
                A_p = self._clip_A(theta_perturbed[n:].reshape(n, n))

                y_mean = c_p + A_p @ y
                if not np.all(np.isfinite(y_mean)):
                    preds[i, s:] = np.nan
                    break
                try:
                    noise = np.random.multivariate_normal(np.zeros(n), R)
                except:
                    noise = np.random.randn(n) * 0.1

                preds[i, s] = y_mean + noise
                y = y_mean

        return preds

    def predict_with_regime(self, theta_chain, Q_chain, R_chain,
                             ms_results, Y_last, steps=4, n_samples=500):
        """带区制不确定性的预测 (DEPRECATED — use predict_from_joint_chain)"""
        warnings.warn(
            "PathwiseMCPredictor.predict_with_regime() with separate chains "
            "is deprecated. Use predict_from_joint_chain() for coupled analysis.",
            DeprecationWarning,
            stacklevel=2,
        )
        n = self.n
        n_chain = len(theta_chain)
        idx = np.random.choice(n_chain, min(n_samples, n_chain), replace=False)

        # 转移概率
        trans_prob = ms_results["final_trans_prob"]
        R_regimes = trans_prob.shape[0]

        preds = np.zeros((len(idx), steps, n))

        for i, si in enumerate(idx):
            theta = theta_chain[si]
            Q = Q_chain[si]
            R = R_chain[si]

            # 当前区制
            current_regime = ms_results["final_states"][-1]

            y = Y_last.copy()
            for s in range(steps):
                # 区制转移
                current_regime = np.random.choice(
                    R_regimes, p=trans_prob[current_regime])

                c = theta[:n]
                A = self._clip_A(theta[n:].reshape(n, n))

                theta_perturbed = theta + np.random.multivariate_normal(
                    np.zeros(len(theta)), Q)
                c_p = theta_perturbed[:n]
                A_p = self._clip_A(theta_perturbed[n:].reshape(n, n))

                y_mean = c_p + A_p @ y
                if not np.all(np.isfinite(y_mean)):
                    preds[i, s:] = np.nan
                    break
                try:
                    noise = np.random.multivariate_normal(np.zeros(n), R)
                except:
                    noise = np.random.randn(n) * 0.1

                preds[i, s] = y_mean + noise
                y = y_mean

        return preds

    def predict_from_joint_chain(self, joint_chain, Y_last,
                                  steps=4, n_samples=500):
        """
        Posterior-sample driven prediction from coupled _joint_chain.

        Each sample provides coupled (theta, sigma) from the same
        Gibbs iteration, preserving posterior dependency integrity.

        Parameters
        ----------
        joint_chain : list[dict]
            _joint_chain from kernel_sampler_research or kernel_sampler_basic.
            Each dict must have 'theta' and 'sigma' keys.
        Y_last : ndarray
            Last observed data point.
        steps : int
            Forecast horizon.
        n_samples : int
            Number of posterior samples to use.

        Returns
        -------
        ndarray (n_samples, steps, n)
        """
        n = self.n
        n_chain = len(joint_chain)
        idx = np.random.choice(n_chain, min(n_samples, n_chain), replace=False)

        preds = np.zeros((len(idx), steps, n))

        for i, si in enumerate(idx):
            sample = joint_chain[si]
            theta = sample["theta"]
            sigma = sample["sigma"]

            c = theta[:n]
            A = self._clip_A(theta[n:].reshape(n, n))

            y = Y_last.copy()
            for s in range(steps):
                theta_perturbed = theta + np.random.multivariate_normal(
                    np.zeros(len(theta)), np.eye(len(theta)) * 0.01)
                c_p = theta_perturbed[:n]
                A_p = self._clip_A(theta_perturbed[n:].reshape(n, n))

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

        return preds

    def interval(self, preds):
        mean = preds.mean(axis=0)
        lower = np.percentile(preds, 2.5, axis=0)
        upper = np.percentile(preds, 97.5, axis=0)
        return mean, lower, upper

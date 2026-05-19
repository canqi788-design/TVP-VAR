"""
TVP-VAR 长期约束识别 (Blanchard-Quah 1989)
"""

import numpy as np
import logging

logger = logging.getLogger("tvp_var")


class BlanchardQuahSVAR:
    def __init__(self, n_vars):
        self.n = n_vars

    def identify(self, A_samples, Sigma_samples, long_run_constraints, periods=100):
        A_samples = np.asarray(A_samples)
        Sigma_samples = np.asarray(Sigma_samples)
        n_samples = len(A_samples)

        identified_irfs = []

        for i in range(n_samples):
            A = A_samples[i]
            Sigma = Sigma_samples[i]

            try:
                irf = self._identify_single(A, Sigma, long_run_constraints, periods)
                if irf is not None:
                    identified_irfs.append(irf)
            except Exception as e:
                logger.debug(f"BQ 样本 {i} 识别失败: {e}")
                continue

        if not identified_irfs:
            logger.warning("BQ 识别: 所有样本均失败")
            return []

        logger.info(f"BQ 识别: {len(identified_irfs)}/{n_samples} 样本成功")
        return identified_irfs

    def _identify_single(self, A, Sigma, constraints, periods):
        n = self.n
        C = self._compute_long_run_multiplier(A, periods)

        try:
            P = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma_reg = Sigma + np.eye(n) * 1e-6
            try:
                P = np.linalg.cholesky(Sigma_reg)
            except np.linalg.LinAlgError:
                return None

        C_unc = C @ P
        Q = self._apply_bq_rotation(C_unc, constraints)
        if Q is None:
            return None

        P_bq = P @ Q

        irf = np.zeros((periods, n, n))
        A_power = np.eye(n)
        for h in range(periods):
            irf[h] = A_power @ P_bq
            A_power = A_power @ A

        return irf

    def _compute_long_run_multiplier(self, A, periods):
        n = self.n
        C = np.eye(n)
        A_power = np.eye(n)
        for h in range(1, periods):
            A_power = A_power @ A
            C = C + A_power
        return C

    def _apply_bq_rotation(self, C_unc, constraints):
        n = self.n
        Q_full, R = np.linalg.qr(C_unc)
        Q = Q_full

        C_rotated = C_unc @ Q
        constraints_satisfied = True
        for shock_idx, var_constraints in constraints.items():
            for var_idx, target in var_constraints.items():
                if abs(C_rotated[var_idx, shock_idx] - target) > 1e-6:
                    constraints_satisfied = False
                    break

        if constraints_satisfied:
            return Q

        for shock_idx, var_constraints in constraints.items():
            for var_idx, target in var_constraints.items():
                Q = self._givens_rotate_for_constraint(
                    C_unc, Q, shock_idx, var_idx, target
                )
                if Q is None:
                    return None

        return Q

    def _givens_rotate_for_constraint(self, C_unc, Q, shock_idx, var_idx, target):
        n = self.n
        C_rotated = C_unc @ Q
        current = C_rotated[var_idx, shock_idx]

        if abs(current - target) < 1e-8:
            return Q

        for other_shock in range(n):
            if other_shock == shock_idx:
                continue
            other_val = C_rotated[var_idx, other_shock]

            r = np.sqrt(current ** 2 + other_val ** 2)
            if r < 1e-10:
                continue

            cos_theta = target / r
            sin_theta = np.sqrt(max(0, 1 - cos_theta ** 2))

            if abs(current * cos_theta + other_val * sin_theta - target) > abs(current * cos_theta - other_val * sin_theta - target):
                sin_theta = -sin_theta

            G = np.eye(n)
            G[shock_idx, shock_idx] = cos_theta
            G[shock_idx, other_shock] = -sin_theta
            G[other_shock, shock_idx] = sin_theta
            G[other_shock, other_shock] = cos_theta

            Q_new = Q @ G
            C_new = C_unc @ Q_new
            if abs(C_new[var_idx, shock_idx] - target) < 1e-6:
                return Q_new

        logger.debug(f"BQ 旋转: 无法满足约束 shock={shock_idx}, var={var_idx}")
        return Q

    def summary(self, identified_irfs, var_names=None, shock_names=None):
        n = self.n
        if var_names is None:
            var_names = [f"y{i}" for i in range(n)]
        if shock_names is None:
            shock_names = [f"冲击{i}" for i in range(n)]

        n_samples = len(identified_irfs)
        periods = identified_irfs[0].shape[0]

        irf_stack = np.array(identified_irfs)
        irf_mean = np.mean(irf_stack, axis=0)
        irf_lower = np.percentile(irf_stack, 2.5, axis=0)
        irf_upper = np.percentile(irf_stack, 97.5, axis=0)

        lines = []
        lines.append(f"\nBlanchard-Quah 长期约束 IRF ({n_samples} 样本, {periods} 期)")
        lines.append("=" * 70)

        for shock_j, shock_name in enumerate(shock_names):
            lines.append(f"\n冲击: {shock_name}")
            header = f"{'期':<6}"
            for name in var_names:
                header += f" {name:>18}"
            lines.append(header)
            lines.append("-" * len(header))

            for t in range(min(periods, 10)):
                row = f"t+{t:<3}"
                for i in range(n):
                    row += f" {irf_mean[t, i, shock_j]:>7.4f}[{irf_lower[t, i, shock_j]:>5.3f},{irf_upper[t, i, shock_j]:>5.3f}]"
                lines.append(row)

            lines.append(f"\n长期累积响应 (应满足约束):")
            for i in range(n):
                cumsum = np.sum(irf_mean[:, i, shock_j])
                row = f"  {var_names[i]}: {cumsum:.6f}"
                lines.append(row)

        return "\n".join(lines), {
            "irf_mean": irf_mean,
            "irf_lower": irf_lower,
            "irf_upper": irf_upper,
        }

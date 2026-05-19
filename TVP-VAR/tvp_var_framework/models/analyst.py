import numpy as np
from tvp_var_framework.core.theta_layout import build_equation_layout, split_theta


class TVP_VAR_Analyst:
    def __init__(self, n_vars=3, n_exog=0, q=1e-5, r=1e-3):
        self.n = n_vars
        self.m = n_exog
        self.components_per_equation = 1 + self.n + self.m
        self.k = self.n * self.components_per_equation
        self.meta = build_equation_layout(self.n, self.m)

        self.theta = np.zeros(self.k)

        # Equation-by-equation layout: [c_i, A_i*, B_i*] per response.
        for i in range(self.n):
            self.theta[self.meta["lag_matrix_A"]["start"] + i * self.components_per_equation + i] = 0.5

        self.P = np.eye(self.k)
        self.Q = np.eye(self.k) * q
        self.R = np.eye(self.n) * r

        self.history = []

    def theta_matrix(self, theta=None):
        theta = self.theta if theta is None else theta
        return np.asarray(theta).reshape(self.n, self.components_per_equation)

    def split_theta(self, theta=None):
        theta = self.theta if theta is None else theta
        return split_theta(theta, self.meta)

    def _build_Z(self, y_prev, x_exog=None):
        x_exog = np.zeros(self.m) if x_exog is None else np.asarray(x_exog).reshape(-1)
        if len(x_exog) != self.m:
            raise ValueError(f"x_exog 维度不匹配: expected {self.m}, got {len(x_exog)}")

        regressors = np.hstack([1.0, np.asarray(y_prev).reshape(-1), x_exog])
        return np.kron(np.eye(self.n), regressors)

    def update(self, y_prev, y_true, x_exog=None):

        Z = self._build_Z(y_prev, x_exog=x_exog)

        # ===== prediction =====
        theta_pred = self.theta.copy()
        P_pred = self.P + self.Q

        y_pred = Z @ theta_pred

        v = y_true - y_pred

        S = Z @ P_pred @ Z.T + self.R
        K = P_pred @ Z.T @ np.linalg.inv(S)

        # ===== update =====
        self.theta = theta_pred + K @ v
        self.P = (np.eye(self.k) - K @ Z) @ P_pred

        # extract matrix
        c, A, B = self.split_theta()

        self.history.append({
            "c": c.copy(),
            "A": A.copy(),
            "B": B.copy(),
            "y_pred": y_pred,
            "innovation": v
        })

        return A

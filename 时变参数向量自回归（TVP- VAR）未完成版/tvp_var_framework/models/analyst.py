import numpy as np


class TVP_VAR_Analyst:
    def __init__(self, n_vars=3, q=1e-5, r=1e-3):
        self.n = n_vars
        self.k = n_vars + n_vars * n_vars

        self.theta = np.zeros(self.k)

        # 截距项初始化为 0
        self.theta[:self.n] = 0.0
        # 系数矩阵对角线初始化为 0.5
        for i in range(self.n):
            self.theta[self.n + i * self.n + i] = 0.5

        self.P = np.eye(self.k)
        self.Q = np.eye(self.k) * q
        self.R = np.eye(self.n) * r

        self.history = []

    def _build_Z(self, y_prev):
        Z = np.zeros((self.n, self.k))

        for i in range(self.n):
            Z[i, i] = 1.0
            base = self.n + i * self.n
            Z[i, base:base + self.n] = y_prev

        return Z

    def update(self, y_prev, y_true):

        Z = self._build_Z(y_prev)

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
        c = self.theta[:self.n]
        A = self.theta[self.n:].reshape(self.n, self.n)

        self.history.append({
            "c": c.copy(),
            "A": A.copy(),
            "y_pred": y_pred,
            "innovation": v
        })

        return A

"""Theta memory layout helpers.

The VARX bayesian path uses equation-by-equation flattened theta:
    row i = [c_i, A_i1..A_in, B_i1..B_im]
"""

import numpy as np


def build_equation_layout(n_endog, n_exog=0):
    n = int(n_endog)
    m = int(n_exog)
    p = 1 + n + m
    return {
        "layout": "equation_by_equation",
        "n_endog": n,
        "n_exog": m,
        "components_per_equation": p,
        "theta_dim": n * p,
        "total_dimension": n * p,
        "slices": {
            "intercept": slice(0, 1),
            "endog_lags": slice(1, 1 + n),
            "exog_inputs": slice(1 + n, p),
        },
        "intercept_c": {"column": 0},
        "lag_matrix_A": {"start": 1, "stop": 1 + n},
        "exog_matrix_B": {"start": 1 + n, "stop": p},
    }


def theta_matrix(theta, layout):
    return np.asarray(theta, dtype=float).reshape(
        layout["n_endog"], layout["components_per_equation"]
    )


def split_theta(theta, layout):
    mat = theta_matrix(theta, layout)
    n = int(layout["n_endog"])
    p = int(layout["components_per_equation"])
    c = mat[:, 0]
    A = mat[:, 1:1 + n]
    B = mat[:, 1 + n:p]
    return c, A, B


def extract_transition(theta, layout):
    return split_theta(theta, layout)[1]


def extract_intercept(theta, layout):
    return split_theta(theta, layout)[0]


def extract_exogenous(theta, layout):
    return split_theta(theta, layout)[2]

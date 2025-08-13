from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..core.estimator import OnlineEstimator
from ..core.utils import log_sum_exp


class OnlineEM(OnlineEstimator):
    """
    Online Expectation Maximization
    """

    def __init__(self, K: int, alpha: float = 0.65):
        self.K = K
        self.alpha = alpha
        self.t: int = 0
        self.theta: NDArray[np.float64] = np.full(K, 1.0 / K)

    def update(self, y: int, g_row: NDArray[np.float64]) -> None:
        self.t += 1

        log_pi_unnorm = np.log(self.theta) + np.log(g_row)
        log_den       = log_sum_exp(log_pi_unnorm)
        pi_post       = np.exp(log_pi_unnorm - log_den)

        gamma_t = self.t ** (-self.alpha)
        self.theta = (1.0 - gamma_t) * self.theta + gamma_t * pi_post
        return self.theta

    def current(self) -> NDArray[np.float64]:
        return self.theta.copy()
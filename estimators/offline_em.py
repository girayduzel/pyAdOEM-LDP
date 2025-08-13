from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..core.utils import log_sum_exp

class OfflineEM:
    """
    Offline Expectation Maximization
    """

    def __init__(self, K: int, M: int):
        self.K = K
        self.M = M

    def fit_from_logs(
        self,
        log_P_yx: NDArray[np.float64],   # shape (K, T)
        theta0: NDArray[np.float64] | None = None,
        return_path: bool = False,
    ) -> NDArray[np.float64]:
        K, T = log_P_yx.shape

        theta = (np.full(K, 1.0 / K) if theta0 is None else np.asarray(theta0, dtype=np.float64)).copy()
        theta = np.maximum(theta, 1e-300)
        Theta_path = np.zeros((self.M, K), dtype=np.float64) if return_path else None

        for m in range(self.M):
            # E-step
            log_Pi_post = log_P_yx + np.log(theta)[:, None]

            vmax = np.max(log_Pi_post, axis=0)
            log_den = vmax + np.log(np.exp(log_Pi_post - vmax).sum(axis=0))
            Pi_post = np.exp(log_Pi_post - log_den)

            # M-step
            theta = Pi_post.mean(axis=1)
            theta = np.maximum(theta, 1e-300)
            if Theta_path is not None:
                Theta_path[m] = theta

        if Theta_path is not None:
            return Theta_path
        return theta

    def fit(
        self,
        ys: list[int],
        g_rows: list[NDArray[np.float64]],
        theta0: NDArray[np.float64] | None = None,
        return_path: bool = False,
    ) -> NDArray[np.float64]:
        if len(ys) == 0:
            return (np.full(self.K, 1.0 / self.K) if not return_path else np.zeros((self.M, self.K)))
        log_P_yx = np.log(np.stack(g_rows, axis=1))
        return self.fit_from_logs(log_P_yx, theta0=theta0, return_path=return_path)

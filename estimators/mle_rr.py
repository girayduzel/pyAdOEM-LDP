# AdOEM_LDP/estimators/mle_rr.py
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

class MLERR:
    """
    Exact MLE for GRR.
    """
    def __init__(self, K: int, eps: float):
        self.K = K
        self.eps = eps
        self.S: NDArray[np.int64] = np.zeros(K, dtype=np.int64)
        self.t: int = 0

    # Online
    def update(self, y: int) -> None:
        self.S[y] += 1
        self.t += 1

    def current(self) -> NDArray[np.float64]:
        return self._theta_from_counts(self.S)

    # Offline
    def fit(self, Y: NDArray[np.int_]) -> NDArray[np.float64]:
        S = np.bincount(Y, minlength=self.K).astype(np.int64)
        return self._theta_from_counts(S)

    # Algorithm
    def _theta_from_counts(self, S: NDArray[np.int64]) -> NDArray[np.float64]:
        K = self.K
        ee = np.exp(self.eps)

        S_sorted = np.sort(S)
        k = 0
        while k < K - 1:
            if S_sorted[k] == 0:
                k += 1
                continue
            tail_sum = int(S_sorted[k:].sum())
            denom = int(S_sorted[k])
            
            cond = (K - k) + ee - 1 - (tail_sum / denom)
            if cond < 0:
                k += 1
            else:
                break

        tail_sum = S_sorted[k:].sum()

        Phi_vec = S.astype(np.float64) / float(tail_sum)
        theta = np.maximum(
            0.0,
            Phi_vec * (((K - k) / (ee - 1.0)) + 1.0) - (1.0 / (ee - 1.0)),
        )
        
        return theta
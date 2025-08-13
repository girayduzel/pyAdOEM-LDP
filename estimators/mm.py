# AdOEM_LDP/estimators/mm.py
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

class MomentMatching:
    """
    Simple moment matching.
    """
    def __init__(self, K: int, eps: float):
        self.K = K
        self.eps = eps
        self.C: NDArray[np.int64] = np.zeros(K, dtype=np.int64)
        self.t: int = 0

        ee = np.exp(eps)
        self.p = ee / (ee + K - 1.0)
        self.q = 1.0 / (ee + K - 1.0)

    # Online
    def update(self, y: int) -> None:
        self.C[y] += 1
        self.t += 1

    def current(self) -> NDArray[np.float64]:
        return self._theta_from_counts(self.C, self.t)

    # Offline
    def fit(self, Y: NDArray[np.int_]) -> NDArray[np.float64]:
        C = np.bincount(Y, minlength=self.K).astype(np.int64)
        t = int(C.sum())
        return self._theta_from_counts(C, t)

    # Algorithm
    def _theta_from_counts(self, C: NDArray[np.int64], t: int) -> NDArray[np.float64]:
        if t == 0:
            return np.full(self.K, 1.0 / self.K)

        F = (C.astype(np.float64) - t * self.q) / (self.p - self.q)
        F = np.maximum(F, 0.0)
        
        s = float(F.sum())
        return (F / s) if s > 0 else np.full(self.K, 1.0 / self.K)
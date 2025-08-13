# AdOEM_LDP/core/hashing.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class Hashing:
    """
    OLH Layer
    """

    K: int
    K_prime: int
    enabled: bool
    rng: np.random.Generator

    _H_vec: NDArray[np.int_] | None = None

    # Create an instance of the class for no hashing
    @classmethod
    def identity(cls, K: int, rng: np.random.Generator) -> "Hashing":
        return cls(K=K, K_prime=K, enabled=False, rng=rng)

    # Generate a new hashing vector at every time step
    def new_round(self) -> NDArray[np.int_]:
        if not self.enabled:
            self._H_vec = np.arange(self.K, dtype=np.int_)
        else:
            self._H_vec = self.rng.integers(0, self.K_prime, size=self.K, dtype=np.int_)
        return self._H_vec

    def current_H(self) -> NDArray[np.int_]:
        return self._H_vec

    def project_row_to_K(self, row_Kp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Project hashed likelihood vector to original domain: p_yx = p_yx_g(H)
        """
        H = self.current_H()
        return row_Kp[H].astype(np.float64, copy=False)

    def GH_curr(self, G_KpKp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute GH matrix column-wise
        """
        H = self.current_H()
        return G_KpKp[:, H]

    def compress_to_K_prime(self, theta_K: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compress theta to hashed domain
        """
        H = self.current_H()
        return np.bincount(H, weights=theta_K, minlength=self.K_prime).astype(np.float64)
    
    def HT_times_vec(self, vec_Kp: NDArray[np.float64]) -> NDArray[np.float64]:
        H = self.current_H()
        return vec_Kp[H].astype(np.float64, copy=False)

    def HT_times_mat(self, mat_KpK: NDArray[np.float64]) -> NDArray[np.float64]:
        H = self.current_H()
        return mat_KpK[H, :].astype(np.float64, copy=False)


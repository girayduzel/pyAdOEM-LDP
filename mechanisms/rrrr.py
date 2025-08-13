# AdOEM_LDP/mechanisms/rrrr.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.mechanism import Mechanism
from ..core.adapt import RRRRPolicy
from ..core.hashing import Hashing


class RRRRMechanism(Mechanism):
    def __init__(
        self,
        K: int,
        policy: RRRRPolicy,
        hashing: Hashing,
        *,
        theta_provider: callable | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(K)
        self.policy = policy
        self.hashing = hashing
        self.theta_provider = theta_provider or (lambda: np.full(K, 1.0 / K))
        self.rng = np.random.default_rng() if rng is None else rng

        self.t: int = 0
        self._last_row_hashed: NDArray[np.float64] | None = None
        self._last_G_hashed: NDArray[np.float64] | None = None 

    def privatise(self, x: int) -> int:
        self.t += 1
        H = self.hashing.new_round()
        theta_hat = self.theta_provider()
        G_t = self.policy.next_kernel(theta_hat, self.t, self.hashing)
        j = int(H[x])
        y = int(self.rng.choice(self.hashing.K_prime, p=G_t[:, j]))
        self._last_row_hashed = G_t[y]
        self._last_G_hashed = G_t
        return y

    def likelihood_row(self, y: int) -> NDArray[np.float64]:
        return self.hashing.project_row_to_K(self._last_row_hashed).copy()

    def last_kernel_hashed(self) -> NDArray[np.float64]:
        return self._last_G_hashed

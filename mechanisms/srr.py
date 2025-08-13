# AdOEM_LDP/mechanisms/srr.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.mechanism import Mechanism
from ..core.channels import make_G_srr
from ..core.hashing import Hashing


class SRRMechanism(Mechanism):
    def __init__(
        self,
        K: int,
        eps: float,
        hashing: Hashing,
        *,
        store_y: bool = False,
        track_mm: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(K)
        self.eps = float(eps)
        self.hashing = hashing
        self.rng = np.random.default_rng() if rng is None else rng

        self.G_srr: NDArray[np.float64] = make_G_srr(self.hashing.K_prime, self.eps)
        self._last_row_hashed: NDArray[np.float64] | None = None

        self._store_y = bool(store_y)
        self._track_mm = bool(track_mm)
        self._y_history: list[int] = [] if self._store_y else []
        self._C_mm: NDArray[np.int64] = np.zeros(K, dtype=np.int64) if self._track_mm else np.zeros(K, dtype=np.int64)

    def privatise(self, x: int) -> int:
        H = self.hashing.new_round()
        j = int(H[x])
        y = int(self.rng.choice(self.hashing.K_prime, p=self.G_srr[:, j]))
        self._last_row_hashed = self.G_srr[y]

        if self._store_y:
            self._y_history.append(y)
        if self._track_mm:
            self._C_mm[H == y] += 1

        return y

    def likelihood_row(self, y: int) -> NDArray[np.float64]:
        return self.hashing.project_row_to_K(self._last_row_hashed).copy()

    # For QP Accumulators
    def last_kernel_hashed(self) -> NDArray[np.float64]:
        return self.G_srr

    # For MLE and MM
    def y_history_array(self) -> NDArray[np.int_]:
        return np.asarray(self._y_history, dtype=int)

    def mm_counts(self) -> NDArray[np.int64]:
        return self._C_mm.copy()

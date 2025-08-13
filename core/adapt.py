# AdOEM_LDP/core/adapt.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .utils import calculate_utility, Utility, dirichlet_perturb
from .channels import make_G_rrrr, make_G_srr
from .hashing import Hashing


class RRRRPolicy:
    """
    Adaptation Policy for RRRR
    """
    def __init__(
        self,
        K_prime: int,
        eps: float,
        *,
        eps1_coeff_vec: list[float],
        util_type: int | Utility = Utility.PROB_Y_EQUALS_X,
        dirichlet_scale: float = 0.01,
        total_T: int | None = None,
        warmup_fraction: float = 0.10,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.Kp = int(K_prime)
        self.eps = float(eps)
        self.eps1_coeff_vec = [float(c) for c in eps1_coeff_vec]
        self.util_type = Utility(util_type)
        self.dirichlet_scale = float(dirichlet_scale)
        self.total_T = int(total_T) if total_T is not None else None
        self.warmup_fraction = float(warmup_fraction)
        self.rng = np.random.default_rng() if rng is None else rng

        self.G_warmup: NDArray[np.float64] = make_G_srr(self.Kp, self.eps)
        self._GG = self._precompute_all_candidates()

    def next_kernel(
        self,
        theta_hat_K: NDArray[np.float64],
        t: int,
        hashing: Hashing,
    ) -> NDArray[np.float64]:
        if self._in_warmup(t):
            return self.G_warmup

        theta_hat_K = self._normalize(theta_hat_K)
        theta_tilde_K = dirichlet_perturb(theta_hat_K, self.dirichlet_scale, t, self.rng)
        theta_tilde_g = hashing.compress_to_K_prime(theta_tilde_K)

        # Permutation
        ord_ind = np.argsort(theta_tilde_g)[::-1]
        ord_inv = np.empty_like(ord_ind)
        ord_inv[ord_ind] = np.arange(self.Kp, dtype=ord_ind.dtype)
        theta_ord = theta_tilde_g[ord_ind]

        best_key = None
        best_L = -np.inf
        for key, Gk in self._GG.items():
            L = calculate_utility(theta_ord, Gk, self.util_type)
            if L > best_L:
                best_L = L
                best_key = key

        G_best = self._GG[best_key]
        G_current = G_best[np.ix_(ord_inv, ord_inv)]
        return G_current


    def _in_warmup(self, t: int) -> bool:
        if self.total_T is None:
            return False
        return t <= int(self.warmup_fraction * self.total_T)

    def _normalize(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(v, dtype=np.float64).ravel()
        s = float(v.sum())
        return (v / s) if s > 0 else np.full(v.size, 1.0 / v.size, dtype=np.float64)

    def _precompute_all_candidates(self) -> dict[tuple[int, int], NDArray[np.float64]]:
        Kp = self.Kp
        eps = self.eps
        GG: dict[tuple[int, int], NDArray[np.float64]] = {}

        for i, c in enumerate(self.eps1_coeff_vec):
            eps1 = c * eps
            Sc_min = int(np.ceil(np.exp(eps - eps1)))

            eps2_vec = np.full(Kp + 1, eps, dtype=np.float64) 
            for k in range(Sc_min, Kp + 1):
                denom = np.exp(eps1 - eps) * k - 1.0
                if denom > 0.0:
                    eps2_ub = (k - 1.0) / denom
                    eps2_vec[k] = min(eps, eps2_ub)
                else:
                    eps2_vec[k] = eps

            for k in range(1, Kp + 1):
                GG[(i, k)] = make_G_rrrr(Kp, k, eps1, float(eps2_vec[k]))

        return GG
# AdOEM_LDP/data/stream.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# Sample True Theta
def sample_theta_dirichlet(
    K: int,
    rho_coeff: float | None = None,
    *,
    rho_vec: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Sample θ ~ Dirichlet(ρ).
    """
    if rng is None:
        rng = np.random.default_rng()

    if rho_vec is not None:
        rho = np.asarray(rho_vec, dtype=np.float64)
        assert rho.shape == (K,)
    else:
        assert rho_coeff is not None
        rho = np.full(K, float(rho_coeff), dtype=np.float64)

    gam = rng.gamma(shape=rho, scale=1.0)
    s = gam.sum()
    return (gam / s).astype(np.float64)


# Sample Online Data from Cat(θ)
@dataclass
class CategoricalStream(Iterable[int]):
    """
    IID categorical data generator from fixed θ.
    """
    theta_true: NDArray[np.float64]
    T: int
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.theta_true = np.asarray(self.theta_true, dtype=np.float64)
        self.K: int = self.theta_true.size
        s = float(self.theta_true.sum())
        self.theta_true /= s

    # Iterator
    def __iter__(self) -> Iterator[int]:
        remaining = self.T
        batch = 128
        while remaining > 0:
            n = min(batch, remaining)
            xs = self.rng.choice(self.K, size=n, p=self.theta_true)
            for x in xs:
                yield int(x)
            remaining -= n

    # Batch Sampling
    def sample_all(self) -> NDArray[np.int_]:
        return self.rng.choice(self.K, size=self.T, p=self.theta_true)

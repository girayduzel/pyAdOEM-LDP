#adAdOEM_LDP/core/utils.py
"""Utility functions"""

from __future__ import annotations

import numpy as np
from enum import IntEnum, unique
from numpy.typing import NDArray
from typing import Final

# Enumerator
@unique
class Utility(IntEnum):
    """Identifiers for the utility functions."""
    FISHER_INFO        = 1
    ENTROPY            = 2
    TV_INPUT_OUTPUT    = 3
    TV_POSTERIOR       = 4
    EXP_MSE            = 5
    PROB_Y_EQUALS_X    = 6


# Numerical Stability Helpers
_LOG_ZERO: Final = -1.0e30

def log_sum_exp(vec: NDArray[np.float64]) -> float:
    vmax = float(vec.max(initial=-np.inf))
    if vmax == -np.inf:
        return _LOG_ZERO
    return vmax + float(np.log(np.exp(vec - vmax).sum()))


# Fisher Information Matrix
def calculate_fim(
    G: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculates the Fisher information matrix.
    """
    K = theta.size
    p_y = G @ theta                      
    H = G[:, :K-1] - G[:, [K-1]]   
    fim = (H.T / p_y).dot(H)            
    return fim


# Utility Functions
def calculate_utility(
    theta: NDArray[np.float64],
    G: NDArray[np.float64],
    util_type: Utility | int,
) -> float:
    """
    Calculates the utility value based on function type.
    """
    theta_sorted = np.sort(theta)[::-1]       
    util_type    = Utility(util_type)
    p = G @ theta_sorted                  

    if util_type is Utility.FISHER_INFO:
        fim = calculate_fim(G, theta_sorted)
        return -np.trace(np.linalg.inv(fim))

    if util_type is Utility.ENTROPY:
        return float(np.sum(p * np.log(p, where=p > 0)))

    if util_type is Utility.TV_INPUT_OUTPUT:
        return -float(np.abs(theta_sorted - p).sum())

    if util_type is Utility.TV_POSTERIOR:
        posterior = (G * theta_sorted) / p
        tv = np.abs(posterior - theta_sorted).sum(axis=1) @ p
        return -float(tv)

    if util_type is Utility.EXP_MSE:
        numer = (G**2) @ (theta_sorted**2)
        return -1.0 + float(np.sum(numer / p, where=p > 0))

    if util_type is Utility.PROB_Y_EQUALS_X:
        return float(np.sum(theta_sorted * np.diag(G)))
    
    
#  Exploration
def dirichlet_perturb(
    theta_hat: NDArray[np.float64],
    scale: float,
    t: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    shape = 1.0 + float(scale) * float(t) * theta_hat
    gam = rng.gamma(shape=shape, scale=1.0)
    return gam / float(gam.sum())

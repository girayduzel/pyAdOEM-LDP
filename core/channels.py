#AdOEM_LDP/core/channels.py
"""
Mechanism-specific channel builders.
"""
from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray


# Transition Matrices
def make_G_srr(K: int, eps: float) -> NDArray[np.float64]:
    """
    Generates the transition matrix for SRR.
    """
    exp_eps = math.exp(eps)
    p = exp_eps / (exp_eps + K - 1)
    q = 1.0 / (exp_eps + K - 1)
    return (p - q) * np.eye(K, dtype=np.float64) + q


def make_G_rrrr(K: int, k0: int, eps1: float, eps2: float) -> NDArray[np.float64]:
    """
    Generates the transition matrix for RRRR.
    """
    # Transition Probabilities
    exp_eps1 = math.exp(eps1)
    exp_eps2 = math.exp(eps2)

    p11 = exp_eps1 / (exp_eps1 + min(k0, K - 1))
    p12 = 1.0 / (exp_eps1 + min(k0, K - 1))
    p22 = exp_eps2 / (exp_eps2 + K - k0 - 1)
    p21 = 1.0 / (exp_eps2 + K - k0 - 1)

    G = np.zeros((K, K), dtype=np.float64)

    # Upper-left block
    G[:k0, :k0] = (p11 - p12) * np.eye(k0) + p12

    if K > k0:
        # Lower-right block
        G[k0:, k0:] = p11 * ((p22 - p21) * np.eye(K - k0) + p21)

        # Off-diagonal blocks
        G[k0:, :k0] = p12 / (K - k0)
        G[:k0, k0:] = p12

    return G

# TODO bound k0 by K-1


def make_p_yx_vec(
    y: int,
    K: int,
    S: set[int],
    k0: int,
    eps1: float,
    eps2: float,
) -> NDArray[np.float64]:
    """
    Generates one row of the transition matrix G.
    """
    exp_eps1 = math.exp(eps1)
    exp_eps2 = math.exp(eps2)

    p11 = exp_eps1 / (exp_eps1 + min(k0, K - 1))
    p12 = 1.0 / (exp_eps1 + min(k0, K - 1))
    p21 = 1.0 / (exp_eps2 + K - k0 - 1)
    p22 = exp_eps2 / (exp_eps2 + K - k0 - 1)

    g = np.empty(K, dtype=np.float64)

    if y in S:                                      
        g.fill(p12)
        g[y] = p11
    else:                                          
        g.fill(p11 * p21)                          
        g[list(S)] = p12 / (K - k0)                
        g[y] = p11 * p22                          

    return g

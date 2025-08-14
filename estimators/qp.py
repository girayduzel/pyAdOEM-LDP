# AdOEM_LDP/estimators/qp.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, Bounds, LinearConstraint

from ..core.hashing import Hashing


# Accumulator

@dataclass
class QPAccumulators:
    """Count vectors for QP1 and QP2"""
    K: int

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        K = int(self.K)
        self.Sum_GHTGH: NDArray[np.float64] = np.zeros((K, K), dtype=np.float64)
        self.sum_GHTc:  NDArray[np.float64] = np.zeros(K, dtype=np.float64)
        self.Sum_HTGH:  NDArray[np.float64] = np.zeros((K, K), dtype=np.float64)
        self.sum_HTc:   NDArray[np.float64] = np.zeros(K, dtype=np.float64)

    def update(self, G_hashed: NDArray[np.float64], hashing: Hashing, y_hashed: int) -> None:
        GH_curr = hashing.GH_curr(G_hashed)

        Kp = G_hashed.shape[0]
        ct = np.zeros(Kp, dtype=np.float64)
        ct[y_hashed] = 1.0

        self.Sum_GHTGH += GH_curr.T @ GH_curr         
        self.sum_GHTc  += GH_curr.T @ ct              

        if hashing.enabled:
            self.Sum_HTGH += hashing.HT_times_mat(GH_curr)  
            self.sum_HTc  += hashing.HT_times_vec(ct)      
        else:
            self.Sum_HTGH += GH_curr
            self.sum_HTc  += ct


# Solver
@dataclass
class QPSolver:
    """
    Solve the quadratic programming problem with constraints to yield a consistent estimator  
    """
    stabilizer: float = 1e-10
    ftol: float = 1e-5
    maxiter: int = 100

    def solve_qp1(self, acc: QPAccumulators, warm_start: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        M = acc.Sum_GHTGH.copy()
        v = -acc.sum_GHTc.copy()
        return self._solve(M, v, warm_start)

    def solve_qp2(self, acc: QPAccumulators, warm_start: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        A = acc.Sum_HTGH
        b = acc.sum_HTc
        M = A.T @ A
        v = -(A.T @ b)
        return self._solve(M, v, warm_start)

    # Core
    def _solve(self, M: NDArray[np.float64], v: NDArray[np.float64], x0: NDArray[np.float64] | None) -> NDArray[np.float64]:
        K = v.size
        M = (M + self.stabilizer * np.eye(K)).astype(np.float64, copy=False)
        v = v.astype(np.float64, copy=False).ravel()
        x0 = np.full(K, 1.0 / K) if x0 is None else np.asarray(x0, dtype=np.float64)

        def obj(x: NDArray[np.float64]) -> float:
            return 0.5 * float(x @ (M @ x)) + float(v @ x)

        def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return (M @ x) + v

        bounds = Bounds(lb=np.zeros(K), ub=np.full(K, np.inf))
        Aeq = np.ones((1, K), dtype=np.float64)
        beq = np.array([1.0], dtype=np.float64)
        lincon = LinearConstraint(Aeq, beq, beq)

        res = minimize(
            fun=obj,
            x0=x0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=[lincon],
            options={"ftol": self.ftol, "maxiter": self.maxiter, "disp": False},
        )
        return np.asarray(res.x, dtype=np.float64)

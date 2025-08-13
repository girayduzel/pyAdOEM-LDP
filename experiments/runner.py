# AdOEM_LDP/experiments/runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from ..data.stream import sample_theta_dirichlet, CategoricalStream
from ..core.hashing import Hashing
from ..core.adapt import RRRRPolicy
from ..mechanisms.srr import SRRMechanism
from ..mechanisms.rrrr import RRRRMechanism

from ..estimators.online_em import OnlineEM
from ..estimators.mle_rr import MLERR
from ..estimators.mm import MomentMatching
from ..estimators.offline_em import OfflineEM
from ..estimators.qp import QPAccumulators, QPSolver


MechName = Literal["SRR", "RRRR"]
OnlineName = Literal["online_em", "mle_rr", "mm", "qp1", "qp2"]
OfflineName = Literal["offline_em", "mle_rr", "mm", "qp1", "qp2"]


@dataclass
class RunnerConfig:
    # Experiment Configurations
    K: int
    T: int
    eps: float

    # Mechanism
    mech: MechName # SRR or RRRR

    # Hashing Toggle
    hashing_enabled: bool = False
    K_prime: Optional[int] = None # if None and hashing_enabled -> ceil(exp(eps))+1

    # Online Estimator
    online: OnlineName = "online_em"

    # Offline Estimator
    offline: OfflineName = "offline_em"

    # RRRR parameters
    eps1_coeff_vec: list[float] = (0.9,)
    util_type: int = 6 # Defaults to probability of honest response                 
    dirichlet_scale: float = 0.01 # For perturbation
    warmup_fraction: float = 0.10

    # Scheduling
    P_int_EM: int = 0 # Offline EM refinement period
    M_int_EM: int = 0 # Offline EM refinement iterations
    P_quad_prog: int = 0 # Online QP period

    # Final Offline EM iterations
    M_EM_final: int = 1000

    # Online EM step size
    alpha_EM: float = 0.65

    # Data sampling parameters
    rho_coeff: float = 0.05
    seed: int = 42


@dataclass
class RunnerResult:
    theta_true: NDArray[np.float64]
    theta_online_final: NDArray[np.float64]
    theta_offline_final: NDArray[np.float64]
    tv_online: float
    tv_offline: float


class SingleRunner:
    def __init__(self, cfg: RunnerConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Hashing
        if cfg.hashing_enabled:
            Kp = cfg.K_prime if cfg.K_prime is not None else int(np.ceil(np.exp(cfg.eps)) + 1)
            self.hashing = Hashing(K=cfg.K, K_prime=Kp, enabled=True, rng=self.rng)
        else:
            self.hashing = Hashing.identity(cfg.K, rng=self.rng)

        # Mechanism configuration
        self.mech = self._build_mechanism()

        # Online estimator configuration
        self.online_est, self.qp_accum, self.qp_solver = self._build_online_estimator()

        # Theta for adaptation
        if self.cfg.mech == "RRRR":
            if self.online_est is None:
                # Online QP
                self._theta_online_current = np.full(cfg.K, 1.0 / cfg.K)
                self.mech.theta_provider = lambda: self._theta_online_current
            else:
                self.mech.theta_provider = self.online_est.current

        # Offline EM configuration
        self.offline_em_model = OfflineEM(cfg.K, cfg.M_EM_final) if (cfg.offline == "offline_em" or cfg.P_int_EM > 0) else None

        # Log-likelihood buffer for offline EM
        self.log_P_yx = np.zeros((cfg.K, cfg.T), dtype=np.float64)


    def _build_mechanism(self):
        cfg = self.cfg
        if cfg.mech == "SRR":
            return SRRMechanism(
                cfg.K, cfg.eps, self.hashing,
                rng=self.rng,
                store_y=(cfg.offline == "mle_rr"),
                track_mm=(cfg.offline == "mm"),
            )
        else:
            policy = RRRRPolicy(
                K_prime=self.hashing.K_prime,
                eps=cfg.eps,
                eps1_coeff_vec=list(cfg.eps1_coeff_vec),
                util_type=cfg.util_type,
                dirichlet_scale=cfg.dirichlet_scale,
                total_T=cfg.T,
                warmup_fraction=cfg.warmup_fraction,
                rng=self.rng,
            )
            return RRRRMechanism(cfg.K, policy, self.hashing, rng=self.rng)

    def _build_online_estimator(self):
        cfg = self.cfg
        if cfg.online == "online_em":
            return OnlineEM(cfg.K, alpha=cfg.alpha_EM), None, None
        if cfg.online == "mle_rr":
            return MLERR(cfg.K, cfg.eps), None, None
        if cfg.online == "mm":
            return MomentMatching(cfg.K, cfg.eps), None, None
        # QP1 / QP2 as online
        return None, QPAccumulators(cfg.K), QPSolver()


    def run(self) -> RunnerResult:
        cfg = self.cfg

        # 1) Sample theta and generate data stream
        theta_true = sample_theta_dirichlet(cfg.K, rho_coeff=cfg.rho_coeff, rng=self.rng)
        stream = CategoricalStream(theta_true, cfg.T, self.rng)

        # Online Loop
        t = 0
        for x in stream:
            t += 1

            # Privatize and return likelihood row
            y = self.mech.privatise(x)
            g_row = self.mech.likelihood_row(y)     
            # log-safe
            lr = np.full_like(g_row, -1e30)
            mask = g_row > 0
            lr[mask] = np.log(g_row[mask])
            self.log_P_yx[:, t - 1] = lr

            # Online update
            if isinstance(self.online_est, OnlineEM):
                self.online_est.update(y, g_row)
            elif isinstance(self.online_est, (MLERR, MomentMatching)):
                self.online_est.update(y)

            # Offline EM Refinement
            if (cfg.P_int_EM > 0) and (t % cfg.P_int_EM == 0) and (self.offline_em_model is not None):
                theta0 = self._get_online_theta()
                theta_ref = self.offline_em_model.fit_from_logs(self.log_P_yx[:, :t], theta0=theta0)
                self._set_online_theta(theta_ref)

            # Update Count Vectors for QP
            if self.qp_accum is not None:
                self.qp_accum.update(self.mech.last_kernel_hashed(), self.hashing, y)

            # Online QP Solve
            if (self.qp_accum is not None) and (self.online_est is None) and (cfg.P_quad_prog > 0) and (t % cfg.P_quad_prog == 0):
                if cfg.online == "qp1":
                    self._theta_online_current = self.qp_solver.solve_qp1(self.qp_accum)
                else: #QP2
                    self._theta_online_current = self.qp_solver.solve_qp2(self.qp_accum)

        # 3) Final online theta
        theta_online = self._get_online_theta()

        # 4) Final offline theta
        theta_offline = self._run_offline()

        # 5) Metrics
        tv_online = 0.5 * float(np.abs(theta_online - theta_true).sum())
        tv_offline = 0.5 * float(np.abs(theta_offline - theta_true).sum())

        return RunnerResult(
            theta_true=theta_true,
            theta_online_final=theta_online,
            theta_offline_final=theta_offline,
            tv_online=tv_online,
            tv_offline=tv_offline,
        )


    def _get_online_theta(self) -> NDArray[np.float64]:
        if isinstance(self.online_est, (OnlineEM, MLERR, MomentMatching)):
            return self.online_est.current()
        # QP
        return getattr(self, "_theta_online_current", np.full(self.cfg.K, 1.0 / self.cfg.K))

    def _set_online_theta(self, theta: NDArray[np.float64]) -> None:
        if isinstance(self.online_est, OnlineEM):
            self.online_est.theta = theta.copy()
        elif isinstance(self.online_est, (MLERR, MomentMatching)):
            # Feed via provider
            if self.cfg.mech == "RRRR":
                self.mech.theta_provider = lambda: theta
        else:
            self._theta_online_current = theta.copy()

    def _run_offline(self) -> NDArray[np.float64]:
        cfg = self.cfg

        # Offline EM
        if cfg.offline == "offline_em":
            theta0 = self._get_online_theta()
            return self.offline_em_model.fit_from_logs(self.log_P_yx, theta0=theta0)

        # Offline MLE
        if cfg.offline == "mle_rr":
            ys = self.mech.y_history_array()         
            return MLERR(cfg.K, cfg.eps).fit(ys)

        # Offline MM 
        if cfg.offline == "mm": # TODO use the MM class instead of this
            C = self.mech.mm_counts()
            g = self.hashing.K_prime
            ee = float(np.exp(cfg.eps))
            p = ee / (ee + g - 1.0)
            q = 1.0 / g
            F = (C.astype(np.float64) - float(cfg.T) * q) / (p - q)
            F = np.maximum(F, 0.0)
            s = float(F.sum())
            return (F / s) if s > 0 else np.full(cfg.K, 1.0 / cfg.K)

        # Offline QP
        if cfg.offline in ("qp1", "qp2"):
            warm = self._get_online_theta()
            if cfg.offline == "qp1":
                return self.qp_solver.solve_qp1(self.qp_accum, warm)
            else:
                return self.qp_solver.solve_qp2(self.qp_accum, warm)
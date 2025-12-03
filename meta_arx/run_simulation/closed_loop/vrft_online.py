# run_simulation/closed_loop/vrft_online.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.linalg import lstsq
from scipy import signal
from scipy.signal import lfilter
import sympy as sm

from .closed_loop_sim import PIDParams


# ---------------------------------------------------------------------------
# 1) Reference model and weighting (same math as your VRFT script)
# ---------------------------------------------------------------------------

def M_cont_to_disc(tau: float, t_settle: float, q: int, Ts: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build discrete-time reference model M(z) corresponding to

        M(s) = e^{-tau s} / (1 + 0.2 * t_settle * s)^q

    via bilinear transform + integer sample delay.
    """
    den_coeff = np.polynomial.polynomial.polypow([1.0, 0.2 * t_settle], q)
    den_coeff = list(reversed(den_coeff))  # SciPy TransferFunction wants descending
    num_coeff = [1.0]

    M_CT = signal.TransferFunction(num_coeff, den_coeff)
    M_DT = M_CT.to_discrete(Ts, method="bilinear")

    delay_samples = int(round(tau / Ts))
    if delay_samples > 0:
        M_DT_num = np.concatenate([np.zeros(delay_samples), np.atleast_1d(M_DT.num)])
    else:
        M_DT_num = np.atleast_1d(M_DT.num)

    M_DT_den = np.atleast_1d(M_DT.den)

    return M_DT_num.astype(float), M_DT_den.astype(float)


def W_cont_to_disc(omega: float, Ts: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build discrete-time frequency weighting

        W(s) = omega / (s + omega)
    """
    W_CT = signal.TransferFunction([omega], [1.0, omega])
    W_DT = W_CT.to_discrete(Ts, method="bilinear")

    return np.atleast_1d(W_DT.num).astype(float), np.atleast_1d(W_DT.den).astype(float)


# ---------------------------------------------------------------------------
# 2) Symbolic construction of VRFT filters F(z) and F_aux(z)
# ---------------------------------------------------------------------------

def _construct_poly(coeffs: List[float], var) -> sm.Expr:
    # coeffs as [a0, a1, ..., an] in descending powers
    deg = len(coeffs) - 1
    return sum(c * var ** (deg - i) for i, c in enumerate(coeffs))


def _construct_rational(num: List[float], den: List[float], var) -> sm.Expr:
    return _construct_poly(num, var) / _construct_poly(den, var)


def _get_coeffs(expr: sm.Expr, var) -> Tuple[List[float], List[float]]:
    """
    Return numerator/denominator coeffs for lfilter, in ascending order.
    """
    num, den = sm.fraction(sm.simplify(expr))
    num_coeffs = sm.Poly(num, var).all_coeffs()
    den_coeffs = sm.Poly(den, var).all_coeffs()
    # lfilter expects ascending powers
    return list(reversed(num_coeffs)), list(reversed(den_coeffs))


def get_filter_coeff(
    M_num: np.ndarray,
    M_den: np.ndarray,
    W_num: np.ndarray,
    W_den: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build VRFT filters F(z), F_aux(z) given M(z) and W(z):

        y_v = F(z) y
        e_v = F_aux(z) y - y_v
        u_l = F(z) u
    """
    x = sm.symbols("x")

    M = _construct_rational(M_num.tolist(), M_den.tolist(), x)
    W = _construct_rational(W_num.tolist(), W_den.tolist(), x)

    # Main VRFT filter
    F = M * (1 - M) * W / 2
    F_num, F_den = _get_coeffs(F, x)

    # Auxiliary filter (avoid explicit r_v)
    F_aux = (1 - M) * W / 2
    aux_num, aux_den = _get_coeffs(F_aux, x)

    # Convert SymPy -> float np arrays
    F_num = np.array([float(c) for c in F_num], dtype=float)
    F_den = np.array([float(c) for c in F_den], dtype=float)
    aux_num = np.array([float(c) for c in aux_num], dtype=float)
    aux_den = np.array([float(c) for c in aux_den], dtype=float)

    return F_num, F_den, aux_num, aux_den


# ---------------------------------------------------------------------------
# 3) Online VRFT PID tuner (sliding-window, called every time step)
# ---------------------------------------------------------------------------

@dataclass
class OnlineVRFTPID:
    """
    Online VRFT tuner with sliding window.

    - Same PID structure as your original VRFT script:
        u[k] ≈ θ1 * e_v[k]
             + θ2 * (-(y_v[k] - y_v[k-1]) / Ts)
             + θ3 * (sum_{i<=k} e_v[i]) * Ts

    - At every time step k, it uses the last T_window seconds of (u, y)
      to recompute θ by least squares, optionally with ridge regularization.

    - Lives entirely outside the simulator; it just sees (y_k, u_k).
    """

    Ts: float
    T_window: float
    tau: float
    t_settle: float
    q_order: int
    omega: float
    lam: float = 0.0        # ridge regularization for LS
    alpha: float = 0.2      # smoothing factor for parameter updates
    initial_pid: PIDParams = field(
        default_factory=lambda: PIDParams(Kp=0.0, Ki=0.0, Kd=0.0)
    )

    def __post_init__(self) -> None:
        # Number of samples in the sliding window
        self.Nw: int = max(3, int(round(self.T_window / self.Ts)))

        # Precompute reference model + weighting in discrete time
        M_num, M_den = M_cont_to_disc(self.tau, self.t_settle, self.q_order, self.Ts)
        W_num, W_den = W_cont_to_disc(self.omega, self.Ts)

        # Precompute VRFT filters
        self.F_num, self.F_den, self.aux_num, self.aux_den = get_filter_coeff(
            M_num, M_den, W_num, W_den
        )

        # History buffers (last Nw samples)
        self.y_hist: List[float] = []
        self.u_hist: List[float] = []

        # Internal parameter vector θ = [Kp, Kd, Ki]
        self.theta = np.array(
            [self.initial_pid.Kp, self.initial_pid.Kd, self.initial_pid.Ki],
            dtype=float,
        )

    # ---------- Data handling ----------

    def update_buffers(self, y_k: float, u_k: float) -> None:
        """
        Store newest plant sample (y_k, u_k) and keep only last Nw samples.
        """
        self.y_hist.append(float(y_k))
        self.u_hist.append(float(u_k))

        if len(self.y_hist) > self.Nw:
            self.y_hist.pop(0)
            self.u_hist.pop(0)

    # ---------- Core VRFT update ----------

    def maybe_retrain(self) -> PIDParams:
        """
        If we have enough samples in the window, recompute θ via VRFT (LS on
        filtered data) and return the corresponding PIDParams.

        Can be called at every time step. If not enough data, returns the
        current PIDParams (no change).
        """
        # Not enough data yet – just return current parameters
        if len(self.y_hist) < self.Nw:
            return PIDParams(Kp=self.theta[0], Kd=self.theta[1], Ki=self.theta[2])

        # ---- Build window arrays ----
        y = np.asarray(self.y_hist, dtype=float)
        u = np.asarray(self.u_hist, dtype=float)

        # Basic finite check
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
            return PIDParams(Kp=self.theta[0], Kd=self.theta[1], Ki=self.theta[2])

        # ---- VRFT filtering on the current window ----
        y_v = lfilter(self.F_num, self.F_den, y)
        aux = lfilter(self.aux_num, self.aux_den, y)
        e_v = aux - y_v
        u_f = lfilter(self.F_num, self.F_den, u)

        # Incremental controller model: Δu_f[k] ≈ θᵀ φ[k]
        du_f = np.diff(u_f)    # length Nw-1

        ev = e_v[1:]           # length Nw-1
        yv = y_v[1:]           # length Nw-1
        dy = (yv - y_v[:-1]) / self.Ts
        eint = np.cumsum(ev) * self.Ts

        Phi = np.column_stack(
            [
                ev,        # proportional on filtered error
                -dy,       # derivative on output (minus sign)
                eint,      # integral on error
            ]
        )
        u_target = du_f

        # Second finite check
        if not np.all(np.isfinite(Phi)) or not np.all(np.isfinite(u_target)):
            return PIDParams(Kp=self.theta[0], Kd=self.theta[1], Ki=self.theta[2])

        # ---- Solve LS or ridge-regularized LS ----
        if self.lam > 0.0:
            G = Phi.T @ Phi + self.lam * np.eye(Phi.shape[1])
            b = Phi.T @ u_target
            theta_new = np.linalg.solve(G, b)
        else:
            theta_new, *_ = lstsq(Phi, u_target, rcond=None)

        # Smooth parameter adaptation
        self.theta = (1.0 - self.alpha) * self.theta + self.alpha * theta_new

        return PIDParams(Kp=self.theta[0], Kd=self.theta[1], Ki=self.theta[2])


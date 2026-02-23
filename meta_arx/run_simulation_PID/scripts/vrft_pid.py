# run_simulation_PID_VRFT/vrft/vrft_pid.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import signal


@dataclass
class PIDParams:
    Kp: float
    Ki: float
    Kd: float


def M_cont_to_disc(tau: float, t: float, q: int, Ts: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous ref model:
        M(s) = e^{-tau s} / (1 + 0.2 t s)^q

    Discretized with bilinear transform. Time delay tau is approximated
    as z^{-d} in discrete time (numerator zero-padding).
    """
    # Denominator (1 + 0.2 t s)^q as polynomial in s (ascending powers)
    den_coeff = np.polynomial.polynomial.polypow([1.0, 0.2 * t], q)
    # Convert to descending powers for scipy.TransferFunction
    den_coeff = list(reversed(den_coeff))
    num_coeff = [1.0]

    M_CT = signal.TransferFunction(num_coeff, den_coeff)
    M_DT = M_CT.to_discrete(Ts, method="bilinear")

    M_num = np.array(M_DT.num, dtype=float).ravel()
    M_den = np.array(M_DT.den, dtype=float).ravel()

    delay_samples = int(round(tau / Ts))
    if delay_samples > 0:
        # Implement delay as z^-d: pad numerator with zeros
        M_num = np.concatenate([np.zeros(delay_samples), M_num])

    return M_num, M_den


def W_cont_to_disc(omega: float, Ts: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous weighting:
        W(s) = omega / (s + omega)

    Discretized with bilinear transform.
    """
    W_CT = signal.TransferFunction([omega], [1.0, omega])
    W_DT = W_CT.to_discrete(Ts, method="bilinear")
    W_num = np.array(W_DT.num, dtype=float).ravel()
    W_den = np.array(W_DT.den, dtype=float).ravel()
    return W_num, W_den


def _poly_add(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Add two polynomials in descending powers."""
    p = np.array(p, dtype=float).ravel()
    q = np.array(q, dtype=float).ravel()
    if len(p) < len(q):
        p = np.pad(p, (len(q) - len(p), 0))
    elif len(q) < len(p):
        q = np.pad(q, (len(p) - len(q), 0))
    return p + q


def _poly_sub(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Subtract q from p: p - q, descending powers."""
    return _poly_add(p, -np.array(q, dtype=float))


def _poly_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Multiply two polynomials in descending powers."""
    p_asc = np.array(p, dtype=float)[::-1]
    q_asc = np.array(q, dtype=float)[::-1]
    r_asc = np.polynomial.polynomial.polymul(p_asc, q_asc)
    return r_asc[::-1]


def build_filtered_vrft_filters(
    M_num: np.ndarray,
    M_den: np.ndarray,
    W_num: np.ndarray,
    W_den: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build discrete-time filters used in filtered VRFT:

        F(z)     = 1/2 * M(z) (1 - M(z)) W(z)
        F_aux(z) = 1/2 * (1 - M(z)) W(z)

    All polynomials are in descending powers.
    """
    M_num = np.array(M_num, dtype=float).ravel()
    M_den = np.array(M_den, dtype=float).ravel()
    W_num = np.array(W_num, dtype=float).ravel()
    W_den = np.array(W_den, dtype=float).ravel()

    # 1 - M(z) = (M_den(z) - M_num(z)) / M_den(z)
    num_1_minus_M = _poly_sub(M_den, M_num)
    den_1_minus_M = M_den.copy()

    # M(z)(1-M(z)) = M_num * (den - num) / (den^2)
    num_M_1_minus_M = _poly_mul(M_num, num_1_minus_M)
    den_M_1_minus_M = _poly_mul(M_den, M_den)

    # F(z) = 1/2 * M(1-M) W
    F_num = _poly_mul(num_M_1_minus_M, W_num)
    F_den = _poly_mul(den_M_1_minus_M, W_den)
    F_num = 0.5 * F_num

    # F_aux(z) = 1/2 * (1 - M) W
    aux_num = _poly_mul(num_1_minus_M, W_num)
    aux_den = _poly_mul(den_1_minus_M, W_den)
    aux_num = 0.5 * aux_num

    # Normalize so leading denominator coeff = 1
    F_den_lead = F_den[0]
    aux_den_lead = aux_den[0]
    F_num /= F_den_lead
    F_den /= F_den_lead
    aux_num /= aux_den_lead
    aux_den /= aux_den_lead

    return F_num, F_den, aux_num, aux_den


def build_vrft_regressors_filtered(
    y: np.ndarray,
    u: np.ndarray,
    Ts: float,
    tau: float,
    t: float,
    q: int,
    omega: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtered VRFT:

      - Reference model M(s) = e^{-tau s} / (1 + 0.2 t s)^q
      - Weighting W(s) = omega / (s + omega)
      - Discretize M, W
      - Build F(z) and F_aux(z)
      - e_v_filtered = lfilter(F_aux, y) - lfilter(F, y)
      - u_filtered   = lfilter(F, u)
      - PID regressors on filtered e_v
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)

    if len(y) != len(u):
        raise ValueError("y and u must have the same length")
    if len(y) < 10:
        raise ValueError("Need at least ~10 samples for filtered VRFT")

    M_num, M_den = M_cont_to_disc(tau=tau, t=t, q=q, Ts=Ts)
    W_num, W_den = W_cont_to_disc(omega=omega, Ts=Ts)
    F_num, F_den, aux_num, aux_den = build_filtered_vrft_filters(M_num, M_den, W_num, W_den)

    # Filtered error & input
    e_v = signal.lfilter(aux_num, aux_den, y) - signal.lfilter(F_num, F_den, y)
    u_l = signal.lfilter(F_num, F_den, u)

    # PID regressor on filtered error
    P = e_v[1:]                      # proportional term
    D = (e_v[1:] - e_v[:-1]) / Ts    # derivative (backward diff)
    I = np.cumsum(e_v)[1:] * Ts      # integral (cumulative sum)

    Phi = np.column_stack([P, D, I])
    target = u_l[1:]

    return Phi, target


def vrft_pid_from_csv_filtered(
    csv_path: str,
    y_col: str = "y_pred_mOhm",
    u_col: str = "u_El1_pos_m",
    Ts: float = 1.0,
    tau: float = 0.0,
    t: float = 5.0,
    q: int = 3,
    omega: float = 10.0,
) -> PIDParams:
    """
    High-level filtered VRFT entry point.

    Reads a CSV with columns y_col (output) and u_col (input), and returns
    a set of PIDParams tuned such that the closed-loop approximately follows:

        M(s) = e^{-tau s} / (1 + 0.2 t s)^q

    with frequency weighting:

        W(s) = omega / (s + omega)
    """
    df = pd.read_csv(csv_path)
    if y_col not in df or u_col not in df:
        raise ValueError(f"CSV must contain '{y_col}' and '{u_col}'")

    y = df[y_col].to_numpy(dtype=float)
    u = df[u_col].to_numpy(dtype=float)

    Phi, target = build_vrft_regressors_filtered(
        y=y,
        u=u,
        Ts=Ts,
        tau=tau,
        t=t,
        q=q,
        omega=omega,
    )

    theta, *_ = lstsq(Phi, target, rcond=None)
    theta1, theta2, theta3 = theta  # P, D, I structure

    Kp = float(theta1)
    Kd = float(theta2)
    Ki = float(theta3)

    # Probably not use clamp
    # if Kp < 0.0:
    #     Kp = 0.0
    # if Ki < 0.0:
    #     Ki = 0.0

    return PIDParams(Kp=Kp, Ki=Ki, Kd=Kd)
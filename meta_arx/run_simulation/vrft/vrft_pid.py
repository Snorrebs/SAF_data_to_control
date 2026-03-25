from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter


@dataclass
class PIDParams:
    Kp: float
    Ki: float = 0.0
    Kd: float = 0.0


def _lowpass(x: np.ndarray, ts: float, omega: float) -> np.ndarray:
    """Simple first-order low-pass filter with cutoff omega [rad/s]."""
    if omega <= 0:
        raise ValueError("omega must be > 0")
    fs = 1.0 / ts
    nyquist = 0.5 * fs
    cutoff_hz = omega / (2.0 * np.pi)
    wn = min(cutoff_hz / nyquist, 0.999)
    b, a = butter(N=1, Wn=wn, btype="low")
    return lfilter(b, a, x)


def _reference_model_error(y: np.ndarray, ts: float, tau: float, t: float, q: int) -> np.ndarray:
    """Compute a virtual tracking error by passing y through a stable q-th order shaping model."""
    if q <= 0:
        raise ValueError("q must be >= 1")
    if t <= 0:
        raise ValueError("t must be > 0")

    a = np.exp(-ts / (0.2 * t))
    y_model = y.copy()
    for _ in range(q):
        z = np.zeros_like(y_model)
        for k in range(1, len(y_model)):
            z[k] = a * z[k - 1] + (1.0 - a) * y_model[k]
        y_model = z

    delay_samples = int(round(tau / ts)) if tau > 0 else 0
    if delay_samples > 0:
        y_delayed = np.concatenate([np.full(delay_samples, y_model[0]), y_model[:-delay_samples]])
    else:
        y_delayed = y_model

    return y_delayed - y


def vrft_pid_from_csv_filtered(
    csv_path: str,
    y_col: str,
    u_col: str,
    Ts: float,
    tau: float,
    t: float,
    q: int,
    omega: float,
) -> PIDParams:
    """Estimate PID parameters with a lightweight filtered VRFT-style regression."""
    df = pd.read_csv(csv_path)
    if y_col not in df.columns or u_col not in df.columns:
        raise KeyError(f"Missing required columns: {y_col}, {u_col}")

    y = df[y_col].to_numpy(dtype=float)
    u = df[u_col].to_numpy(dtype=float)

    n = min(len(y), len(u))
    y = y[:n]
    u = u[:n]

    e_v = _reference_model_error(y=y, ts=Ts, tau=tau, t=t, q=q)
    e_f = _lowpass(e_v, ts=Ts, omega=omega)
    u_f = _lowpass(u, ts=Ts, omega=omega)

    e_int = np.cumsum(e_f) * Ts
    e_der = np.gradient(e_f, Ts)

    phi = np.column_stack([e_f, e_int, e_der])
    theta, *_ = np.linalg.lstsq(phi, u_f, rcond=None)

    return PIDParams(Kp=float(theta[0]), Ki=float(theta[1]), Kd=float(theta[2]))

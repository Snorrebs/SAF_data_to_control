from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from run_simulation.closed_loop.arx_state import ArxState


_U_BASES = ["El1_pos_m_filt", "El2_pos_m_filt", "El3_pos_m_filt"]
_Y_BASES = [
    "El1_Resistance_mOhm_filt",
    "El2_Resistance_mOhm_filt",
    "El3_Resistance_mOhm_filt",
]


@dataclass
class MPCParams:
    """Tuning parameters for LinearMPC."""

    N: int = 20                                                     # prediction horizon [steps]
    Nc: int = 5                                                     # control horizon [steps]
    Q: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])        # output tracking weights (per electrode)
    R: list[float] = field(default_factory=lambda: [1000.0, 1000.0, 1000.0])  # input-rate penalty weights
    u_min: float = 0.1                                              # position lower bound [m]
    u_max: float = 2.0                                              # position upper bound [m]
    du_max: float = 0.01                                            # max rate per step [m/step]


def load_mpc_params_csv(path: str | Path) -> MPCParams:
    """Load MPC parameters from a single-row CSV.

    Accepted column names (case-insensitive):
      n, nc, q (broadcast), r (broadcast),
      q1/q2/q3 (per-electrode), r1/r2/r3 (per-electrode),
      u_min, u_max, du_max.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    row = df.iloc[0]

    p = MPCParams()

    if "n" in row.index:
        p.N = int(row["n"])
    if "nc" in row.index:
        p.Nc = int(row["nc"])
    if all(f"q{i+1}" in row.index for i in range(3)):
        p.Q = [float(row[f"q{i+1}"]) for i in range(3)]
    elif "q" in row.index:
        p.Q = [float(row["q"])] * 3
    if all(f"r{i+1}" in row.index for i in range(3)):
        p.R = [float(row[f"r{i+1}"]) for i in range(3)]
    elif "r" in row.index:
        p.R = [float(row["r"])] * 3
    if "u_min" in row.index:
        p.u_min = float(row["u_min"])
    if "u_max" in row.index:
        p.u_max = float(row["u_max"])
    if "du_max" in row.index:
        p.du_max = float(row["du_max"])

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Model extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_linear_model(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, c) such that y_physical = M @ x_physical + c.

    Expands the sklearn StandardScaler standardisation so predictions can be
    computed directly in physical units without touching the sklearn pipeline.
    """
    model = bundle["model"]
    xs = bundle["X_scaler"]
    ys = bundle.get("Y_scaler") or bundle["y_scaler"]

    # MultiOutputRegressor: one Ridge estimator per output
    W = np.vstack([est.coef_ for est in model.estimators_])          # (3, n_feat)
    b = np.array([float(np.squeeze(est.intercept_)) for est in model.estimators_])  # (3,)

    mu_x, sig_x = xs.mean_, xs.scale_
    mu_y, sig_y = ys.mean_, ys.scale_

    # y = mu_y + sig_y * ( W @ ((x - mu_x)/sig_x) + b )
    M = sig_y[:, None] * (W / sig_x[None, :])
    c = mu_y + sig_y * b - M @ mu_x
    return M, c


def _build_state_transition(
    x_cols: list[str],
    M: np.ndarray,
    c: np.ndarray,
    u_bases: list[str] = _U_BASES,
    y_bases: list[str] = _Y_BASES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build linear state-space matrices from the VARX lag structure.

    Returns (A, B_u, d) for  x+ = A @ x + B_u @ u + d
    The output equation y = M @ x + c is supplied externally.

    Parses x_cols generically:
      - u_bases lag-1 columns → driven by u_new (B_u)
      - y_bases lag-1 columns → driven by y_pred = M @ x + c (folded into A, d)
      - Other lag-1 columns   → frozen (A[j,j] = 1)
      - lag-k columns (k>1)   → shift from lag-(k-1)  (A[j, prev] = 1)
    """
    n = len(x_cols)
    col_idx = {col: j for j, col in enumerate(x_cols)}

    A = np.zeros((n, n))
    B_u = np.zeros((n, len(u_bases)))
    d = np.zeros(n)

    for col, j in col_idx.items():
        lag, base = None, None
        for k in range(1, 10):
            suffix = f"_lag{k}"
            if col.endswith(suffix):
                lag, base = k, col[: -len(suffix)]
                break
        if lag is None:
            continue

        if lag == 1:
            if base in u_bases:
                B_u[j, u_bases.index(base)] = 1.0
            elif base in y_bases:
                y_idx = y_bases.index(base)
                A[j, :] = M[y_idx, :]
                d[j] = c[y_idx]
            else:
                A[j, j] = 1.0  # exogenous: frozen
        else:
            prev_col = col.replace(f"_lag{lag}", f"_lag{lag - 1}")
            if prev_col in col_idx:
                A[j, col_idx[prev_col]] = 1.0

    return A, B_u, d


# ─────────────────────────────────────────────────────────────────────────────
# MPC controller
# ─────────────────────────────────────────────────────────────────────────────

class LinearMPC:
    """Receding-horizon MIMO MPC for 3-electrode VARX resistance control.

    Uses the linearised (physical-space) ARX model to build multi-step
    predictions and solves a constrained QP via scipy SLSQP at each step.
    The gradient is computed analytically using the adjoint (backward-pass)
    method, avoiding finite differences.

    Decision variables: incremental electrode moves ΔU ∈ R^{Nc×3}.
    Cost: Σ_k ||y_k - r_k||²_Q  +  Σ_j ||Δu_j||²_R
    Constraints: position bounds and rate limits (both enforced as hard constraints).
    """

    def __init__(self, params: MPCParams, bundle: dict) -> None:
        self.params = params
        self._x_cols: list[str] = bundle["X_cols"]
        self._xs = bundle["X_scaler"]

        self.M, self.c = _extract_linear_model(bundle)
        self.A, self.B_u, self.d = _build_state_transition(
            self._x_cols, self.M, self.c
        )

    def reset(self) -> None:
        pass  # stateless: no inter-step memory needed

    # ── helpers ──────────────────────────────────────────────────────────────

    def _x_from_state(self, state: ArxState) -> np.ndarray:
        x = state.row.reindex(self._x_cols).to_numpy(dtype=float)
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x[nan_mask] = self._xs.mean_[nan_mask]
        return x

    # ── public interface ──────────────────────────────────────────────────────

    def step(
        self,
        reference_window: np.ndarray,
        state: ArxState,
        u_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve MPC and return the first optimal control move.

        Args:
            reference_window: setpoint trajectory over the horizon, shape
                ``(N, 3)`` or ``(3,)`` (held constant if scalar) [mΩ]
            state:   current ArxState (read-only)
            u_prev:  (3,) last applied electrode positions [m]

        Returns:
            u_des: (3,) desired positions [m]  (actuator limits applied externally)
            e:     (3,) tracking error = ref[0] - y_linearised  [mΩ]
        """
        p = self.params
        N, Nc = p.N, p.Nc
        A, B_u, dv = self.A, self.B_u, self.d
        M, c = self.M, self.c

        # ── reference window (N, 3) ───────────────────────────────────────
        ref = np.asarray(reference_window, dtype=float)
        if ref.ndim == 1:
            ref = np.tile(ref, (N, 1))
        if ref.shape[0] < N:
            ref = np.vstack([ref, np.tile(ref[-1:], (N - ref.shape[0], 1))])
        ref = ref[:N]

        x0 = self._x_from_state(state)
        u0 = np.asarray(u_prev, dtype=float)
        Q = np.asarray(p.Q, dtype=float)   # (3,)
        R = np.asarray(p.R, dtype=float)   # (3,)

        # ── objective + analytical gradient (adjoint method) ─────────────
        def cost_and_grad(du_flat: np.ndarray) -> tuple[float, np.ndarray]:
            dU = du_flat.reshape(Nc, 3)
            U = u0 + np.cumsum(dU, axis=0)          # absolute positions (Nc, 3)

            # Forward pass
            X = np.empty((N + 1, len(x0)))
            Y = np.empty((N, 3))
            X[0] = x0
            for k in range(N):
                Y[k] = M @ X[k] + c
                X[k + 1] = A @ X[k] + B_u @ U[min(k, Nc - 1)] + dv

            E = Y - ref                              # (N, 3) residuals
            J = float(np.sum(E**2 * Q) + np.sum(dU**2 * R))

            # Backward adjoint pass: λ_k = ∂J/∂x_k
            lam = np.zeros(len(x0))
            grad_U = np.zeros((Nc, 3))               # ∂J/∂U[m]
            for k in range(N - 1, -1, -1):
                lam = A.T @ lam + M.T @ (2.0 * Q * E[k])
                grad_U[min(k, Nc - 1)] += B_u.T @ lam

            # Chain rule: ∂J/∂ΔU[j] = Σ_{m≥j} ∂J/∂U[m]  (reverse cumsum)
            grad_dU = np.flipud(np.cumsum(np.flipud(grad_U), axis=0))
            grad_dU += 2.0 * R * dU

            return J, grad_dU.reshape(-1)

        # ── bounds (rate limits) ──────────────────────────────────────────
        bounds = [(-p.du_max, p.du_max)] * (Nc * 3)

        # ── position constraints ──────────────────────────────────────────
        # u_prev[i] + Σ_{j=0}^{k} ΔU[j,i] ∈ [u_min, u_max]  ∀ k, i
        T = np.tril(np.ones((Nc, Nc)))  # cumsum matrix
        constraints = []
        for i in range(3):
            cols_i = np.array([j * 3 + i for j in range(Nc)])
            for k in range(Nc):
                row = np.zeros(Nc * 3)
                row[cols_i] = T[k, :]
                constraints += [
                    {
                        "type": "ineq",
                        "fun": lambda df, r=row, ui=u0[i]: r @ df + ui - p.u_min,
                        "jac": lambda df, r=row: r,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda df, r=row, ui=u0[i]: p.u_max - (r @ df + ui),
                        "jac": lambda df, r=row: -r,
                    },
                ]

        result = minimize(
            cost_and_grad,
            np.zeros(Nc * 3),
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 300, "disp": False},
        )

        dU_opt = result.x.reshape(Nc, 3)
        u_des = u0 + dU_opt[0]
        e = ref[0] - (M @ x0 + c)
        return u_des, e

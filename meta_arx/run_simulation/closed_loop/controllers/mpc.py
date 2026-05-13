from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from run_simulation.closed_loop.arx_state import ArxState


_U_BASES = ["El1_pos_m_filt", "El2_pos_m_filt", "El3_pos_m_filt"]
_Y_BASES = [
    "El1_Resistance_mOhm_filt",
    "El2_Resistance_mOhm_filt",
    "El3_Resistance_mOhm_filt",
]


@dataclass
class MPCParams:
    """Tuning parameters for LinearMPC.

    The MPC optimises directly over Δu (holder movements per step) rather
    than absolute positions. This is consistent with the VARX model, which
    was trained on Δpos (first-differenced holder position) as its control
    input. Absolute position is not tracked — it is not observable from the
    model state and cannot be reliably reconstructed due to electrode
    consumption and slip.

    Constraints:
      du_min / du_max : symmetric rate limits on movement per step [m/step]
    """

    N:      int         = 15
    Nc:     int         = 5
    Q:      list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    R:      list[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])
    du_min: float       = -0.01
    du_max: float       =  0.01


def load_mpc_params_csv(path: str | Path) -> MPCParams:
    """Load MPC parameters from a single-row CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    row = df.iloc[0]
    p   = MPCParams()

    if "n"  in row.index: p.N  = int(row["n"])
    if "nc" in row.index: p.Nc = int(row["nc"])

    if all(f"q{i+1}" in row.index for i in range(3)):
        p.Q = [float(row[f"q{i+1}"]) for i in range(3)]
    elif "q" in row.index:
        p.Q = [float(row["q"])] * 3

    if all(f"r{i+1}" in row.index for i in range(3)):
        p.R = [float(row[f"r{i+1}"]) for i in range(3)]
    elif "r" in row.index:
        p.R = [float(row["r"])] * 3

    if "du_min" in row.index:
        p.du_min = float(row["du_min"])
    elif "du_max" in row.index:
        p.du_min = -float(row["du_max"])

    if "du_max" in row.index:
        p.du_max = float(row["du_max"])

    if ("u_min" in row.index or "u_max" in row.index) and "du_max" not in row.index:
        print(
            "  WARNING: MPC CSV contains u_min/u_max but not du_max. "
            "MPC now operates in Δu space — add du_max to MPC_params.csv."
        )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Model extraction  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_linear_model(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, c) such that y_physical = M @ x_physical + c."""
    ys       = bundle.get("Y_scaler") or bundle["y_scaler"]
    mu_y, sig_y = ys.mean_, ys.scale_

    if "models" in bundle and "X_scalers" in bundle and "X_cols_per_eq" in bundle:
        models      = bundle["models"]
        x_scalers   = bundle["X_scalers"]
        x_cols_flat = bundle["X_cols"]
        n_out       = len(models)
        n_feat      = len(x_cols_flat)
        flat_idx    = {c: i for i, c in enumerate(x_cols_flat)}
        x_col_lists = bundle["X_cols_per_eq"]

        M = np.zeros((n_out, n_feat))
        c = np.zeros(n_out)

        for ei in range(n_out):
            xs    = x_scalers[ei]
            ridge = models[ei]
            cols  = x_col_lists[ei]
            mu_x  = xs.mean_
            sig_x = xs.scale_
            W     = ridge.coef_
            b     = float(np.squeeze(ridge.intercept_))
            M_eq  = sig_y[ei] * (W / sig_x)
            c_eq  = mu_y[ei] + sig_y[ei] * b - M_eq @ mu_x
            for j, col in enumerate(cols):
                M[ei, flat_idx[col]] = M_eq[j]
            c[ei] = c_eq
        return M, c

    # Legacy bundle
    model        = bundle["model"]
    xs           = bundle["X_scaler"]
    mu_x, sig_x  = xs.mean_, xs.scale_
    W = np.vstack([est.coef_ for est in model.estimators_])
    b = np.array([float(np.squeeze(est.intercept_)) for est in model.estimators_])
    M = sig_y[:, None] * (W / sig_x[None, :])
    c = mu_y + sig_y * b - M @ mu_x
    return M, c


# ─────────────────────────────────────────────────────────────────────────────
# Reduced state space
# ─────────────────────────────────────────────────────────────────────────────

def _build_reduced_model(
    x_cols:   list[str],
    M_full:   np.ndarray,
    c_full:   np.ndarray,
    u_bases:  list[str] = _U_BASES,
    y_bases:  list[str] = _Y_BASES,
) -> tuple:
    """Build reduced-order matrices containing only AR and Δpos lag columns.

    Frozen exogenous columns (kA, etc.) are excluded from the dynamic state.
    Their contribution to the output is a constant offset computed once per
    MPC call from the current frozen state values — exact because advance()
    freezes them between steps.

    Returns
    -------
    A_r        : (nr, nr)   reduced state transition
    Bu_r       : (nr, 3)    input matrix (Δu → reduced state)
    M_r        : (3, nr)    output matrix (reduced state → y)
    c_full     : (3,)       full output offset (frozen part added at call time)
    d_r        : (nr,)      constant drift (from cross-AR offsets)
    active_idx : list[int]  indices of active columns in x_cols
    frozen_idx : list[int]  indices of frozen columns
    frozen_M   : (3, nf)    output weights for frozen columns
    """
    col_idx = {c: i for i, c in enumerate(x_cols)}
    n_full  = len(x_cols)

    def is_active(col: str) -> bool:
        if "Resistance_mOhm_filt_lag" in col:
            return True
        if any(col.startswith(f"El{i+1}_pos_m_filt_lag") for i in range(3)):
            return True
        return False

    active_idx = [i for i, c in enumerate(x_cols) if is_active(c)]
    frozen_idx = [i for i in range(n_full) if i not in active_idx]
    nr         = len(active_idx)

    # ── Build full A, Bu, d ──────────────────────────────────────────────────
    A_full  = np.zeros((n_full, n_full))
    Bu_full = np.zeros((n_full, len(u_bases)))
    d_full  = np.zeros(n_full)

    for col, j in col_idx.items():
        if "->" in col:
            arrow_pos = col.index("->")
            left      = col[:arrow_pos]
            lag, base = None, None
            for k in range(1, 20):
                sfx = f"_lag{k}"
                if left.endswith(sfx):
                    lag, base = k, left[: -len(sfx)]
                    break
            if lag is None:
                continue
            if lag == 1:
                if base in y_bases:
                    yi           = y_bases.index(base)
                    A_full[j, :] = M_full[yi, :]
                    d_full[j]    = c_full[yi]
                else:
                    A_full[j, j] = 1.0
            else:
                prev = col.replace(f"_lag{lag}->", f"_lag{lag-1}->")
                if prev in col_idx:
                    A_full[j, col_idx[prev]] = 1.0
            continue

        lag, base = None, None
        for k in range(1, 20):
            sfx = f"_lag{k}"
            if col.endswith(sfx):
                lag, base = k, col[: -len(sfx)]
                break
        if lag is None:
            continue

        if lag == 1:
            if base in u_bases:
                Bu_full[j, u_bases.index(base)] = 1.0
            elif base in y_bases:
                yi           = y_bases.index(base)
                A_full[j, :] = M_full[yi, :]
                d_full[j]    = c_full[yi]
            else:
                A_full[j, j] = 1.0
        else:
            prev = col.replace(f"_lag{lag}", f"_lag{lag-1}")
            if prev in col_idx:
                A_full[j, col_idx[prev]] = 1.0

    # ── Extract reduced submatrices ──────────────────────────────────────────
    A_r      = A_full[np.ix_(active_idx, active_idx)]
    Bu_r     = Bu_full[active_idx, :]
    d_r      = d_full[active_idx]
    M_r      = M_full[:, active_idx]
    frozen_M = M_full[:, frozen_idx]

    return A_r, Bu_r, M_r, c_full, d_r, active_idx, frozen_idx, frozen_M


# ─────────────────────────────────────────────────────────────────────────────
# MPC controller
# ─────────────────────────────────────────────────────────────────────────────

class LinearMPC:
    """Receding-horizon MIMO MPC for 3-electrode VARX resistance control.

    Optimises over Δu (movements) in a reduced state space containing only
    AR and Δpos lag columns. Frozen exogenous signals are handled as a
    constant offset, making the adjoint backward pass ~4x faster than
    operating on the full state.

    Cost:
        J = Σ_{k=1}^{N}  ||y_k - r_k||²_Q  +  Σ_{j=0}^{Nc-1} ||Δu_j||²_R

    Constraints:
        du_min ≤ Δu_j ≤ du_max  (symmetric rate limits)
    """

    def __init__(self, params: MPCParams, bundle: dict) -> None:
        self.params  = params
        self._x_cols = bundle["X_cols"]
        self._xs     = bundle["X_scaler"]
        Nc=self.params.Nc
        self._du_prev_sol = np.zeros((Nc, 3))

        self.M_full, self.c_full = _extract_linear_model(bundle)

        (self.A_r, self.Bu_r, self.M_r, self.c_full,
         self.d_r, self.active_idx, self.frozen_idx,
         self.frozen_M) = _build_reduced_model(
            self._x_cols, self.M_full, self.c_full
        )

        nr = len(self.active_idx)
        nf = len(self.frozen_idx)
        print(f"  LinearMPC: state reduced {len(self._x_cols)} → {nr} "
              f"(+{nf} frozen as constant offset)")

        # Sparse matrices for fast adjoint pass
        self.A_sp  = csr_matrix(self.A_r)
        self.Bu_sp = csr_matrix(self.Bu_r)
        self.MT_sp = csr_matrix(self.M_r.T)

        # AR subspace stability check (own resistance lags only)
        ar_cols = [c for c in self._x_cols
                   if "Resistance_mOhm_filt_lag" in c and "->" not in c]
        x_list  = list(self._x_cols)
        ar_in_active = [self.active_idx.index(x_list.index(c))
                        for c in ar_cols if x_list.index(c) in self.active_idx]
        if ar_in_active:
            A_ar = self.A_r[np.ix_(ar_in_active, ar_in_active)]
            me   = float(np.max(np.abs(np.linalg.eigvals(A_ar))))
            print(f"  LinearMPC: AR subspace max|eigenvalue| = {me:.4f}  "
                  f"({'stable' if me < 1.0 else 'UNSTABLE ⚠'})")

    def reset(self) -> None:
        pass

    def _get_reduced_state(self, state: ArxState) -> tuple[np.ndarray, np.ndarray]:
        x_full      = state.row.reindex(self._x_cols).to_numpy(dtype=float)
        nan_m       = np.isnan(x_full)
        if nan_m.any():
            x_full[nan_m] = self._xs.mean_[nan_m]
        return x_full[self.active_idx], x_full[self.frozen_idx]

    def step(
        self,
        reference_window: np.ndarray,
        state:    ArxState,
        du_prev:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve one MPC step.

        Args:
            reference_window: (N, 3) or (3,) setpoint [mΩ]
            state:   current ArxState
            du_prev: last Δu (for R warm-start, not position tracking)

        Returns:
            du_opt: (3,) optimal movement this step [m/step]
            e:      (3,) tracking error [mΩ]
        """
        p     = self.params
        N, Nc = p.N, p.Nc

        # Reference window
        ref = np.asarray(reference_window, dtype=float)
        if ref.ndim == 1:
            ref = np.tile(ref, (N, 1))
        if ref.shape[0] < N:
            ref = np.vstack([ref, np.tile(ref[-1:], (N - ref.shape[0], 1))])
        ref = ref[:N]

        # Reduced state + frozen offset (computed once per step)
        x0_r, x_frozen = self._get_reduced_state(state)
        frozen_offset  = self.frozen_M @ x_frozen     # (3,) — constant over horizon
        c_eff          = self.c_full + frozen_offset   # effective output offset

        Q      = np.asarray(p.Q, dtype=float)
        R      = np.asarray(p.R, dtype=float)
        A_sp   = self.A_sp
        Bu_sp  = self.Bu_sp
        MT_sp  = self.MT_sp
        M_r    = self.M_r
        d_r    = self.d_r

        
        def cost_and_grad(du_flat: np.ndarray) -> tuple[float, np.ndarray]:
            dU = du_flat.reshape(Nc, 3)

            # ── Forward pass ────────────────────────────────────────────────────────
            X    = np.empty((N + 1, len(x0_r)))
            Y    = np.empty((N, 3))
            X[0] = x0_r
            for k in range(N):
                Y[k]     = M_r @ X[k] + c_eff
                X[k + 1] = A_sp @ X[k] + Bu_sp @ dU[min(k, Nc - 1)] + d_r

            if not np.isfinite(Y).all():
                return 1e30, np.zeros(Nc * 3)

            E  = Y - ref
            J  = float(np.sum(E ** 2 * Q) + np.sum(dU ** 2 * R))

            # ── Backward adjoint pass ────────────────────────────────────────────────
            # dJ/d(dU[j]) = Σ_{k=j}^{N-1}  Bu_r.T @ λ_k    (where k uses dU[min(k,Nc-1)])
            # Accumulate into the correct Nc slot directly.
            lam    = np.zeros(len(x0_r))
            grad_U = np.zeros((Nc, 3))
            for k in range(N - 1, -1, -1):
                lam = A_sp.T @ lam + MT_sp @ (2.0 * Q * E[k])
                grad_U[min(k, Nc - 1)] += Bu_sp.T @ lam

            # Regularisation gradient
            grad_U += 2.0 * R * dU

            return J, grad_U.reshape(-1)

        bounds = [(p.du_min, p.du_max)] * (Nc * 3)
        x0_warm = np.vstack([self._du_prev_sol[1:], self._du_prev_sol[-1:]])

        result = minimize(
            cost_and_grad,
            x0_warm.reshape(-1),
            method  = "SLSQP",
            jac     = True,
            bounds  = bounds,
            options = {"ftol": 1e-6, "maxiter": 300, "disp": False},
        )
        self._du_prev_sol = result.x.reshape(Nc, 3)    
        dU_opt = result.x.reshape(Nc, 3)
        du_opt = dU_opt[0]
        e      = ref[0] - (M_r @ x0_r + c_eff)
        return du_opt, e
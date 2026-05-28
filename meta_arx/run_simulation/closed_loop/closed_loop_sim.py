from __future__ import annotations

import re
import numpy as np

from .arx_state import ArxState
from .controller_api import Controller


# Per-electrode rate limit (shared across all electrodes)
_DPOS_MAX = 0.01   # max movement per step [m/step]
_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<lag>\d+)(?P<suffix>.*)$")


def apply_rate_limit(dpos_des: float) -> float:
    """Clip requested movement to the physical rate limit."""
    return float(np.clip(dpos_des, -_DPOS_MAX, _DPOS_MAX))


def _as_3col_reference(reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError(f"reference must have shape (n,) or (n, 3), got {reference.shape}")
    return reference


def _all_x_cols(bundle: dict) -> list[str]:
    if "X_cols" in bundle:
        return list(bundle["X_cols"])
    if "X_cols_flat" in bundle:
        return list(bundle["X_cols_flat"])
    if "X_cols_per_eq" in bundle:
        return [col for cols in bundle["X_cols_per_eq"] for col in cols]
    raise KeyError("Bundle missing X column definitions")


def _lag_bases(state: ArxState) -> set[str]:
    bases: set[str] = set()
    for col in state.row.index:
        match = _LAG_RE.match(str(col))
        if match:
            bases.add(match.group("base"))
    return bases


def _first_available_base(state: ArxState, candidates: list[str]) -> str:
    available = _lag_bases(state) | set(map(str, state.row.index))
    for base in candidates:
        if base in available:
            return base
    return candidates[0]


def _current_update_bases(state: ArxState) -> list[str]:
    return [
        _first_available_base(state, [f"El{i}_kA_filt", f"El{i}_kA", f"kA{i}"])
        for i in range(1, 4)
    ]


def _resistance_update_bases(state: ArxState) -> list[str]:
    return [
        _first_available_base(state, [f"El{i}_Resistance_mOhm_filt", f"R_tilde_El{i}"])
        for i in range(1, 4)
    ]


def _current_model_resistance_lag0_overrides(
    current_model: dict,
    r_abs: np.ndarray,
) -> dict[str, float]:
    """Build temporary same-time resistance inputs for the current model."""
    overrides: dict[str, float] = {}
    x_cols = _all_x_cols(current_model)

    for i in range(1, 4):
        prefix = f"El{i}_Resistance_mOhm_filt_lag0"
        for col in x_cols:
            if str(col).startswith(prefix):
                overrides[str(col)] = float(r_abs[i - 1])

    return overrides


def _ka_exog_for_resistance_state(state: ArxState, kA: np.ndarray) -> dict[str, float]:
    bases = _current_update_bases(state)
    return {bases[i]: float(kA[i]) for i in range(3)}


def run_closed_loop(
    model: dict,
    state: ArxState,
    reference: np.ndarray,
    controllers: list[Controller],
    exog_traj: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run closed-loop simulation for a 3-output VARX model.

    The control signal is holder movement per step [m/step], which is the
    first-differenced actuator input used during model identification.
    """
    reference = _as_3col_reference(reference)
    n = reference.shape[0]

    y = np.zeros((n + 1, 3), dtype=float)
    dpos = np.zeros((n, 3), dtype=float)
    e = np.zeros((n, 3), dtype=float)

    y_cols = model.get("y_cols") or [model.get("y_col")]
    y[0] = state.current_y(y_cols=y_cols)

    if len(controllers) != 3:
        raise ValueError(f"Expected 3 controllers, got {len(controllers)}")

    for controller in controllers:
        controller.reset()

    for k in range(n):
        y_pred = state.predict_next_y(model)

        dpos_cmd = np.zeros(3, dtype=float)
        for i in range(3):
            dpos_des, e[k, i] = controllers[i].step(
                reference=float(reference[k, i]),
                y_pred=float(y_pred[i]),
                dpos_prev=0.0,
            )
            dpos_cmd[i] = apply_rate_limit(dpos_des)

        dpos[k] = dpos_cmd
        y[k + 1] = y_pred

        exog_k: dict[str, float] | None = None
        if exog_traj:
            exog_k = {sig: float(traj[k]) for sig, traj in exog_traj.items()}

        state.advance(u_new=dpos_cmd, y_new=y_pred, exog_new=exog_k)

    return y, dpos, e


def run_coupled_closed_loop(
    resistance_model: dict,
    resistance_state: ArxState,
    current_model: dict,
    current_state: ArxState,
    reference: np.ndarray,
    controllers: list[Controller],
    trend: np.ndarray,
    ka_noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run coupled closed-loop simulation with internally predicted current.

    The resistance model predicts detrended resistance R_tilde. Absolute
    resistance is reconstructed as

        R_abs = trend + R_tilde

    The current model then predicts electrode current from current lags and
    same-time absolute resistance. Optional zero-mean current disturbance can
    be added to the predicted current.

    Args:
        resistance_model:  trained VARX bundle for R_tilde prediction
        resistance_state:  lag state for the resistance model
        current_model:     trained VARX bundle for current prediction
        current_state:     lag state for the current model
        reference:         shape (n,) or (n, 3), reference in R_tilde space
        controllers:       list of 3 Controller instances
        trend:             shape (3,), absolute resistance baseline [mOhm]
        ka_noise:          optional zero-mean disturbance, shape (n+1, 3) or (n, 3)

    Returns:
        y:      (n+1, 3) predicted detrended resistance R_tilde [mOhm]
        dpos:   (n,   3) applied holder movement [m/step]
        e:      (n,   3) tracking error in R_tilde space [mOhm]
        kA:     (n+1, 3) predicted/disturbed electrode current [kA]
        r_abs:  (n+1, 3) reconstructed absolute resistance [mOhm]
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)
    if reference.shape[1] != 3:
        raise ValueError(f"reference must have 3 columns, got {reference.shape}")

    trend = np.asarray(trend, dtype=float).reshape(-1)
    if trend.shape != (3,):
        raise ValueError(f"trend must have shape (3,), got {trend.shape}")

    n = reference.shape[0]

    if ka_noise is not None:
        ka_noise = np.asarray(ka_noise, dtype=float)
        if ka_noise.shape == (n, 3):
            ka_noise = np.vstack([np.zeros((1, 3)), ka_noise])
        if ka_noise.shape != (n + 1, 3):
            raise ValueError(
                f"ka_noise must have shape {(n, 3)} or {(n + 1, 3)}, "
                f"got {ka_noise.shape}"
            )

    y = np.zeros((n + 1, 3), dtype=float)
    dpos = np.zeros((n, 3), dtype=float)
    e = np.zeros((n, 3), dtype=float)
    kA = np.zeros((n + 1, 3), dtype=float)
    r_abs = np.zeros((n + 1, 3), dtype=float)

    resistance_y_cols = resistance_model.get("y_cols") or [resistance_model.get("y_col")]
    current_y_cols = current_model.get("y_cols") or [current_model.get("y_col")]

    y[0] = resistance_state.current_y(y_cols=resistance_y_cols)
    r_abs[0] = trend + y[0]
    kA[0] = current_state.current_y(y_cols=current_y_cols)

    if ka_noise is not None:
        kA[0] = kA[0] + ka_noise[0]

    kA[0] = np.clip(kA[0], 80.0, 180.0)

    if len(controllers) != 3:
        raise ValueError(f"Expected 3 controllers, got {len(controllers)}")

    for controller in controllers:
        controller.reset()

    for k in range(n):
        # 1. Predict next detrended resistance from resistance VARX state.
        y_pred = resistance_state.predict_next_y(resistance_model)
        y[k + 1] = y_pred

        # 2. Reconstruct absolute resistance for the current model.
        r_abs_pred = trend + y_pred
        r_abs[k + 1] = r_abs_pred

        # 3. Predict same-time current using R_abs(k) as lag0 override.
        current_overrides = {
            "El1_Resistance_mOhm_filt_lag0": float(r_abs_pred[0]),
            "El2_Resistance_mOhm_filt_lag0": float(r_abs_pred[1]),
            "El3_Resistance_mOhm_filt_lag0": float(r_abs_pred[2]),
        }

        kA_pred = current_state.predict_next_y(
            current_model,
            overrides=current_overrides,
        )

        # 4. Add optional zero-mean current disturbance.
        if ka_noise is not None:
            kA_pred = kA_pred + ka_noise[k + 1]

        kA_pred = np.clip(kA_pred, 80.0, 180.0)
        kA[k + 1] = kA_pred

        # 5. Compute movement command from controller.
        dpos_cmd = np.zeros(3, dtype=float)
        for i in range(3):
            dpos_des, e[k, i] = controllers[i].step(
                reference=float(reference[k, i]),
                y_pred=float(y_pred[i]),
                dpos_prev=float(dpos[k - 1, i]) if k > 0 else 0.0,
            )
            dpos_cmd[i] = apply_rate_limit(dpos_des)

        dpos[k] = dpos_cmd

        # 6. Advance resistance-model state.
        resistance_state.advance_lags({
            "El1_Resistance_mOhm_filt": float(y_pred[0]),
            "El2_Resistance_mOhm_filt": float(y_pred[1]),
            "El3_Resistance_mOhm_filt": float(y_pred[2]),

            "R_tilde_El1": float(y_pred[0]),
            "R_tilde_El2": float(y_pred[1]),
            "R_tilde_El3": float(y_pred[2]),

            "El1_dpos_m_filt": float(dpos_cmd[0]),
            "El2_dpos_m_filt": float(dpos_cmd[1]),
            "El3_dpos_m_filt": float(dpos_cmd[2]),

            "El1_pos_m_filt": float(dpos_cmd[0]),
            "El2_pos_m_filt": float(dpos_cmd[1]),
            "El3_pos_m_filt": float(dpos_cmd[2]),

            "kA1": float(kA_pred[0]),
            "kA2": float(kA_pred[1]),
            "kA3": float(kA_pred[2]),
        })

        # 7. Advance current-model state.
        current_state.advance_lags({
            "El1_kA_filt": float(kA_pred[0]),
            "El2_kA_filt": float(kA_pred[1]),
            "El3_kA_filt": float(kA_pred[2]),

            "kA1": float(kA_pred[0]),
            "kA2": float(kA_pred[1]),
            "kA3": float(kA_pred[2]),

            "El1_Resistance_mOhm_filt": float(r_abs_pred[0]),
            "El2_Resistance_mOhm_filt": float(r_abs_pred[1]),
            "El3_Resistance_mOhm_filt": float(r_abs_pred[2]),
        })

    return y, dpos, e, kA, r_abs


def run_mpc_closed_loop(
    model: dict,
    state: ArxState,
    reference: np.ndarray,
    mpc,
    exog_traj: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run closed-loop simulation with a MIMO MPC controller.

    The MPC optimises over holder movement per step and returns that movement
    directly. The state is advanced with the same input convention used during
    model identification.
    """
    reference = _as_3col_reference(reference)
    n = reference.shape[0]
    N = mpc.params.N

    y = np.zeros((n + 1, 3), dtype=float)
    dpos = np.zeros((n, 3), dtype=float)
    e = np.zeros((n, 3), dtype=float)

    y_cols = model.get("y_cols") or [model.get("y_col")]
    y[0] = state.current_y(y_cols=y_cols)

    dpos_prev = state.current_u()
    mpc.reset()

    for k in range(n):
        y_pred = state.predict_next_y(model)
        ref_window = reference[k : k + N]

        dpos_des, _ = mpc.step(ref_window, state, dpos_prev)
        dpos_cmd = np.clip(dpos_des, mpc.params.du_min, mpc.params.du_max)

        dpos[k] = dpos_cmd
        y[k + 1] = y_pred
        e[k] = reference[k] - y_pred

        exog_k: dict[str, float] | None = None
        if exog_traj:
            exog_k = {sig: float(traj[k]) for sig, traj in exog_traj.items()}

        state.advance(u_new=dpos_cmd, y_new=y_pred, exog_new=exog_k)
        dpos_prev = dpos_cmd

    return y, dpos, e

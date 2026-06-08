"""
fusion/run_closed_loop_rollout.py
---------------------------------
Drop-in replacement for run_closed_loop_from_config that uses:
  - varx_pi_retrained.joblib   step9 VARX (Fusion_multielectrode, R_tilde space)
  - gp_el{1,2,3}_rollout.pt   rollout SVGP correction (same naming convention)

The VARX state uses Fusion_multielectrode column names which match the GP
feature names exactly — no translation layer needed.

Usage
-----
    from fusion.run_closed_loop_rollout import run_closed_loop_from_config
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_META_ARX     = _PROJECT_ROOT / "meta_arx"
for _p in [str(_PROJECT_ROOT), str(_META_ARX)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion.training.gp_loader import predict_single_certainty

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_MODELS_DIR     = _HERE / "models"
_VARX_PATH      = _MODELS_DIR / "varx_pi_retrained.joblib"
_DETREND_WINDOW = 1800

_OP_KA   = 118.0
_OP_REAC = 0.88
_OP_R    = {1: 1.08, 2: 1.07, 3: 1.07}
_OP_POS  = {1: 1.04, 2: 1.03, 3: 1.04}
_OP_V    = 165.0
_DPOS_MAX = 0.01
_U_MIN, _U_MAX = 0.0, 2.0

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_varx_bundle: dict | None = None
_gp_bundles:  dict[int, dict] = {}


def _load_models() -> None:
    global _varx_bundle, _gp_bundles
    if _varx_bundle is not None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fusion.training.train_joint_arx_v3 import ReducedRankRidge  # noqa
        _varx_bundle = joblib.load(_VARX_PATH)

    for el in (1, 2, 3):
        p = _MODELS_DIR / f"gp_el{el}_rollout.pt"
        if p.exists():
            _gp_bundles[el] = torch.load(str(p), map_location="cpu", weights_only=False)
            print(f"[rollout] El{el} GP: {p.name}  features={len(_gp_bundles[el]['feature_names'])}")
        else:
            print(f"[rollout] El{el} GP not found at {p.name}")
    print(f"[rollout] VARX: {_varx_bundle['model_name']}")


# ---------------------------------------------------------------------------
# VARX prediction
# ---------------------------------------------------------------------------

def _varx_step(state: dict) -> np.ndarray:
    xcols_per_eq = _varx_bundle["X_cols_per_eq"]
    models       = _varx_bundle["models"]
    x_scalers    = _varx_bundle.get("X_scalers")
    preds = np.zeros(3, dtype=float)
    for eq_i, (xcols_eq, model) in enumerate(zip(xcols_per_eq, models)):
        x = np.array([state.get(c, 0.0) for c in xcols_eq], dtype=float).reshape(1, -1)
        if x_scalers:
            x = x_scalers[eq_i].transform(x)
        preds[eq_i] = float(model.predict(x)[0])
    return preds


# ---------------------------------------------------------------------------
# GP correction (features come directly from VARX state — same naming)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gp_correct(el: int, state: dict, y_sim_el: float) -> float:
    bun = _gp_bundles.get(el)
    if bun is None:
        return 0.0
    x_mean = bun["x_mean"]; x_std = bun["x_std"]
    y_mean = float(bun["y_mean"][0]); y_std = float(bun["y_std"][0])
    x_raw = np.array([
        y_sim_el if f == "y_sim" else float(state.get(f, 0.0))
        for f in bun["feature_names"]
    ], dtype=np.float32)
    x_s = ((x_raw - x_mean) / x_std).reshape(1, -1)
    model = bun["model"]; lik = bun["likelihood"]
    model.eval(); lik.eval()
    pred = lik(model(torch.tensor(x_s)))
    return float(np.clip(pred.mean.item() * y_std + y_mean, -0.15, 0.15))


# ---------------------------------------------------------------------------
# VARX state initialisation and update
# ---------------------------------------------------------------------------

def _init_varx_state() -> dict:
    """Seed step9 VARX state at typical operating point (R_tilde = 0 at t=0)."""
    xcols_flat = _varx_bundle.get("X_cols_flat", _varx_bundle.get("X_cols", []))
    state = {c: 0.0 for c in xcols_flat}
    for i in (1, 2, 3):
        for k in range(1, 11):
            state[f"El{i}_Resistance_mOhm_filt_lag{k}"] = 0.0          # R_tilde = 0 at OP
            state[f"El{i}_pos_m_filt_lag{k}"]           = float(_OP_POS[i])
            state[f"kA{i}_lag{k}"]                      = float(_OP_KA)
        for j in range(i + 1, 4):
            for k in range(1, 11):
                state[f"El{i}_Resistance_mOhm_filt_lag{k}->El{j}"] = 0.0
    return state


def _update_varx_state(state: dict, r_pred_tilde: np.ndarray,
                        dpos: np.ndarray, ka: np.ndarray) -> None:
    """Advance VARX lag registers. R lags use VARX prediction (lag-feedback sep.)"""
    for i in (1, 2, 3):
        rp = float(r_pred_tilde[i - 1])
        # Shift R_tilde lags
        for k in range(10, 1, -1):
            state[f"El{i}_Resistance_mOhm_filt_lag{k}"] = \
                state.get(f"El{i}_Resistance_mOhm_filt_lag{k-1}", 0.0)
            for j in range(i + 1, 4):
                key = f"El{i}_Resistance_mOhm_filt_lag{k}->El{j}"
                if key in state:
                    state[key] = state.get(
                        f"El{i}_Resistance_mOhm_filt_lag{k-1}->El{j}", 0.0)
        state[f"El{i}_Resistance_mOhm_filt_lag1"] = rp
        for j in range(i + 1, 4):
            k1 = f"El{i}_Resistance_mOhm_filt_lag1->El{j}"
            if k1 in state:
                state[k1] = rp
        # Shift position lags (absolute position)
        new_pos = np.clip(state.get(f"El{i}_pos_m_filt_lag1", _OP_POS[i])
                          + float(dpos[i - 1]), _U_MIN, _U_MAX)
        for k in range(10, 1, -1):
            state[f"El{i}_pos_m_filt_lag{k}"] = state.get(
                f"El{i}_pos_m_filt_lag{k-1}", float(_OP_POS[i]))
        state[f"El{i}_pos_m_filt_lag1"] = new_pos
        # Shift kA lags (updated from external trajectory or kept constant)
        for k in range(10, 1, -1):
            state[f"kA{i}_lag{k}"] = state.get(f"kA{i}_lag{k-1}", float(_OP_KA))
        state[f"kA{i}_lag1"] = float(ka[i - 1])


# ---------------------------------------------------------------------------
# Reference loading helper
# ---------------------------------------------------------------------------

def _load_ref(ref_csv: str | Path) -> np.ndarray:
    df = pd.read_csv(ref_csv)
    if all(f"r{i}" in df.columns for i in (1, 2, 3)):
        return df[["r1", "r2", "r3"]].to_numpy(dtype=float)
    if "reference" in df.columns:
        r = df["reference"].to_numpy(dtype=float).reshape(-1, 1)
        return np.repeat(r, 3, axis=1)
    raise ValueError(f"Reference CSV must have r1/r2/r3 or reference column")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_closed_loop_from_config(
    ref_csv:           str | Path,
    controller_name:   str,
    controller_config: str | Path,
    out_csv:           str | Path,
    dt:                float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Closed-loop simulation using step9 VARX + rollout SVGP.
    Drop-in replacement for fusion.run_closed_loop.run_closed_loop_from_config.
    """
    _load_models()

    ref_abs = _load_ref(ref_csv)  # (n, 3) absolute R references
    n = len(ref_abs)

    # Rolling median trend tracker so controller references match R_tilde space
    r_hist = {i: deque([_OP_R[i]] * _DETREND_WINDOW, maxlen=_DETREND_WINDOW)
              for i in (1, 2, 3)}

    def _trend():
        return np.array([np.median(r_hist[i]) for i in (1, 2, 3)])

    ref_tilde = np.array([ref_abs[k] - _trend() for k in range(n)])

    # Relay controller is self-contained in fusion/controllers — handle it here.
    # All other controller types are delegated to the meta_arx registry.
    key = controller_name.strip().lower()
    if key == "relay":
        from fusion.controllers.relay import RelayController, load_relay_params
        params_list = load_relay_params(str(controller_config))
        controllers = [RelayController(**p) for p in params_list]
    else:
        from run_simulation.closed_loop.controller_registry import make_controllers
        controllers = make_controllers(name=controller_name,
                                       config_path=str(controller_config), dt=dt)
    _unified = not isinstance(controllers, list)
    if _unified:
        controllers.reset()
    else:
        for c in controllers: c.reset()

    y_out     = np.zeros((n + 1, 3))
    r_abs_out = np.zeros((n + 1, 3))
    u_out     = np.zeros((n,     3))
    e_out     = np.zeros((n,     3))

    varx_st = _init_varx_state()
    u_prev     = np.array([_OP_POS[i] for i in (1, 2, 3)])
    u_initial  = u_prev.copy()   # save for output at t=0
    ka         = np.full(3, _OP_KA)

    # Seed t=0 output
    r0 = _varx_step(varx_st)
    for i in (1, 2, 3):
        y_out[0, i-1]     = float(r0[i-1]) + _gp_correct(i, varx_st, float(r0[i-1]))
        r_abs_out[0, i-1] = y_out[0, i-1] + _trend()[i-1]

    print(f"[rollout] R_initial (R_tilde): {[round(y_out[0,i],4) for i in range(3)]}")

    for k in range(n):
        # 1. VARX + GP prediction
        r_tilde_raw = _varx_step(varx_st)
        y_pred = np.array([float(r_tilde_raw[i-1]) + _gp_correct(i, varx_st, float(r_tilde_raw[i-1]))
                            for i in (1, 2, 3)])
        y_out[k+1]     = y_pred
        r_abs_out[k+1] = y_pred + _trend()

        # 2. Controller
        u_new = np.zeros(3)
        if _unified:
            u_des, e_k = controllers.step(
                reference=ref_tilde[k],
                y_pred={i+1: float(y_pred[i]) for i in range(3)},
                u_prev=u_prev)
            e_out[k] = np.array(e_k)
            for i in range(3):
                du = np.clip(u_des[i] - u_prev[i], -_DPOS_MAX, _DPOS_MAX)
                u_new[i] = np.clip(u_prev[i] + du, _U_MIN, _U_MAX)
        else:
            for i in range(3):
                u_des_i, e_i = controllers[i].step(
                    reference=float(ref_tilde[k, i]),
                    y_pred=float(y_pred[i]),
                    u_prev=float(u_prev[i]))
                du = np.clip(u_des_i - u_prev[i], -_DPOS_MAX, _DPOS_MAX)
                u_new[i] = np.clip(u_prev[i] + du, _U_MIN, _U_MAX)
                e_out[k, i] = e_i

        # Anti-windup: if electrode hit a position limit, reset integrator
        # to stop the integral term from accumulating while the actuator is
        # saturated and cannot move further.
        for i in range(3):
            saturated = (u_new[i] <= _U_MIN + 1e-6) or (u_new[i] >= _U_MAX - 1e-6)
            if saturated:
                if _unified:
                    # PIDController stores per-electrode integral in _i_term[i]
                    if hasattr(controllers, "_i_term"):
                        controllers._i_term[i] = 0.0
                else:
                    if hasattr(controllers[i], "_integrator"):
                        controllers[i]._integrator = 0.0
                    elif hasattr(controllers[i], "integrator"):
                        controllers[i].integrator = 0.0

        u_out[k] = u_new
        dpos = u_new - u_prev
        u_prev = u_new

        # 3. Advance VARX state (lag-feedback separation)
        _update_varx_state(varx_st, r_tilde_raw, dpos, ka)

        # 4. Update rolling trend
        for i in (1, 2, 3):
            r_hist[i].append(float(r_abs_out[k+1, i-1]))

    out = pd.DataFrame({
        "t_s": np.arange(n+1, dtype=float) * dt,
        "y1": y_out[:, 0], "y2": y_out[:, 1], "y3": y_out[:, 2],
        "R_abs1": r_abs_out[:, 0], "R_abs2": r_abs_out[:, 1], "R_abs3": r_abs_out[:, 2],
        "r1": np.r_[np.nan, ref_tilde[:, 0]],
        "r2": np.r_[np.nan, ref_tilde[:, 1]],
        "r3": np.r_[np.nan, ref_tilde[:, 2]],
        "u1": np.r_[u_initial[0], u_out[:, 0]],
        "u2": np.r_[u_initial[1], u_out[:, 1]],
        "u3": np.r_[u_initial[2], u_out[:, 2]],
        "e1": np.r_[e_out[:, 0], np.nan],
        "e2": np.r_[e_out[:, 1], np.nan],
        "e3": np.r_[e_out[:, 2], np.nan],
        "v_transformer": _OP_V,
    })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[rollout] Done ({n} steps). Output: {out_path}")
    return out

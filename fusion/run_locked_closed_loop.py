"""
run_locked_closed_loop.py
Closed-loop simulation with a single-mover constraint.

At every timestep only the electrode furthest from its setpoint is allowed to
actuate. The other two hold their previous position.

For relay: the inactive relay wait counters do not advance.
For PID:   use a very small ki so integrator accumulation on inactive
           electrodes is negligible. Integrators are not reset between steps.

Usage
    from fusion.run_locked_closed_loop import run_locked_closed_loop_from_config

    df = run_locked_closed_loop_from_config(
        ref_csv           = "ref.csv",
        controller_name   = "relay",        # or "pid"
        controller_config = "params.csv",
        out_csv           = "results.csv",
        gp_variant        = "v9",
    )
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_META_ARX     = _PROJECT_ROOT / "meta_arx"
for _p in [str(_PROJECT_ROOT), str(_META_ARX)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion.run_closed_loop import (
    _build_sim_and_gps,
    _apply_tap_from_reference,
    _seed_dpos_lags,
    _gp_corrected_r,
    _RollingFeatures,
    _load_reference,
    _GP_VARIANT,
    _TYPICAL_POS_BY_EL,
    _TYPICAL_V,
    _TYPICAL_KA_FOR,
    _TYPICAL_REAC_FOR,
    _NEEDS_RTILDE_APPROX,
    _OOD_GATE_THRESHOLD,
)
from fusion.controllers.relay import RelayController, load_relay_params

def run_locked_closed_loop_from_config(
    ref_csv:           str | Path,
    controller_config: str | Path,
    out_csv:              str | Path,
    controller_name:      str   = "relay",
    dt:                   float = 1.0,
    gp_variant:           str   = _GP_VARIANT,
    tap_schedule:         "dict | None" = None,
    warmup_hold_steps:    int   = 0,
    gp_scale:             float = 0.1,
    init_row:             "dict | None" = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run closed-loop simulation where only the electrode furthest from its
    setpoint moves each step. Inactive relay wait counters do not advance;
    inactive PID integrators are frozen to prevent wind-up.

    tap_schedule: optional {step_index: {1: tca, 2: tcb, 3: tcc}} applied
    before each ARX prediction at the listed steps.
    """
    from run_simulation.closed_loop.closed_loop_sim import apply_actuator_limits

    key = controller_name.strip().lower()
    if key == "relay":
        params_list = load_relay_params(controller_config)
        controllers = [RelayController(**p) for p in params_list]
        _unified    = False
        for c in controllers:
            c.reset()
    elif key == "pid":
        from run_simulation.closed_loop.controllers.pid import (
            PIDController, load_pid_params_csv)
        pid_params  = load_pid_params_csv(controller_config)
        controllers = PIDController(params=pid_params, dt=dt)
        _unified    = True
        controllers.reset()
    else:
        raise ValueError(f"Locked simulation supports 'relay' or 'pid', got '{key}'")

    sim, gp_bundles, linear_models = _build_sim_and_gps(gp_variant)
    reference = _load_reference(ref_csv)   # (n, 3)

    if init_row is not None:
        for k, v in init_row.items():
            if k in sim._row.index:
                sim._row[k] = v
        print("[locked] Starting from pre-warmed initial state.")
    elif kwargs.pop("auto_tap", True):
        _apply_tap_from_reference(sim, reference)
        _seed_dpos_lags(sim, gp_variant)
    else:
        kwargs.pop("auto_tap", None)

    r_initial = None
    if gp_variant in _NEEDS_RTILDE_APPROX:
        r_initial = {
            "R":   {i: float(sim._row.get(f"El{i}_y_filt_lag1",    0.0)) for i in (1,2,3)},
            "pos": {i: float(sim._row.get(f"El{i}_pos_m_lag1",      1.04)) for i in (1,2,3)},
            "kA":  {i: float(sim._row.get(f"El{i}_kA_filt_lag1", 118.0)) for i in (1,2,3)},
        }

    n              = len(reference)
    y              = np.zeros((n + 1, 3))
    gp_var_arr     = np.zeros((n + 1, 3))
    norm_var_arr   = np.zeros((n + 1, 3))
    ind_dist_arr   = np.zeros((n + 1, 3))
    u              = np.zeros((n,     3))
    e              = np.zeros((n,     3))

    plant_cache: dict = {}
    u_prev = np.array([_TYPICAL_POS_BY_EL[i] for i in (1, 2, 3)])

    _ka_eq   = float(_TYPICAL_KA_FOR.get(gp_variant,   65.0))
    _reac_eq = float(_TYPICAL_REAC_FOR.get(gp_variant, 0.82))
    _DAMP    = 0.10

    rolling_feats = _RollingFeatures()
    rolling_feats.update(sim._row)
    rolling_feats.inject(sim._row)

    # GP baseline: subtract the correction at initial conditions so the GP
    # contributes zero at the cold-start operating point.
    _gp_biases = {1: 0.0, 2: 0.0, 3: 0.0}
    if gp_scale > 0.0 and gp_variant in ("v14", "v15", "v15b"):
        for i in (1, 2, 3):
            r_gp, _, _, _ = _gp_corrected_r(
                sim, gp_bundles, i, plant_cache, step=9999,
                linear_models=linear_models, gp_variant=gp_variant,
                r_initial=r_initial, gp_ramp_offset=0, gp_scale=1.0, gp_bias=0.0)
            r_arx = float(sim._predict_r(i))
            _gp_biases[i] = r_gp - r_arx
        print(f"[locked] GP bias calibration:  "
              f"El1={_gp_biases[1]:+.4f}  El2={_gp_biases[2]:+.4f}  "
              f"El3={_gp_biases[3]:+.4f} mOhm")

    sim._electrode = 1
    for i in (1, 2, 3):
        y[0, i-1], gp_var_arr[0, i-1], norm_var_arr[0, i-1], ind_dist_arr[0, i-1] = \
            _gp_corrected_r(sim, gp_bundles, i, plant_cache, step=0,
                            linear_models=linear_models, gp_variant=gp_variant,
                            r_initial=r_initial, gp_ramp_offset=warmup_hold_steps,
                            gp_scale=gp_scale, gp_bias=_gp_biases[i])
        plant_cache[f"r{i}"]      = y[0, i-1]
        plant_cache[f"r{i}_lag2"] = y[0, i-1]
    sim._electrode = 1

    # Pre-load tap lookup once for dynamic tap updates
    _tap_lookup_path = _HERE / "models" / "tap_lookup.json"
    _tap_lkp = None
    if _tap_lookup_path.exists():
        from fusion.training.tap_lookup import TapLookup
        _tap_lkp = TapLookup.load(_tap_lookup_path)
    _tc_col = {1: "TCA", 2: "TCB", 3: "TCC"}
    _prev_taps = {i: float(sim._row.get(_tc_col[i], 15.0)) for i in (1, 2, 3)}

    for k in range(n):
        plant_cache["step"] = k
        rolling_feats.inject(sim._row)

        # Update tap changer if the optimal tap for the current reference differs
        # from the current setting.  Tap changers are discrete and slow in the real
        # furnace; here we allow one tap step per simulation step for simplicity.
        if _tap_lkp is not None:
            for _i in (1, 2, 3):
                _r_ref  = float(reference[k, _i - 1])
                _best   = _tap_lkp.get_tap(_i, _r_ref)
                _col    = _tc_col[_i]
                if _col in sim._row.index and _best != _prev_taps[_i]:
                    sim._row[_col] = _best
                    _prev_taps[_i] = _best

        # 1. Predict R for all electrodes
        y_pred:     dict = {}
        gp_var_k:   dict = {}
        norm_var_k: dict = {}
        ind_dist_k: dict = {}
        for i in (1, 2, 3):
            y_pred[i], gp_var_k[i], norm_var_k[i], ind_dist_k[i] = _gp_corrected_r(
                sim, gp_bundles, i, plant_cache, step=k,
                linear_models=linear_models, gp_variant=gp_variant,
                r_initial=r_initial, gp_ramp_offset=warmup_hold_steps,
                gp_scale=gp_scale, gp_bias=_gp_biases[i])
        sim._electrode = 1

        # 2. OOD gate
        mean_nv  = float(np.mean([norm_var_k[i] for i in (1, 2, 3)]))
        ood_hold = (mean_nv > _OOD_GATE_THRESHOLD) and (gp_variant not in _NEEDS_RTILDE_APPROX)

        # 3. Single-mover constraint: find electrode furthest from setpoint.
        # If no electrode exceeds the deadband all hold. Same as relay behaviour.
        abs_errors = {i: abs(float(reference[k, i-1]) - float(y_pred[i]))
                      for i in (1, 2, 3)}
        _deadband  = kwargs.get("deadband", 0.07)

        # During warm-up hold, all electrodes freeze regardless of error.
        # This lets the GP predictions settle from the cold-start state before
        # any controller fires.
        if k < warmup_hold_steps:
            active_el = None
        else:
            active_el  = max(abs_errors, key=abs_errors.__getitem__)
            if abs_errors[active_el] < _deadband:
                active_el = None   # all hold

        u_new: dict = {}
        if ood_hold or active_el is None:
            for i in (1, 2, 3):
                u_new[i]  = float(u_prev[i-1])
                u[k, i-1] = u_new[i]
                e[k, i-1] = reference[k, i-1] - y_pred[i]
        elif _unified:
            # PID unified: call once to get all desired positions + errors.
            # Freeze integrators for inactive electrodes to prevent wind-up
            # while those electrodes are holding position.
            prev_i_term = controllers._i_term.copy()
            u_des, e_k = controllers.step(
                reference=reference[k], y_pred=y_pred, u_prev=u_prev)
            for i in (1, 2, 3):
                if i != active_el:
                    controllers._i_term[i - 1] = prev_i_term[i - 1]
            e[k] = e_k
            for i in (1, 2, 3):
                if i == active_el:
                    raw = float(u_des[i-1][0] if hasattr(u_des[i-1], '__len__')
                                else u_des[i-1])
                    u_new[i] = apply_actuator_limits(raw, u_prev[i-1])
                else:
                    u_new[i] = float(u_prev[i-1])
            u[k] = [u_new[i] for i in (1, 2, 3)]
        else:
            # Relay per-electrode: only call step for the active electrode
            for i in (1, 2, 3):
                if active_el is not None and i == active_el:
                    u_des, e_i = controllers[i-1].step(
                        reference=float(reference[k, i-1]),
                        y_pred=float(y_pred[i]),
                        u_prev=float(u_prev[i-1]),
                    )
                    u_ki = apply_actuator_limits(u_des, u_prev[i-1])
                else:
                    u_ki = float(u_prev[i-1])
                    e_i  = float(reference[k, i-1]) - float(y_pred[i])
                u_new[i]  = u_ki
                u[k, i-1] = u_ki
                e[k, i-1] = e_i

        # 4. Advance simulator
        y_arx_vec = {i: sim._predict_r(i) for i in (1, 2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec=u_new, y_new_vec=y_arx_vec)

        for _i in (1, 2, 3):
            for _lag in (1, 2, 3):
                _kc = f"El{_i}_kA_filt_lag{_lag}"
                _rc = f"El{_i}_CalcReac_filt_lag{_lag}"
                if _kc in sim._row.index:
                    sim._row[_kc] = (1 - _DAMP) * _ka_eq   + _DAMP * float(sim._row[_kc])
                if _rc in sim._row.index:
                    sim._row[_rc] = (1 - _DAMP) * _reac_eq + _DAMP * float(sim._row[_rc])
        _vc = "RMS_V_transformer_filt_lag1"
        if _vc in sim._row.index:
            sim._row[_vc] = (1 - _DAMP) * _TYPICAL_V + _DAMP * float(sim._row[_vc])

        rolling_feats.update(sim._row, y_override=y_pred)

        for i in (1, 2, 3):
            y[k+1, i-1]             = y_pred[i]
            gp_var_arr[k+1, i-1]   = gp_var_k[i]
            norm_var_arr[k+1, i-1] = norm_var_k[i]
            ind_dist_arr[k+1, i-1] = ind_dist_k[i]
            plant_cache[f"r{i}_lag2"] = plant_cache.get(f"r{i}", y_arx_vec[i])
            plant_cache[f"r{i}"]      = y_arx_vec[i]

        u_prev = np.array([u_new[i] for i in (1, 2, 3)])

    t   = np.arange(n + 1) * dt
    u0  = np.array([_TYPICAL_POS_BY_EL[i] for i in (1, 2, 3)])
    ref = np.vstack([np.full((1, 3), np.nan), reference])

    out = pd.DataFrame({
        "t_s": t,
        "y1": y[:, 0],  "y2": y[:, 1],  "y3": y[:, 2],
        "r1": ref[:, 0], "r2": ref[:, 1], "r3": ref[:, 2],
        "u1": np.r_[u0[0], u[:, 0]],
        "u2": np.r_[u0[1], u[:, 1]],
        "u3": np.r_[u0[2], u[:, 2]],
        "e1": np.r_[np.nan, e[:, 0]],
        "e2": np.r_[np.nan, e[:, 1]],
        "e3": np.r_[np.nan, e[:, 2]],
        "v_transformer": _TYPICAL_V,
        "active_el": np.r_[np.nan, [max({i: abs(float(reference[k, i-1]) - y[k+1, i-1]) for i in (1,2,3)}, key=lambda x: {i: abs(float(reference[k, i-1]) - y[k+1, i-1]) for i in (1,2,3)}[x]) for k in range(n)]],
    })

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[locked] Simulation done ({n} steps). Output: {out_csv}")
    out._sim_final_row = dict(sim._row)   # attach final state for chaining
    return out

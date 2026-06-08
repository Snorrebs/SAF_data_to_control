"""
fusion/mpc/model.py
Builds the do-mpc discrete-time model for the 3-electrode SAF system
using the joint V9 or V15 ARX bundle already in fusion/models/.

State variables (55 total):
    Per electrode i in {1, 2, 3} (17 x 3 = 51):
        pos{i}           electrode position (m)
        dp{i}_1..5       dpos lags 1-5
        r{i}_1..5        R lags 1-5
        ka{i}_1..3       kA lags 1-3
        rx{i}_1..3       CalcReac lags 1-3
    Shared (1):
        V_1              transformer RMS voltage lag 1
    Integrators (3):
        int{i}           accumulated R tracking error for offset-free control

Control inputs (3): u1, u2, u3 (absolute electrode position commands, m)

Time-varying parameters: tca, tcb, tcc, rstd_rx1/2/3, rstd_r1/2/3,
                         step_in_win, r_nom, y_sim1/2/3
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent


_COL_TO_STATE: dict = {}
for _i in (1, 2, 3):
    _COL_TO_STATE.update({
        f"El{_i}_pos_m_lag1":            f"pos{_i}",
        f"El{_i}_dpos_mps_filt_lag1":    f"dp{_i}_1",
        f"El{_i}_dpos_mps_filt_lag2":    f"dp{_i}_2",
        f"El{_i}_dpos_mps_filt_lag3":    f"dp{_i}_3",
        f"El{_i}_dpos_mps_filt_lag4":    f"dp{_i}_4",
        f"El{_i}_dpos_mps_filt_lag5":    f"dp{_i}_5",
        f"El{_i}_y_filt_lag1":           f"r{_i}_1",
        f"El{_i}_y_filt_lag2":           f"r{_i}_2",
        f"El{_i}_y_filt_lag3":           f"r{_i}_3",
        f"El{_i}_y_filt_lag4":           f"r{_i}_4",
        f"El{_i}_y_filt_lag5":           f"r{_i}_5",
        f"El{_i}_kA_filt_lag1":          f"ka{_i}_1",
        f"El{_i}_kA_filt_lag2":          f"ka{_i}_2",
        f"El{_i}_kA_filt_lag3":          f"ka{_i}_3",
        f"El{_i}_CalcReac_filt_lag1":    f"rx{_i}_1",
        f"El{_i}_CalcReac_filt_lag2":    f"rx{_i}_2",
        f"El{_i}_CalcReac_filt_lag3":    f"rx{_i}_3",
    })
_COL_TO_STATE["RMS_V_transformer_filt_lag1"] = "V_1"

_COL_TO_TVP: dict = {
    "TCA": "tca", "TCB": "tcb", "TCC": "tcc",
    "step_in_window": "step_in_win",
    "TCA_diff": "tca_diff",
}
for _i in (1, 2, 3):
    _COL_TO_TVP[f"El{_i}_rolling_std_CalcReac_30s"] = f"rstd_rx{_i}"
    _COL_TO_TVP[f"El{_i}_rolling_std_R_30s"]        = f"rstd_r{_i}"
    _COL_TO_TVP[f"El{_i}_R_imbalance"]              = f"r_imb{_i}"
    _COL_TO_TVP["y_sim"]    = f"y_sim{_i}"
    _COL_TO_TVP["y_sim_sq"] = f"y_sim_sq{_i}"


def load_joint_bundle(gp_variant: str = "v9") -> dict:
    """Load a joint ARX bundle from fusion/models/."""
    import joblib
    from fusion.run_closed_loop import _ARX_FOR_VARIANT
    arx_name = _ARX_FOR_VARIANT.get(gp_variant, "arx_joint_v9.joblib")
    path = _HERE / "models" / arx_name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def build_mpc_model(
    joint_bundle: dict,
    gp_variant:   str = "v9",
) -> "do_mpc.model.Model":
    """Build a do-mpc discrete-time model from a joint V9 or V15 ARX bundle."""
    import do_mpc
    import casadi as ca
    from fusion.mpc.arx_casadi import joint_output_expr, delta_arx_r_expr, ca_clip

    is_delta = type(joint_bundle["model"]).__name__ == "DeltaARXWrapper"

    model = do_mpc.model.Model("discrete")

    states: dict = {}
    for i in (1, 2, 3):
        for name in [f"pos{i}",
                     f"dp{i}_1", f"dp{i}_2", f"dp{i}_3", f"dp{i}_4", f"dp{i}_5",
                     f"r{i}_1",  f"r{i}_2",  f"r{i}_3",  f"r{i}_4",  f"r{i}_5",
                     f"ka{i}_1", f"ka{i}_2", f"ka{i}_3",
                     f"rx{i}_1", f"rx{i}_2", f"rx{i}_3"]:
            states[name] = model.set_variable("_x", name)
    states["V_1"] = model.set_variable("_x", "V_1")
    for i in (1, 2, 3):
        states[f"int{i}"] = model.set_variable("_x", f"int{i}")

    _u = {i: model.set_variable("_u", f"u{i}") for i in (1, 2, 3)}

    tvp: dict = {}
    for name in ["tca", "tcb", "tcc", "step_in_win", "r_nom", "tca_diff",
                 "rstd_rx1", "rstd_rx2", "rstd_rx3",
                 "rstd_r1",  "rstd_r2",  "rstd_r3",
                 "y_sim1",   "y_sim2",   "y_sim3",
                 "y_sim_sq1","y_sim_sq2","y_sim_sq3",
                 "r_imb1",   "r_imb2",   "r_imb3"]:
        tvp[name] = model.set_variable("_tvp", name)

    all_syms = {**states, **tvp}

    def col_vals_for_bundle(b: dict) -> dict:
        cv: dict = {}
        for c in b["X_cols"]:
            if c in _COL_TO_STATE and _COL_TO_STATE[c] in all_syms:
                cv[c] = all_syms[_COL_TO_STATE[c]]
            elif c in _COL_TO_TVP and _COL_TO_TVP[c] in all_syms:
                cv[c] = all_syms[_COL_TO_TVP[c]]
            else:
                cv[c] = ca.DM(0.0)
        return cv

    y_index = joint_bundle.get("y_index", {
        "R":    {1: 0, 2: 1, 3: 2},
        "kA":   {1: 3, 2: 4, 3: 5},
        "reac": {1: 6, 2: 7, 3: 8},
        "v":    9,
    })

    cv = col_vals_for_bundle(joint_bundle)

    # y_arx is computed before advance_multi in the simulation loop, so the ARX
    # sees dpos(t-1) not dpos(t). Matching that here keeps model and plant consistent;
    # u(t) still reaches the gradient via lag propagation at horizon step 2+.
    dpos_cur = {i: _u[i] - states[f"pos{i}"] for i in (1, 2, 3)}

    rhs: dict = {}

    for i in (1, 2, 3):
        u_i      = _u[i]
        dpos_new = dpos_cur[i]

        if is_delta:
            r_arx = delta_arx_r_expr(cv, joint_bundle, i)
        else:
            r_arx = joint_output_expr(cv, joint_bundle, y_index["R"][i])
        r_arx = ca_clip(r_arx, 0.5, 2.5)

        ka_next = states[f"ka{i}_1"]
        rx_next = states[f"rx{i}_1"]

        model.set_expression(f"r{i}_pred",  ca.reshape(r_arx,  1, 1))
        pf_safe = ca.sqrt(ca.fmax(r_arx**2 + rx_next**2, ca.DM(1e-12)))
        model.set_expression(f"pf{i}_pred", ca.reshape(r_arx / pf_safe, 1, 1))

        rhs[f"pos{i}"]  = u_i
        rhs[f"dp{i}_5"] = states[f"dp{i}_4"]
        rhs[f"dp{i}_4"] = states[f"dp{i}_3"]
        rhs[f"dp{i}_3"] = states[f"dp{i}_2"]
        rhs[f"dp{i}_2"] = states[f"dp{i}_1"]
        rhs[f"dp{i}_1"] = dpos_new
        rhs[f"r{i}_5"]  = states[f"r{i}_4"]
        rhs[f"r{i}_4"]  = states[f"r{i}_3"]
        rhs[f"r{i}_3"]  = states[f"r{i}_2"]
        rhs[f"r{i}_2"]  = states[f"r{i}_1"]
        rhs[f"r{i}_1"]  = r_arx
        rhs[f"ka{i}_3"] = states[f"ka{i}_2"]
        rhs[f"ka{i}_2"] = states[f"ka{i}_1"]
        rhs[f"ka{i}_1"] = ka_next
        rhs[f"rx{i}_3"] = states[f"rx{i}_2"]
        rhs[f"rx{i}_2"] = states[f"rx{i}_1"]
        rhs[f"rx{i}_1"] = rx_next
        rhs[f"int{i}"]  = states[f"int{i}"] + (r_arx - tvp["r_nom"])

    rhs["V_1"] = states["V_1"]

    for name, expr in rhs.items():
        model.set_rhs(name, expr)

    model.setup()
    return model

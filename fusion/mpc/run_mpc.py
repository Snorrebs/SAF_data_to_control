"""
fusion/mpc/run_mpc.py
NMPC closed-loop simulation using the joint ARX + GP plant.

Drop-in replacement for run_closed_loop.py using NMPC instead of PID/relay.

Usage in VRFT v5.py - change one import line:

    from fusion.mpc.run_mpc import run_closed_loop_from_config

The function signature is compatible with run_closed_loop_from_config.

Plant:      SaFSimulator with joint V9 or V15 ARX + optional GP correction
Controller: do-mpc NMPC with IPOPT solver, horizon H=5
Objective:  minimise (R - r_nom)^2 per electrode
Constraint: soft R band [r_nom - r_tol, r_nom + r_tol]
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

_HERE         = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _HERE.parent
_META_ARX     = _PROJECT_ROOT / "meta_arx"

for _p in [str(_PROJECT_ROOT), str(_META_ARX)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _apply_limits(
    u_des:  float,
    u_prev: float,
    du_max: float = 0.05,   # safety stop only; do-mpc enforces 0.01 via nl_cons
    u_min:  float = 0.0,
    u_max:  float = 2.0,
) -> float:
    du = float(np.clip(u_des - u_prev, -du_max, du_max))
    return float(np.clip(u_prev + du, u_min, u_max))


def _sim_to_state(sim, int_vals: "dict | None" = None) -> np.ndarray:
    """Extract SaFSimulator._row into the do-mpc state vector (55 states)."""
    row = sim._row
    int_vals = int_vals or {1: 0.0, 2: 0.0, 3: 0.0}
    vals = []
    for i in (1, 2, 3):
        vals.extend([
            float(row.get(f"El{i}_pos_m_lag1",         1.04)),
            float(row.get(f"El{i}_dpos_mps_filt_lag1", 0.0)),
            float(row.get(f"El{i}_dpos_mps_filt_lag2", 0.0)),
            float(row.get(f"El{i}_dpos_mps_filt_lag3", 0.0)),
            float(row.get(f"El{i}_dpos_mps_filt_lag4", 0.0)),
            float(row.get(f"El{i}_dpos_mps_filt_lag5", 0.0)),
            float(row.get(f"El{i}_y_filt_lag1",        1.076)),
            float(row.get(f"El{i}_y_filt_lag2",        1.076)),
            float(row.get(f"El{i}_y_filt_lag3",        1.076)),
            float(row.get(f"El{i}_y_filt_lag4",        1.076)),
            float(row.get(f"El{i}_y_filt_lag5",        1.076)),
            float(row.get(f"El{i}_kA_filt_lag1",      118.0)),
            float(row.get(f"El{i}_kA_filt_lag2",      118.0)),
            float(row.get(f"El{i}_kA_filt_lag3",      118.0)),
            float(row.get(f"El{i}_CalcReac_filt_lag1", 0.88)),
            float(row.get(f"El{i}_CalcReac_filt_lag2", 0.88)),
            float(row.get(f"El{i}_CalcReac_filt_lag3", 0.88)),
        ])
    vals.append(float(row.get("RMS_V_transformer_filt_lag1", 165.0)))
    for i in (1, 2, 3):
        vals.append(float(int_vals.get(i, 0.0)))
    return np.array(vals, dtype=np.float64).reshape(-1, 1)


def run_closed_loop_from_config(
    ref_csv:           "str | Path",
    controller_name:   str,
    controller_config: "str | Path",
    out_csv:           "str | Path",
    dt:                float = 1.0,
    *,
    H:                  int   = 5,
    n:                  "int | None" = None,
    r_nom:              "float | None" = None,
    r_tol:              float = 0.07,
    lam_u:              float = 1.0,
    lam_int:            float = 0.0,
    objective:          str   = "r",
    constraint_penalty: float = 1e4,
    gp_variant:         str   = "v9",
    gp_scale:           float = 1.0,
    auto_tap:           bool  = True,
    locked:             bool  = True,
    verbose:            bool  = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Run a locked single-mover NMPC simulation using the joint ARX plant.

    With locked=True (default) only the electrode furthest from r_nom moves
    each step, matching the V15 training distribution.
    """
    import casadi as ca
    import do_mpc

    from fusion.run_closed_loop import _build_sim_and_gps, _apply_tap_from_reference
    from fusion.mpc.model       import build_mpc_model, load_joint_bundle
    from fusion.mpc.controller  import build_mpc_controller, update_tvp

    ref_df    = pd.read_csv(ref_csv)
    if "reference" in ref_df.columns:
        ref_1d    = ref_df["reference"].to_numpy(dtype=float)
        reference = np.column_stack([ref_1d, ref_1d, ref_1d])
    else:
        reference = ref_df[["r1", "r2", "r3"]].to_numpy(dtype=float)
    if n is None:
        n = len(reference)

    print(f"[mpc] Building plant  gp_variant={gp_variant}  gp_scale={gp_scale}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim, gp_bundles, linear_models = _build_sim_and_gps(gp_variant)

    if auto_tap:
        _apply_tap_from_reference(sim, reference)

    r_nom_actual = r_nom if r_nom is not None else float(
        sim._row.get("El1_y_filt_lag1", 1.076))

    print("[mpc] Building do-mpc model (ARX prediction horizon) ...")
    joint_bundle = load_joint_bundle(gp_variant)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mpc_model = build_mpc_model(joint_bundle, gp_variant=gp_variant)

    mpc = build_mpc_controller(
        mpc_model,
        H                  = H,
        r_nom              = r_nom_actual,
        r_tol              = r_tol,
        lam_u              = lam_u,
        lam_int            = lam_int,
        objective          = objective,
        constraint_penalty = constraint_penalty,
        verbose            = verbose,
    )
    print(f"[mpc] Ready  H={H}  R_nom={r_nom_actual:.3f}  "
          f"band=[{r_nom_actual-r_tol:.3f},{r_nom_actual+r_tol:.3f}]  "
          f"penalty={constraint_penalty:.0e}")

    int_vals = {1: 0.0, 2: 0.0, 3: 0.0}
    x0 = _sim_to_state(sim, int_vals)
    mpc.x0 = x0
    mpc.set_initial_guess()

    _ROLL = 30
    roll_reac = {j: deque([0.0] * _ROLL, maxlen=_ROLL) for j in (1, 2, 3)}
    roll_r    = {j: deque([0.0] * _ROLL, maxlen=_ROLL) for j in (1, 2, 3)}

    def _rstd(buf): return float(np.std(buf)) if len(buf) > 1 else 0.0

    def _cur_r(j):
        from fusion.run_closed_loop import _gp_corrected_r
        r_gp, _, _, _ = _gp_corrected_r(
            sim, gp_bundles, j, {}, step=0,
            gp_variant=gp_variant, gp_scale=gp_scale)
        return r_gp

    R_out  = {j: np.zeros(n + 1) for j in (1, 2, 3)}
    U_out  = {j: np.zeros(n)     for j in (1, 2, 3)}
    u_prev = {j: float(sim._row.get(f"El{j}_pos_m_lag1", 1.04)) for j in (1, 2, 3)}

    for j in (1, 2, 3):
        R_out[j][0] = _cur_r(j)

    for k in range(n):
        if k % 20 == 0:
            r_str = "  ".join(f"R{j}={R_out[j][k]:.4f}" for j in (1, 2, 3))
            print(f"[mpc] step {k:4d}/{n}  {r_str}", flush=True)

        if lam_int > 0:
            for j in (1, 2, 3):
                int_vals[j] += R_out[j][k] - r_nom_actual

        y_sim = {j: float(sim._row.get(f"El{j}_y_filt_lag1", 1.0)) for j in (1, 2, 3)}

        update_tvp(
            mpc,
            tca         = float(sim._row.get("TCA", 0.0)),
            tcb         = float(sim._row.get("TCB", 0.0)),
            tcc         = float(sim._row.get("TCC", 0.0)),
            rstd_rx     = {j: _rstd(roll_reac[j]) for j in (1, 2, 3)},
            rstd_r      = {j: _rstd(roll_r[j])    for j in (1, 2, 3)},
            step_in_win = float(min(k, H - 1)),
            r_nom       = r_nom_actual,
            y_sim       = y_sim,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_k = _sim_to_state(sim, int_vals)
            mpc.x0 = x_k
            u_mpc  = mpc.make_step(x_k)

        if locked:
            # Locked single-mover: only the electrode furthest from r_nom moves.
            active_el = max((1, 2, 3),
                            key=lambda j: abs(R_out[j][k] - r_nom_actual))
            u_new = {}
            for j in (1, 2, 3):
                if j == active_el:
                    u_new[j] = _apply_limits(float(u_mpc.flat[j - 1]), u_prev[j])
                else:
                    u_new[j] = u_prev[j]
        else:
            u_new = {j: _apply_limits(float(u_mpc.flat[j - 1]), u_prev[j])
                     for j in (1, 2, 3)}

        y_arx = {j: sim._predict_r(j) for j in (1, 2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec=u_new, y_new_vec=y_arx)

        for j in (1, 2, 3):
            roll_reac[j].append(float(sim._row.get(f"El{j}_CalcReac_filt_lag1", 0.88)))
            roll_r[j].append(float(sim._row.get(f"El{j}_y_filt_lag1", 1.0)))
            R_out[j][k + 1] = _cur_r(j)
            U_out[j][k]     = u_new[j]
        u_prev = u_new

    t_arr = np.arange(n + 1) * dt
    out_df = pd.DataFrame({"t_s": t_arr})
    for j in (1, 2, 3):
        out_df[f"y{j}"] = R_out[j]
        out_df[f"u{j}"] = np.r_[u_prev[j], U_out[j]]
    out_df["r_nom"] = r_nom_actual
    out_df["r_tol"] = r_tol

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[mpc] Saved: {out_csv}")
    return out_df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="3-electrode NMPC for SAF R regulation")
    ap.add_argument("--out",         default="fusion/results/mpc/mpc_out.csv")
    ap.add_argument("--steps",       type=int,   default=300)
    ap.add_argument("--r-nom",       type=float, default=None)
    ap.add_argument("--r-tol",       type=float, default=0.07)
    ap.add_argument("--H",           type=int,   default=5)
    ap.add_argument("--lam-u",       type=float, default=1.0)
    ap.add_argument("--lam-int",     type=float, default=0.0)
    ap.add_argument("--gp-variant",  default="v9")
    ap.add_argument("--gp-scale",    type=float, default=1.0)
    ap.add_argument("--penalty",     type=float, default=1e4)
    ap.add_argument("--no-auto-tap", action="store_true")
    ap.add_argument("--no-locked",   action="store_true",
                    help="disable locked single-mover (default: locked on)")
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    run_closed_loop_from_config(
        ref_csv           = "meta_arx/run_simulation/init_data/reference_mpc.csv",
        controller_name   = "nmpc",
        controller_config = "meta_arx/run_simulation/init_data/reference_mpc.csv",
        out_csv           = args.out,
        n                 = args.steps,
        H                 = args.H,
        r_nom             = args.r_nom,
        r_tol             = args.r_tol,
        lam_u             = args.lam_u,
        lam_int           = args.lam_int,
        gp_variant        = args.gp_variant,
        gp_scale          = args.gp_scale,
        constraint_penalty= args.penalty,
        auto_tap          = not args.no_auto_tap,
        locked            = not args.no_locked,
        verbose           = args.verbose,
    )

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_simulation.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation.closed_loop.closed_loop_sim import run_mpc_closed_loop
from run_simulation.closed_loop.controller_registry import make_mpc_controller
from run_simulation.closed_loop.reference_converter import ReferenceConverter
from run_simulation.scripts.run_closed_loop import make_ka_exog


MODEL_PATH = Path("run_simulation/models/step9_2sls_varx_model.joblib")
HIST_CSV   = Path("run_simulation/init_data/step9_2sls_varx_init.csv")


def load_reference_csv(path: str | Path) -> np.ndarray:
    """Load reference signal from CSV (absolute mOhm).

    Accepts:
      columns r1, r2, r3                                                     -> (n, 3)
      columns El1_Resistance_mOhm, El2_Resistance_mOhm, El3_Resistance_mOhm -> (n, 3)
      single column r or reference                                            -> (n,) broadcast
    """
    df = pd.read_csv(path)

    if all(c in df.columns for c in ["r1", "r2", "r3"]):
        return df[["r1", "r2", "r3"]].to_numpy(dtype=float)

    if all(c in df.columns for c in [
            "El1_Resistance_mOhm", "El2_Resistance_mOhm", "El3_Resistance_mOhm"]):
        return df[["El1_Resistance_mOhm",
                   "El2_Resistance_mOhm",
                   "El3_Resistance_mOhm"]].to_numpy(dtype=float)

    for col in ("r", "reference"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)

    raise ValueError(
        f"Reference CSV '{path}' must contain ['r1','r2','r3'], "
        "['El1_Resistance_mOhm','El2_Resistance_mOhm','El3_Resistance_mOhm'], "
        "or a scalar column 'r'/'reference'."
    )


def run_mpc_from_config(
    ref_csv:        str | Path,
    mpc_config:     str | Path,
    out_csv:        str | Path,
    dt:             float = 1.0,
    op_point:       list[float] | None = None,
    ka_disturbance: bool = True,
    plotting:       bool = False,
) -> pd.DataFrame:
    """Run closed-loop MPC simulation against the 2SLS VARX model.

    Args:
        ref_csv:        path to reference CSV (absolute mOhm)
        mpc_config:     path to MPC_params CSV
        out_csv:        path to write simulation results
        dt:             sample time [s]
        op_point:       operating point [mOhm] per electrode for trend
                        initialisation. If None, uses Step 1 means
                        [0.95, 1.03, 1.07].
        ka_disturbance: if True (default), add Perlin noise to kA.
                        if False, hold kA frozen at Step 1 means —
                        useful for isolating controller behaviour from
                        disturbance effects.
        plotting:       if True, generate plots after simulation
    """
    bundle = load_arx_bundle(str(MODEL_PATH))
    state  = load_initial_state(str(HIST_CSV), bundle)

    # ── Reference conversion: absolute mOhm -> R-tilde space ─────────────────
    if op_point is None:
        op_point = [0.95, 1.03, 1.07]   # Step 1 electrode resistance means

    converter = ReferenceConverter.from_operating_point(
        op_point=op_point,
        window_s=1800,
    )

    reference_abs   = load_reference_csv(ref_csv)
    trend_at_t0     = converter.current_trend()
    reference_tilde = converter.convert_trajectory(reference_abs, freeze_trend=True)

    print(f"  Trend at t=0:       {trend_at_t0}")
    print(f"  Reference (abs)[0]: {reference_abs[0] if reference_abs.ndim > 1 else reference_abs[0]}")
    print(f"  Reference (R~)[0]:  {reference_tilde[0]}")

    n = len(reference_tilde)

    # ── kA exogenous trajectory ───────────────────────────────────────────────
    print(f"  kA disturbance: {'Perlin noise' if ka_disturbance else 'frozen at means'}")
    exog_traj = make_ka_exog(n, disturbed=ka_disturbance)

    # ── MPC ───────────────────────────────────────────────────────────────────
    mpc = make_mpc_controller(str(mpc_config), bundle)

    y, du, e = run_mpc_closed_loop(
        model     = bundle,
        state     = state,
        reference = reference_tilde,
        mpc       = mpc,
        exog_traj = exog_traj,
    )

    # ── Output DataFrame ──────────────────────────────────────────────────────
    if reference_tilde.ndim == 1:
        reference_tilde = np.repeat(reference_tilde.reshape(-1, 1), 3, axis=1)

    # Cumulative movement (dead-reckoning from zero — informational only)
    u_cumsum = np.cumsum(np.vstack([np.zeros((1, 3)), du]), axis=0)

    out = pd.DataFrame({
        "t_s":   np.arange(len(y), dtype=float) * dt,
        # Outputs (R-tilde space)
        "y1":    y[:, 0],
        "y2":    y[:, 1],
        "y3":    y[:, 2],
        # Reference (R-tilde space)
        "r1":    np.r_[np.nan, reference_tilde[:, 0]],
        "r2":    np.r_[np.nan, reference_tilde[:, 1]],
        "r3":    np.r_[np.nan, reference_tilde[:, 2]],
        # Control: delta-u (movements)
        "du1":   np.r_[0.0, du[:, 0]],
        "du2":   np.r_[0.0, du[:, 1]],
        "du3":   np.r_[0.0, du[:, 2]],
        # Cumulative position estimate (dead-reckoning, informational only)
        "u1_cumsum": u_cumsum[:, 0],
        "u2_cumsum": u_cumsum[:, 1],
        "u3_cumsum": u_cumsum[:, 2],
        # Errors
        "e1":    np.r_[e[:, 0], np.nan],
        "e2":    np.r_[e[:, 1], np.nan],
        "e3":    np.r_[e[:, 2], np.nan],
        # kA disturbance trajectories (frozen or Perlin)
        "kA1":   np.r_[np.nan, exog_traj["kA1"]],
        "kA2":   np.r_[np.nan, exog_traj["kA2"]],
        "kA3":   np.r_[np.nan, exog_traj["kA3"]],
    })

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    if plotting:
        from run_simulation.scripts.plotting import main as plot_main
        plot_main(path=out_csv)

    return out


# python -m run_simulation.scripts.run_mpc
if __name__ == "__main__":
    run_mpc_from_config(
        ref_csv        = "run_simulation/init_data/reference_res_2.csv",
        mpc_config     = "run_simulation/init_data/MPC_params.csv",
        out_csv        = "run_simulation/history/closed_loop_mpc.csv",
        dt             = 1.0,
        ka_disturbance = True,   # set False to freeze kA at means
        plotting       = True,
    )
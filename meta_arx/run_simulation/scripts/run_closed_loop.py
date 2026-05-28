from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_simulation.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation.closed_loop.closed_loop_sim import run_closed_loop
from run_simulation.closed_loop.controller_registry import make_controllers
from run_simulation.closed_loop.reference_converter import ReferenceConverter


MODEL_PATH = Path("run_simulation/models/step9_n10_2sls_varx_model.joblib")
HIST_CSV   = Path("run_simulation/init_data/step9_n10_2sls_varx_init.csv")

# Perlin noise parameters for kA disturbance
# Calibrated to Step 1 data: mean ~126-130 kA, std ~7 kA
KA_MEANS     = np.array([129.7, 126.5, 123.6])   # per-electrode means [kA]
KA_AMP       = 5.0                                 # noise amplitude [kA]
PERLIN_SCALE = 0.005                               # controls rate of variation
PERLIN_OCT   = 4


def make_ka_exog(n: int, disturbed: bool, warmup_len: int = 300) -> dict[str, np.ndarray]:
    """Build kA exogenous trajectory for all three electrodes.

    Args:
        n:           number of simulation steps
        disturbed:   if True, add Perlin noise around KA_MEANS.
                     if False, hold each electrode at its mean (frozen).
        warmup_len:  offset into the Perlin sequence so the simulation
                     does not start at t=0 of the noise (matches run_mpc.py).

    Returns:
        Dict with keys "kA1", "kA2", "kA3", each an array of length n.
    """
    exog: dict[str, np.ndarray] = {}
    if disturbed:
        try:
            from noise import pnoise1
        except ImportError as exc:
            raise ImportError(
                "The 'noise' package is required for Perlin kA disturbance. "
                "Install with: pip install noise"
            ) from exc
        for i in range(3):
            exog[f"kA{i+1}"] = np.array([
                KA_MEANS[i] + KA_AMP * pnoise1(
                    (warmup_len + k) * PERLIN_SCALE + i * 100,
                    octaves=PERLIN_OCT,
                )
                for k in range(n)
            ])
    else:
        for i in range(3):
            exog[f"kA{i+1}"] = np.full(n, KA_MEANS[i])
    return exog


def load_reference_csv(path: str | Path) -> np.ndarray:
    """Load reference signal from CSV.

    Accepts:
    * columns ``r1``, ``r2``, ``r3``                                          -> shape (n, 3)
    * columns ``El1_Resistance_mOhm``, ``El2_Resistance_mOhm``, ``El3_Resistance_mOhm`` -> shape (n, 3)
    * single column ``r`` or ``reference``                                    -> shape (n,) broadcast
    """
    df = pd.read_csv(path)

    if all(c in df.columns for c in ["r1", "r2", "r3"]):
        return df[["r1", "r2", "r3"]].to_numpy(dtype=float)

    if all(c in df.columns for c in ["El1_Resistance_mOhm", "El2_Resistance_mOhm", "El3_Resistance_mOhm"]):
        return df[["El1_Resistance_mOhm", "El2_Resistance_mOhm", "El3_Resistance_mOhm"]].to_numpy(dtype=float)

    for col in ("r", "reference"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)

    raise ValueError(
        f"Reference CSV '{path}' must contain ['r1','r2','r3'], "
        "['El1_Resistance_mOhm','El2_Resistance_mOhm','El3_Resistance_mOhm'], "
        "or a scalar column 'r'/'reference'."
    )


def run_closed_loop_from_config(
    ref_csv:           str | Path,
    controller_name:   str,
    controller_config: str | Path,
    out_csv:           str | Path,
    dt:                float = 1.0,
    op_point:          list[float] | None = None,
    ka_disturbance:    bool = True,
    plotting:          bool = False,
) -> pd.DataFrame:
    """Run closed-loop PID simulation against the 2SLS VARX model.

    Args:
        ref_csv:           path to reference CSV (absolute mOhm)
        controller_name:   name passed to make_controllers() (e.g. "pid")
        controller_config: path to controller params CSV
        out_csv:           path to write simulation results
        dt:                sample time [s]
        op_point:          operating point [mOhm] per electrode for trend
                           initialisation. If None, uses Step 1 means
                           [0.95, 1.03, 1.07].
        ka_disturbance:    if True (default), add Perlin noise to kA.
                           if False, hold kA frozen at Step 1 means —
                           useful for isolating controller behaviour from
                           disturbance effects.
        plotting:          if True, generate plots after simulation
    """
    bundle = load_arx_bundle(str(MODEL_PATH))
    state  = load_initial_state(str(HIST_CSV), bundle)

    # ── Reference conversion: absolute mOhm -> R-tilde space ─────────────────
    # The 2SLS VARX model operates on detrended resistance R-tilde, not
    # absolute mOhm. A reference in absolute mOhm must be converted first.
    if op_point is None:
        op_point = [0.95, 1.03, 1.07]   # Step 1 electrode resistance means

    converter       = ReferenceConverter.from_operating_point(
        op_point=op_point, window_s=1800,
    )
    reference_abs   = load_reference_csv(ref_csv)
    reference_tilde = converter.convert_trajectory(reference_abs, freeze_trend=True)

    trend_at_t0 = converter.current_trend()
    print(f"  Trend at t=0:       {trend_at_t0}")
    print(f"  Reference (abs)[0]: {reference_abs[0] if reference_abs.ndim > 1 else reference_abs[0]}")
    print(f"  Reference (R~)[0]:  {reference_tilde[0]}")

    n = len(reference_tilde)

    # ── kA exogenous trajectory ───────────────────────────────────────────────
    print(f"  kA disturbance: {'Perlin noise' if ka_disturbance else 'frozen at means'}")
    exog_traj = make_ka_exog(n, disturbed=ka_disturbance)

    # ── Controllers ───────────────────────────────────────────────────────────
    controllers = make_controllers(
        name=controller_name,
        config_path=str(controller_config),
        dt=dt,
    )

    # ── Run simulation ────────────────────────────────────────────────────────
    y, du, e = run_closed_loop(
        model       = bundle,
        state       = state,
        reference   = reference_tilde,
        controllers = controllers,
        exog_traj   = exog_traj,
    )

    # ── Output DataFrame ──────────────────────────────────────────────────────
    if reference_tilde.ndim == 1:
        reference_tilde = np.repeat(reference_tilde.reshape(-1, 1), 3, axis=1)

    # Cumulative movement from zero (dead-reckoning — informational only)
    u_cumsum = np.cumsum(np.vstack([np.zeros((1, 3)), du]), axis=0)

    out = pd.DataFrame({
        "t_s":  np.arange(len(y), dtype=float) * dt,
        # Outputs (R-tilde space)
        "y1":   y[:, 0],
        "y2":   y[:, 1],
        "y3":   y[:, 2],
        # Reference (R-tilde space)
        "r1":   np.r_[np.nan, reference_tilde[:, 0]],
        "r2":   np.r_[np.nan, reference_tilde[:, 1]],
        "r3":   np.r_[np.nan, reference_tilde[:, 2]],
        # Control: delta-u (movements)
        "du1":  np.r_[0.0, du[:, 0]],
        "du2":  np.r_[0.0, du[:, 1]],
        "du3":  np.r_[0.0, du[:, 2]],
        # Cumulative position estimate (dead-reckoning, informational only)
        "u1_cumsum": u_cumsum[:, 0],
        "u2_cumsum": u_cumsum[:, 1],
        "u3_cumsum": u_cumsum[:, 2],
        # Errors
        "e1":   np.r_[e[:, 0], np.nan],
        "e2":   np.r_[e[:, 1], np.nan],
        "e3":   np.r_[e[:, 2], np.nan],
        # kA disturbance trajectories (frozen or Perlin)
        "kA1":  np.r_[np.nan, exog_traj["kA1"]],
        "kA2":  np.r_[np.nan, exog_traj["kA2"]],
        "kA3":  np.r_[np.nan, exog_traj["kA3"]],
    })

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    if plotting:
        from run_simulation.scripts.plotting import main as plot_main
        plot_main(path=out_csv)

    return out


if __name__ == "__main__":
    run_closed_loop_from_config(
        ref_csv           = "run_simulation/init_data/reference_res_1.csv",
        controller_name   = "pid",
        controller_config = "run_simulation/init_data/PID_params.csv",
        out_csv           = "run_simulation/history/closed_loop_sim_varx.csv",
        dt                = 1.0,
        ka_disturbance    = True,   # set False to freeze kA at means
        plotting          = True,
    )
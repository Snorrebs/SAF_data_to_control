from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from noise import pnoise1

from run_simulation.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation.closed_loop.closed_loop_sim import run_closed_loop
from run_simulation.closed_loop.controller_registry import make_controllers


MODEL_PATH = Path("run_simulation/models/synthetic_varx_plant.meta.joblib")
HIST_CSV   = Path("run_simulation/init_data/synthetic_varx_plant_init.csv")

# Perlin disturbance — must match parameters used in make_synthetic_plant.py
V_OP         = 400.0   # V
PERLIN_AMP   = 15.0    # V
PERLIN_SCALE = 0.01
PERLIN_OCT   = 4


def load_reference_csv(path: str | Path) -> np.ndarray:
    """Load reference signal from CSV.

    Accepts:
    * columns ``r1``, ``r2``, ``r3``                                          → shape ``(n, 3)``
    * columns ``El1_Resistance_mOhm``, ``El2_Resistance_mOhm``, ``El3_Resistance_mOhm`` → shape ``(n, 3)``
    * single column ``r`` or ``reference``                                    → shape ``(n,)`` (broadcast later)
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
    ref_csv: str | Path,
    controller_name: str,
    controller_config: str | Path,
    out_csv: str | Path,
    dt: float = 1.0,
    plotting: bool = False,
):
    bundle = load_arx_bundle(str(MODEL_PATH))
    state  = load_initial_state(str(HIST_CSV), bundle)
    u0     = state.current_u()
    reference = load_reference_csv(ref_csv)

    n = reference.shape[0] if reference.ndim > 1 else len(reference)

    # Generate live Perlin voltage disturbance for the full simulation horizon.
    # Offset the time index by WARMUP so the disturbance continues from where
    # the warmup left off rather than restarting from the same Perlin phase.
    warmup_len = 300   # must match WARMUP in make_synthetic_plant.py
    v_traj = np.array([
        V_OP + PERLIN_AMP * pnoise1((warmup_len + k) * PERLIN_SCALE, octaves=PERLIN_OCT)
        for k in range(n)
    ])
    exog_traj = {"RMS_V_transformer_filt": v_traj}

    controllers = make_controllers(
        name=controller_name,
        config_path=str(controller_config),
        dt=dt,
    )

    y, u, e = run_closed_loop(
        model=bundle,
        state=state,
        reference=reference,
        controllers=controllers,
        exog_traj=exog_traj,
    )

    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)

    out = pd.DataFrame(
        {
            "t_s": np.arange(len(y), dtype=float) * dt,
            "y1":  y[:, 0],
            "y2":  y[:, 1],
            "y3":  y[:, 2],
            "r1":  np.r_[np.nan, reference[:, 0]],
            "r2":  np.r_[np.nan, reference[:, 1]],
            "r3":  np.r_[np.nan, reference[:, 2]],
            "u1":  np.r_[u0[0],  u[:, 0]],
            "u2":  np.r_[u0[1],  u[:, 1]],
            "u3":  np.r_[u0[2],  u[:, 2]],
            "e1":  np.r_[e[:, 0], np.nan],
            "e2":  np.r_[e[:, 1], np.nan],
            "e3":  np.r_[e[:, 2], np.nan],
            "v_transformer": np.r_[np.nan, v_traj],
        }
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    if plotting:
        from run_simulation.scripts.plotting import main as plot_main
        plot_main(path=out_csv)

    return out


if __name__ == "__main__":

    run_closed_loop_from_config(
        ref_csv="run_simulation/init_data/reference_res.csv",
        controller_name="pid",
        controller_config="run_simulation/init_data/PID_params.csv",
        out_csv="run_simulation/history/closed_loop_sim_varx.csv",
        dt=1.0,
        plotting=True,
    )
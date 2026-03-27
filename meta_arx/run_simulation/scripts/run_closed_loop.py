from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_simulation.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation.closed_loop.closed_loop_sim import run_closed_loop
from run_simulation.closed_loop.controller_registry import make_controllers


MODEL_PATH = Path("run_simulation/models/synthetic_varx_plant.meta.joblib")
HIST_CSV   = Path("run_simulation/init_data/synthetic_varx_plant_init.csv")


def load_reference_csv(path: str | Path) -> np.ndarray:
    """Load reference signal from CSV.

    Accepts:
    * columns ``r1``, ``r2``, ``r3``            → shape ``(n, 3)``
    * columns ``El1_kA``, ``El2_kA``, ``El3_kA`` → shape ``(n, 3)``
    * single column ``r`` or ``reference``       → shape ``(n,)`` (broadcast later)
    """
    df = pd.read_csv(path)

    if all(c in df.columns for c in ["r1", "r2", "r3"]):
        return df[["r1", "r2", "r3"]].to_numpy(dtype=float)

    if all(c in df.columns for c in ["El1_kA", "El2_kA", "El3_kA"]):
        return df[["El1_kA", "El2_kA", "El3_kA"]].to_numpy(dtype=float)

    for col in ("r", "reference"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)

    raise ValueError(
        f"Reference CSV '{path}' must contain ['r1','r2','r3'], "
        "['El1_kA','El2_kA','El3_kA'], or a scalar column 'r'/'reference'."
    )


def run_closed_loop_from_config(
    ref_csv: str | Path,
    controller_name: str,
    controller_config: str | Path,
    out_csv: str | Path,
    dt: float = 1.0,
    plotting: bool = False,
):
    # load trained VARX model
    bundle = load_arx_bundle(str(MODEL_PATH))

    # load initial plant state
    state = load_initial_state(str(HIST_CSV), bundle)

    # initial actuator positions
    u0 = state.current_u()   # shape (3,)

    # load reference signal
    reference = load_reference_csv(ref_csv)

    # build controllers (one per electrode)
    controllers = make_controllers(
        name=controller_name,
        config_path=str(controller_config),
        dt=dt,
    )

    # run simulation
    y, u, e = run_closed_loop(
        model=bundle,
        state=state,
        reference=reference,
        controllers=controllers,
    )

    # broadcast scalar reference to 3 columns for output
    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)

    # save output
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
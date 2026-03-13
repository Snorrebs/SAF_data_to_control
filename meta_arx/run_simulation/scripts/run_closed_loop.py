from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_simulation.closed_loop.arx_state import (
    load_arx_bundle,
    load_initial_state,
    ModelIOConfig,
)
from run_simulation.closed_loop.closed_loop_sim import run_closed_loop
from run_simulation.closed_loop.controller_registry import make_controller


MODEL_PATH = Path("run_simulation/models/arx_el1res_2321_07.meta.joblib")
HIST_CSV = Path("run_simulation/init_data/arx_el1res_2321_07.csv")


def load_reference_csv(path: str | Path) -> np.ndarray:

    df = pd.read_csv(path)

    if "reference" in df.columns:
        ref = df["reference"].to_numpy(dtype=float)
    else:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            raise ValueError("Reference CSV must contain numeric column")
        ref = df[num_cols[0]].to_numpy(dtype=float)

    return ref


def run_closed_loop_from_config(
    ref_csv: str | Path,
    controller_name: str,
    controller_config: str | Path,
    out_csv: str | Path,
    dt: float = 1.0,
):

    # load trained ARX model
    bundle = load_arx_bundle(str(MODEL_PATH))

    # define IO mapping (plant unchanged)
    io_cfg = ModelIOConfig(
        input_base="El1_pos_m",
        output_col="El1_Resistance_mOhm_filt",
        output_lag_base="y_filt",
    )

    # load initial plant state
    state = load_initial_state(str(HIST_CSV), bundle, io_cfg=io_cfg)

    # initial actuator position
    u0 = state.current_u()

    # load reference signal
    reference = load_reference_csv(ref_csv)

    # build controller
    controller = make_controller(
        name=controller_name,
        config_path=controller_config,
        dt=dt,
    )

    # run simulation
    y, u, e = run_closed_loop(
        model=bundle,
        state=state,
        reference=reference,
        controller=controller,
    )

    # save output
    out = pd.DataFrame(
        {
            "t_s": np.arange(len(y)) * dt,
            "y_pred": y,
            "u_cmd": np.r_[u0, u],
            "error": np.r_[np.nan, e],
            "reference": np.r_[np.nan, reference],
        }
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_csv, index=False)

    return out


if __name__ == "__main__":

    run_closed_loop_from_config(
        ref_csv="run_simulation/init_data/reference.csv",
        controller_name="pid",
        controller_config="run_simulation/init_data/PID_params.csv",
        out_csv="run_simulation/history/closed_loop_sim.csv",
        dt=1.0,
    )

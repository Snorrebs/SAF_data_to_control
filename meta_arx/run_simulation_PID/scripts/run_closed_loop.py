# python -m run_simulation_PID.scripts.run_closed_loop

from pathlib import Path

import numpy as np
import pandas as pd

from run_simulation_PID.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation_PID.closed_loop.closed_loop_sim import run_closed_loop
from run_simulation_PID.closed_loop.controller import PIDController, PIDParams

MODEL_PATH = Path("run_simulation_PID/models/arx_el1res_2321_07.meta.joblib")
HIST_CSV = Path("run_simulation_PID/init_data/arx_el1res_2321_07.csv")
OUT_CSV = Path("run_simulation_PID/history/closed_loop_sim.csv")


def main() -> None:
    bundle = load_arx_bundle(str(MODEL_PATH))
    state = load_initial_state(str(HIST_CSV), bundle)
    u0 = state.current_u()

    n_steps = 200
    reference = np.full(n_steps, 1.1)

    controller = PIDController(
        params=PIDParams(Kp=0.002, Ki=0.0, Kd=0.0),
        Ts=1.0,
        u_min=-5,
        u_max=5,
        du_max=0.001,
    )

    y, u, e = run_closed_loop(
        model=bundle,
        state=state,
        reference=reference,
        controller=controller,
    )

    t = np.arange(len(y), dtype=float) * controller.Ts
    out = pd.DataFrame(
        {
            "t_s": t,
            "y_pred_mOhm": y,
            "u_cmd": np.append([u0], u),
            "e": np.append(e, np.nan),
            "r": np.append(reference, np.nan),
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV}")


if __name__ == "__main__":
    main()

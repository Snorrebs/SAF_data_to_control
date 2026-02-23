# python -m run_simulation_PID.scripts.run_closed_loop \
#     --kp -0.2 --ki 0.0 --kd 0.2 \
#     --ref_csv run_simulation_PID/init_data/reference.csv \
#     --out_csv run_simulation_PID/history/closed_loop_sim1.csv

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from run_simulation_PID.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation_PID.closed_loop.closed_loop_sim import run_closed_loop
from run_simulation_PID.closed_loop.controller import PIDController, PIDParams


MODEL_PATH = Path("run_simulation_PID/models/arx_el1res_2321_07.meta.joblib")
HIST_CSV = Path("run_simulation_PID/init_data/arx_el1res_2321_07.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Run closed-loop ARX + PID simulation")

    parser.add_argument("--kp", type=float, required=True)
    parser.add_argument("--ki", type=float, required=True)
    parser.add_argument("--kd", type=float, required=True)

    parser.add_argument(
        "--ref_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        default="run_simulation_PID/history/closed_loop_sim.csv",
    )

    return parser.parse_args()


def load_reference(path: str) -> np.ndarray:
    df = pd.read_csv(path)

    return df["r"].values


def main() -> None:
    args = parse_args()

    bundle = load_arx_bundle(str(MODEL_PATH))
    state = load_initial_state(str(HIST_CSV), bundle)
    u0 = state.current_u()

    reference = load_reference(args.ref_csv)

    controller = PIDController(
        params=PIDParams(Kp=args.kp, Ki=args.ki, Kd=args.kd),
        Ts=1.0,
        u_min=-5,
        u_max=5,
        du_max=0.01,
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

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
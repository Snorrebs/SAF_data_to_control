"""python -m run_simulation_PID.scripts.run_closed_loop

Examples:
  # scalar reference (broadcasts to all 3 currents)
  python -m run_simulation_PID.scripts.run_closed_loop \
    --ref_csv run_simulation_PID/init_data/reference.csv \
    --out_csv run_simulation_PID/history/closed_loop_varx.csv

  # 3-column reference
  # CSV must contain columns: r1,r2,r3 (or El1_kA,El2_kA,El3_kA)
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from run_simulation_PID.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation_PID.closed_loop.closed_loop_sim import run_closed_loop
from run_simulation_PID.closed_loop.controller import PIDController, PIDParams


# Defaults for VARX-current
MODEL_PATH = Path("run_simulation_PID/models/varx_res_2321_07_ridge.meta.joblib")
HIST_CSV = Path("run_simulation_PID/init_data/varx_res_2321_07.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Run closed-loop VARX-current + PID simulation")

    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_PATH),
        help="Path to trained VARX model bundle (.meta.joblib)",
    )

    parser.add_argument(
        "--hist_csv",
        type=str,
        default=str(HIST_CSV),
        help="History CSV for initializing lag state",
    )

    parser.add_argument(
        "--PID_params_csv",
        type=str,
        default="run_simulation_PID/init_data/PID_params.csv",
        help="CSV with PID params. Either 1 row (applied to all) or 3 rows.",
    )

    parser.add_argument(
        "--ref_csv",
        type=str,
        required=True,
        help="Reference CSV: either column 'r' (broadcast) or 3 columns r1,r2,r3.",
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        default="run_simulation_PID/history/closed_loop_sim_varx.csv",
    )

    # basic saturation defaults (tune later)
    parser.add_argument("--u_min", type=float, default=-np.inf)
    parser.add_argument("--u_max", type=float, default=np.inf)
    parser.add_argument("--du_max", type=float, default=None)

    return parser.parse_args()


def load_reference(path: str) -> np.ndarray:
    df = pd.read_csv(path)

    # preferred
    if all(c in df.columns for c in ["r1", "r2", "r3"]):
        return df[["r1", "r2", "r3"]].to_numpy(dtype=float)

    # accept target-like names
    if all(c in df.columns for c in ["El1_kA", "El2_kA", "El3_kA"]):
        return df[["El1_kA", "El2_kA", "El3_kA"]].to_numpy(dtype=float)

    # legacy scalar
    if "r" in df.columns:
        return df["r"].to_numpy(dtype=float)

    raise ValueError("Reference CSV must contain either ['r'] or ['r1','r2','r3']")


def load_PID_params(path: str) -> list[PIDParams]:
    df = pd.read_csv(path)

    needed = {"Kp", "Ki", "Kd"}
    if not needed.issubset(df.columns):
        raise ValueError(f"PID params CSV must contain columns {sorted(needed)}")

    if len(df) == 1:
        p = PIDParams(Kp=float(df["Kp"].iloc[0]), Ki=float(df["Ki"].iloc[0]), Kd=float(df["Kd"].iloc[0]))
        return [p, p, p]

    if len(df) >= 3:
        ps: list[PIDParams] = []
        for i in range(3):
            ps.append(PIDParams(Kp=float(df["Kp"].iloc[i]), Ki=float(df["Ki"].iloc[i]), Kd=float(df["Kd"].iloc[i])))
        return ps

    raise ValueError("PID params CSV must have either 1 row or at least 3 rows")


def main() -> None:
    args = parse_args()

    bundle = load_arx_bundle(args.model_path)
    state = load_initial_state(args.hist_csv, bundle)

    ref = load_reference(args.ref_csv)
    print(f"Loaded reference with shape {ref.shape} from {args.ref_csv}")
    pid_params = load_PID_params(args.PID_params_csv)
    print(f"Loaded PID parameters: {pid_params}")

    controllers = [
        PIDController(
            params=pid_params[i],
            Ts=1.0,
            u_min=0.1 if np.isneginf(args.u_min) else args.u_min,
            u_max=2 if np.isposinf(args.u_max) else args.u_max,
            du_max=0.01 if args.du_max is None else args.du_max,
        )
        for i in range(3)
    ]

    y, u, e = run_closed_loop(
        model=bundle,
        state=state,
        reference=ref,
        controllers=controllers,
    )
    print(f"Closed-loop simulation completed. y shape: {y.shape}, u shape: {u.shape}, e shape: {e.shape}")
    t = np.arange(len(y), dtype=float) * controllers[0].Ts

    out = pd.DataFrame(
        {
            "t_s": t,
            "y1": y[:, 0],
            "y2": y[:, 1],
            "y3": y[:, 2],
            "r1": np.r_[np.nan, ref[:, 0]],
            "r2": np.r_[np.nan, ref[:, 1]],
            "r3": np.r_[np.nan, ref[:, 2]],
            "u1": np.r_[np.nan, u[:, 0]],
            "u2": np.r_[np.nan, u[:, 1]],
            "u3": np.r_[np.nan, u[:, 2]],
            "e1": np.r_[e[:, 0], np.nan],
            "e2": np.r_[e[:, 1], np.nan],
            "e3": np.r_[e[:, 2], np.nan],
        }
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()

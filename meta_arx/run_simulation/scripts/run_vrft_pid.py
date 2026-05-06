from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_simulation.vrft.vrft_pid import vrft_pid_from_csv_filtered


# ── defaults ──────────────────────────────────────────────────────────────────
HISTORY_CSV  = Path("run_simulation/history/closed_loop_sim_varx.csv")
OUT_CSV      = Path("run_simulation/init_data/PID_params_vrft.csv")

# Per-electrode column mapping in the history CSV
_ELECTRODES = [
    {"y_col": "y1", "u_col": "u1"},
    {"y_col": "y2", "u_col": "u2"},
    {"y_col": "y3", "u_col": "u3"},
]

# Reference model and filter parameters
_DEFAULT_TS    = 1.0    # sampling time [s]
_DEFAULT_TAU   = 0.0    # transport delay [s]
_DEFAULT_T     = 30.0   # desired closed-loop time constant [s]
_DEFAULT_Q     = 2      # reference model order
_DEFAULT_OMEGA   = 0.2   # VRFT weighting filter bandwidth [rad/s]
_DEFAULT_WARMUP  = 20    # samples discarded after filtering to skip transient


def run_vrft(
    history_csv: str | Path = HISTORY_CSV,
    out_csv: str | Path = OUT_CSV,
    Ts: float = _DEFAULT_TS,
    tau: float = _DEFAULT_TAU,
    t: float = _DEFAULT_T,
    q: int = _DEFAULT_Q,
    omega: float = _DEFAULT_OMEGA,
    warmup: int = _DEFAULT_WARMUP,
    min_samples: int = 50,
) -> pd.DataFrame:
    """Run filtered VRFT on a closed-loop history CSV and save tuned PID params.

    Args:
        history_csv: path to history CSV with columns y1..y3 and u1..u3
        out_csv:     where to write the 3-row PID params CSV
        Ts:          sampling time [s]
        tau:         reference model transport delay [s]
        t:           desired closed-loop time constant [s] — main tuning knob
        q:           reference model order (1 = first-order, 2 = second-order)
        omega:       VRFT weighting filter bandwidth [rad/s]
        min_samples: minimum number of non-NaN rows required

    Returns:
        DataFrame with columns kp, ki, kd (one row per electrode).
    """
    history_csv = Path(history_csv)
    if not history_csv.exists():
        raise FileNotFoundError(
            f"History CSV not found: {history_csv}\n"
            "Run run_closed_loop.py (or run_mpc.py) first to generate simulation data."
        )

    df = pd.read_csv(history_csv).dropna(subset=["y1", "u1"])

    if len(df) < min_samples:
        raise ValueError(
            f"History CSV has only {len(df)} usable rows (need >= {min_samples}).\n"
            "Re-run the simulation with a longer reference signal."
        )

    rows = []
    for i, cols in enumerate(_ELECTRODES, start=1):
        params = vrft_pid_from_csv_filtered(
            csv_path=str(history_csv),
            y_col=cols["y_col"],
            u_col=cols["u_col"],
            Ts=Ts,
            tau=tau,
            t=t,
            q=q,
            omega=omega,
            warmup=warmup,
        )
        rows.append({"electrode": i, "kp": params.Kp, "ki": params.Ki, "kd": params.Kd})
        print(f"  El{i}: Kp={params.Kp:.6f}  Ki={params.Ki:.6f}  Kd={params.Kd:.6f}")

    result = pd.DataFrame(rows)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result[["kp", "ki", "kd"]].to_csv(out_csv, index=False)
    print(f"\nSaved tuned PID params → {out_csv}")

    return result


if __name__ == "__main__":
    print(f"Running VRFT on {HISTORY_CSV} ...")
    run_vrft()

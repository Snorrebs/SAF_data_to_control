#!/usr/bin/env python3
"""
Run filtered VRFT PID tuning on a closed-loop CSV.

Usage:
    python -m run_simulation_PID.scripts.run_vrft_pid

"""

from pathlib import Path
from run_simulation_PID.scripts.vrft_pid import vrft_pid_from_csv_filtered, PIDParams

# Path to the CSV produced by closed-loop simulation
CSV_PATH = Path("run_simulation_PID/history/closed_loop_sim.csv")

# Column names in that CSV
Y_COL = "y_pred_mOhm"   # output (resistance)
U_COL = "u_El2_pos_m"   # input (electrode position command)

# Sampling times
TS = 1.0  # [s]

# Filtered VRFT hyperparameters (reference model + weighting)
TAU = 0.0        # time delay in M(s) [s]
T_SHAPE = 400.0    # shaping parameter in (1 + 0.2*T_SHAPE*s)^q
Q_ORDER = 3      # reference model order q
OMEGA = 1.0     # cutoff for W(s) = omega / (s + omega)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    pid: PIDParams = vrft_pid_from_csv_filtered(
        csv_path=str(CSV_PATH),
        y_col=Y_COL,
        u_col=U_COL,
        Ts=TS,
        tau=TAU,
        t=T_SHAPE,
        q=Q_ORDER,
        omega=OMEGA,
    )

    print("=== Filtered VRFT PID parameters ===")
    print(f"CSV     : {CSV_PATH}")
    print(f"Ts      : {TS}")
    print(f"tau     : {TAU}")
    print(f"t_shape : {T_SHAPE}")
    print(f"q_order : {Q_ORDER}")
    print(f"omega   : {OMEGA}")
    print()
    print(f"Kp = {pid.Kp:.6g}")
    print(f"Ki = {pid.Ki:.6g}")
    print(f"Kd = {pid.Kd:.6g}")
    print("\nPaste these into run_closed_loop.py (PIDParams).")


if __name__ == "__main__":
    main()


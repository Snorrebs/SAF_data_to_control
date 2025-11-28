# run_simulation/scripts/run_vrft_pid.py
#!/usr/bin/env python3

#python -m run_simulation_PID_VRFT.scripts.run_vrft_pid  --data-csv run_simulation_PID_VRFT/closed_loop/closed_loop_sim.csv  --Ts 1.0  --tau-cl 300

import argparse

from run_simulation_PID_VRFT.vrft.vrft_pid import vrft_pid_from_csv


def main():
    p = argparse.ArgumentParser(description="VRFT PID tuning from closed-loop data")
    p.add_argument("--data-csv", required=True,
                   help="CSV with columns y_pred_mOhm and u_El1_pos_m")
    p.add_argument("--Ts", type=float, default=1.0,
                   help="Sampling time in seconds")
    p.add_argument("--tau-cl", type=float, default=300.0,
                   help="Desired closed-loop time constant in seconds")
    p.add_argument("--no-integral", action="store_true",
                   help="Disable integral term in VRFT")
    p.add_argument("--use-derivative", action="store_true",
                   help="Enable derivative term in VRFT")

    args = p.parse_args()

    pid = vrft_pid_from_csv(
        csv_path=args.data_csv,
        Ts=args.Ts,
        tau_cl=args.tau_cl,
        use_integral=not args.no_integral,
        use_derivative=args.use_derivative,
    )

    print("=== VRFT PID parameters ===")
    print(f"Kp = {pid.Kp:.6g}")
    print(f"Ki = {pid.Ki:.6g}")
    print(f"Kd = {pid.Kd:.6g}")
    print("\nPaste these into run_closed_loop.py's PIDParams.")


if __name__ == "__main__":
    main()

# run_simulation/scripts/run_closed_loop_vrft_online.py
# Usage (from repo root):
#   python -m run_simulation_online_VRFT.scripts.run_closed_loop_vrft_online

import numpy as np
import pandas as pd
from pathlib import Path

from run_simulation_online_VRFT.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation_online_VRFT.closed_loop.closed_loop_sim import PIDParams
from run_simulation_online_VRFT.closed_loop.vrft_online import OnlineVRFTPID


def main():
    # ---- Paths (adapt as needed) ----
    MODEL_PATH = Path("run_simulation_online_VRFT/models/arx_el1res_2321_07.meta.joblib")
    HIST_CSV   = Path("run_simulation_online_VRFT/init_data/arx_el1res_2321_07.csv")
    OUT_CSV    = Path("run_simulation_online_VRFT/closed_loop/closed_loop_sim_vrft_online.csv")

    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"
    assert HIST_CSV.exists(), f"Missing history CSV: {HIST_CSV}"

    # ---- Load ARX model bundle + initial state ----
    bundle = load_arx_bundle(str(MODEL_PATH))
    state = load_initial_state(str(HIST_CSV), bundle)

    # Initial conditions from state
    y0 = state.current_y()
    u0 = state.current_u_el1()

    # ---- Simulation settings ----
    Ts = 1.0        # sampling interval [s]
    N  = 1000       # number of control steps

    # Reference trajectory (example: constant 1.2 mΩ)
    r = np.full(N, 1.05, dtype=float)

    # Actuator limits
    u_min, u_max = -5, 5
    du_max = 1

    # Initial PID guess (before VRFT has enough data)
    pid_init = PIDParams(Kp=0.003, Ki=0.0, Kd=0.0)

    # ---- Online VRFT tuner: last 600 s = 10 minutes window ----
    vrft = OnlineVRFTPID(
        Ts=Ts,
        T_window=200.0,   # seconds of history used for each tuning
        tau=0.0,
        t_settle=5.0,
        q_order=3,
        omega=10.0,
        lam=1,
        alpha=0.005,
        initial_pid=pid_init,
    )

    # ---- Allocate logs ----
    y = np.zeros(N + 1, dtype=float)
    u = np.zeros(N, dtype=float)
    e = np.zeros(N, dtype=float)

    Kp_log = np.zeros(N, dtype=float)
    Ki_log = np.zeros(N, dtype=float)
    Kd_log = np.zeros(N, dtype=float)

    # Initial conditions
    y[0] = y0
    u_prev = u0
    y_prev = y0
    int_e = 0.0

    # ---- Main adaptive closed-loop loop ----
    for k in range(N):
        # 1) Predict next output from current ARX state
        y_pred = state.predict_next_y(bundle)
        y[k + 1] = y_pred

        # 2) Feed the new sample (y_pred, u_prev) to VRFT tuner
        vrft.update_buffers(y_pred, u_prev)
        pid = vrft.maybe_retrain()

        # Log gains
        Kp_log[k] = pid.Kp
        Ki_log[k] = pid.Ki
        Kd_log[k] = pid.Kd

        # 3) PID control using the *measured* (predicted) output y_pred
        e[k] = r[k] - y_pred
        int_e += e[k] * Ts
        dy = (y_pred - y_prev) / Ts if k > 0 else 0.0

        # Controller structure consistent with VRFT regressor and with the baseline
        # closed_loop_sim: we compute an *increment* Δu and add it to u_prev.
        du_raw = (
            pid.Kp * e[k]
            + pid.Kd * (-dy)
            + pid.Ki * int_e
        )

        u_cmd = u_prev + du_raw


        # 4) Apply rate limit and saturation
        if du_max is not None:
            u_cmd = np.clip(u_cmd, u_prev - du_max, u_prev + du_max)

        u_cmd = np.clip(u_cmd, u_min, u_max)


        u[k] = u_cmd
        u_prev = u_cmd
        y_prev = y_pred

        # 5) Advance ARX state using new control + predicted y
        #    NOTE: your ArxState.advance currently expects u_el2_new
        state.advance(u_el2_new=u_cmd, y_new=y_pred)

    # ---- Save to CSV ----
    t = np.arange(N + 1) * Ts
    u_log = np.append([u0], u)  # align u with y

    df_out = pd.DataFrame(
        {
            "t_s": t,
            "y_pred_mOhm": y,
            "u_El1_pos_m": u_log,      # for plotting, even though ARX uses El2 features
            "e": np.append(e, np.nan),
            "r": np.append(r, np.nan),
            "Kp": np.append(Kp_log, np.nan),
            "Ki": np.append(Ki_log, np.nan),
            "Kd": np.append(Kd_log, np.nan),
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV}")


if __name__ == "__main__":
    main()

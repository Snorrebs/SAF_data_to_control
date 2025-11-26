#python3 -m run_simulation.scripts.run_closed_loop

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

from run_simulation.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation.closed_loop.closed_loop_sim import run_closed_loop, PIDParams

def main():
    # ---- paths (adjust to your repo) ----
    MODEL_PATH = Path("run_simulation/models/arx_linear_ridge_stable_yonly.joblib")
    HIST_CSV   = Path("run_simulation/init_data/model_arx_1_5_5.csv")
    OUT_CSV    = Path("run_simulation/closed_loop/closed_loop_sim.csv")

    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"
    assert HIST_CSV.exists(), f"Missing history CSV: {HIST_CSV}"

    # ---- load model + initial ARX state ----
    bundle = load_arx_bundle(str(MODEL_PATH))
    state = load_initial_state(str(HIST_CSV), bundle)

    # initial electrode position for logging
    u0 = state.current_u_el1()



    # ---- build reference trajectory ----
    N = 1000
    r = np.full(N, 1.2) 

    # ---- PID params + limits ----
    pid = PIDParams(Kp=0.003, Ki=0.0, Kd=0.0)  # tune
    Ts = 1.0
    u_min, u_max = 0, 3.8
    du_max = 1

    # ---- run closed-loop sim ----
    y, u, e = run_closed_loop(
        model=bundle,        
        state=state,
        r=r,
        pid=pid,
        Ts=Ts,
        u_min=u_min,
        u_max=u_max,
        du_max=du_max,
    )

    # ---- save to CSV for plotting ----
    t = np.arange(len(y)) * Ts  # y has length N+1

    # Build u log with same length as y: [u0, u[0], u[1], ..., u[N-1]]
    u_log = np.append([u0], u)  # length N+1

    df_out = pd.DataFrame(
        {
            "t_s": t,
            "y_pred_mOhm": y,
            "u_El1_pos_m": u_log,
            "e": np.append(e, np.nan),
            "r": np.append(r, np.nan),
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV}")
    bundle = load("run_simulation/models/arx_linear_ridge_stable_yonly.joblib")
    print(bundle["exog_cols"])

    import run_simulation.scripts.plotting as plotting
    plotting.main()


if __name__ == "__main__":
    main()

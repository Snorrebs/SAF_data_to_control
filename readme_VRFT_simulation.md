# Offline Closed-Loop Simulation & VRFT PID Tuning

## Folder Structure — `run_simulation_offline_VRFT`

run_simulation_offline_VRFT/
│
├─ closed_loop/
│ ├─ arx_state.py # ARX state container and prediction logic
│ ├─ closed_loop_sim.py # Core closed-loop simulation engine
│ ├─ closed_loop_sim.csv # Output log from simulations
│
├─ init_data/
│ └─ model_arx_1_5_5.csv # Initial lag history for ARX state
│
├─ models/
│ ├─ arx_linear_ridge_stable_yonly.joblib # Trained ARX bundle
│ └─ *.csv # Optional coefficient exports
│
├─ scripts/
│ ├─ run_closed_loop.py # Entry point for closed-loop simulation
│ └─ run_vrft_pid_offline.py # Entry point for VRFT PID tuning
│
├─ vrft/
│ └─ vrft_pid.py # Filtered VRFT implementation
│
└─ init.py


## Pipeline

ARX Model + Initial State
│
▼
Closed-Loop Simulation (PID)
│
▼
CSV Log (y, u, e, r, t)
│
▼
Filtered VRFT
│
▼
New PID Parameters
│
└──► Re-run Simulation


## Run Method

### 1. Closed-Loop Simulation  
**Script:** `scripts/run_closed_loop.py`

**Terminal command:**

```bash
python -m run_simulation_offline_VRFT.scripts.run_closed_loop 
```
This runs the simulation and produces: `closed_loop/closed_loop_sim.csv`

CSV Columns:
t_s – time [s]
y_pred_mOhm – predicted resistance
u_El1_pos_m – electrode position command
e – control error
r – reference trajectory

### 2. VRFT PID Tuning
**Script**: `scripts/run_vrft_pid_offline.py`
**Terminal command**:
```bash
python -m run_simulation_offline_VRFT.scripts.run_vrft_pid_offline
```

Uses the closed-loop CSV to estimate PID parameters using Filtered VRFT.

VRFT Configuration (inside script)
| Parameter | Meaning                          |
| --------- | -------------------------------- |
| `TS`      | Sampling time [s]                |
| `TAU`     | Time delay in reference model    |
| `T_SHAPE` | Reference model shaping constant |
| `Q_ORDER` | Reference model order            |
| `OMEGA`   | Weighting filter cutoff          |


Console returns: Kp, Ki, Kd
These gains are then copied into run_closed_loop.py (PIDParams) for the next simulation iteration.
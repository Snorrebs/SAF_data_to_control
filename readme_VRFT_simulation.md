# Offline Closed-Loop Simulation & VRFT PID Tuning

## Folder Structure — `meta_arx/run_simulation_PID`

meta_arx/run_simulation_PID/
│
├─ closed_loop/
│  ├─ arx_state.py               # ARX state container and prediction logic
│  └─ closed_loop_sim.py         # Core closed-loop simulation engine
│
├─ scripts/
│  ├─ run_closed_loop.py         # Entry point for closed-loop simulation
│  ├─ run_vrft_pid.py            # Entry point for VRFT PID tuning
│  └─ plotting.py                # Quick plot utility for closed-loop CSV
│
├─ vrft/
│  └─ vrft_pid.py                # Filtered VRFT implementation
│
└─ __init__.py

## Pipeline

ARX Model + Initial State
→ Closed-Loop Simulation (PID)
→ CSV Log (`y`, `u`, `e`, `r`, `t`)
→ Filtered VRFT
→ New PID Parameters
→ Re-run Simulation

## Run Method

### 1. Closed-Loop Simulation
From the repository root:

```bash
cd meta_arx
python -m run_simulation_PID.scripts.run_closed_loop
```

This creates `run_simulation_PID/history/closed_loop_sim.csv`.

### 2. VRFT PID Tuning
From the same `meta_arx` directory:

```bash
python -m run_simulation_PID.scripts.run_vrft_pid
```

Console returns `Kp`, `Ki`, `Kd` for reuse in `run_closed_loop.py`.

## VRFT Configuration
The VRFT script exposes:

- `TS`: sampling time [s]
- `TAU`: reference-model delay [s]
- `T_SHAPE`: reference-model shaping constant
- `Q_ORDER`: reference-model order
- `OMEGA`: weighting filter cutoff

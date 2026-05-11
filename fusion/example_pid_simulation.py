"""
PID closed-loop example using the ARX + GP plant in fusion/models/.

Usage:
    python fusion/example_pid_simulation.py

Outputs to fusion/results/example_pid/:
    closed_loop_result.csv, resistance.pdf, positions.pdf
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # SAF_data_to_control/
META_ARX     = PROJECT_ROOT / "meta_arx"
for _p in [str(PROJECT_ROOT), str(META_ARX)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion.run_closed_loop import run_closed_loop_from_config

OUT_DIR = PROJECT_ROOT / "fusion" / "results" / "example_pid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-electrode references, with a step at t=150
N = 300
R0 = [1.20, 0.70, 1.14]   #[Ref_el1,Ref_el2,Ref_el3]
R1 = [1.05, 0.58, 1.00]   # step target at t=150

ref_arr = np.array([[R0[i]] * N for i in range(3)], dtype=float).T
ref_arr[150:] = R1

pd.DataFrame({"r1": ref_arr[:, 0], "r2": ref_arr[:, 1], "r3": ref_arr[:, 2]}).to_csv(
    OUT_DIR / "reference.csv", index=False
)

# PI, kp saturates actuator rate limit for errors > 0.1 mOhm, 
pd.DataFrame({"kp": [0.10], "ki": [0.001], "kd": [0.0]}).to_csv(
    OUT_DIR / "PID_params.csv", index=False
)

# Run simulation
df = run_closed_loop_from_config(
    ref_csv           = OUT_DIR / "reference.csv",
    controller_name   = "pid",
    controller_config = OUT_DIR / "PID_params.csv",
    out_csv           = OUT_DIR / "closed_loop_result.csv",
    dt                = 1.0,
)

COLORS = ["blue", "red", "green"]

# Plot 1: resistance, one subplot per electrode so per-electrode references are clear
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, (el, c, ax) in enumerate(zip([1, 2, 3], COLORS, axes), start=1):
    ax.plot(df["t_s"], df[f"y{el}"], color=c, lw=0.6, alpha=0.35)
    ax.plot(df["t_s"], df[f"y{el}"].rolling(15, center=True, min_periods=1).mean(),
            color=c, lw=1.8, label=f"El{el} (filtered)")
    ax.plot(df["t_s"], df[f"r{el}"], "k--", lw=0.9, label="Reference")
    ax.set_ylabel("R (mOhm)")
    ax.set_title(f"Electrode {el}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("PID closed-loop, arc resistance")
fig.tight_layout()
fig.savefig(OUT_DIR / "resistance.pdf")
plt.close(fig)

# Plot 2: electrode positions
fig, ax = plt.subplots(figsize=(10, 4))
for el, c in zip([1, 2, 3], COLORS):
    ax.plot(df["t_s"], df[f"u{el}"], color=c, lw=1.3, label=f"El{el}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (m)")
ax.set_title("PID closed-loop, electrode positions")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "positions.pdf")
plt.close(fig)

print(f"Done. Outputs in {OUT_DIR}")

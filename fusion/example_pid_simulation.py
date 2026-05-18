"""
PID closed-loop example using the ARX + GP plant in fusion/models/.

Usage:
    python fusion/example_pid_simulation.py

Outputs to fusion/results/example_pid/:
    closed_loop_result.csv, resistance.pdf, positions.pdf

To run with a specific model variant, pass gp_variant to run_closed_loop_from_config:

    df = run_closed_loop_from_config(..., gp_variant="v7")

Available variants: "txt2026_512", "pi_512", "combined_512", "combined_deep_512",
                    "v6" (joint ARX + debiased GP), "v7" (joint ARX + linear + GP)
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

# Per-electrode references.
# R0 matches V6's training-data operating point so each electrode starts
# at near-zero error. A step down occurs at t=500 s.
N  = 1000
R0 = [1.08, 1.07, 1.07]   # near V6 training means
R1 = [0.97, 0.97, 0.97]   # step target at t=500

ref_arr = np.array([[R0[i]] * N for i in range(3)], dtype=float).T
ref_arr[500:] = R1

pd.DataFrame({"r1": ref_arr[:, 0], "r2": ref_arr[:, 1], "r3": ref_arr[:, 2]}).to_csv(
    OUT_DIR / "reference.csv", index=False
)

# PI gains tuned for V6 velocity dynamics.
# V6 dpos training std is ~0.7 mm/s; with kp=0.003 a 0.1 mOhm error
# commands 0.3 mm/step (< 1 sigma of training dpos), keeping the ARX
# within its training distribution.
pd.DataFrame({"kp": [0.003], "ki": [0.0002], "kd": [0.0]}).to_csv(
    OUT_DIR / "PID_params.csv", index=False
)

# Run simulation.
# Change gp_variant to switch models: "v7", "v6", "pi_512", "txt2026_512", etc.
df = run_closed_loop_from_config(
    ref_csv           = OUT_DIR / "reference.csv",
    controller_name   = "pid",
    controller_config = OUT_DIR / "PID_params.csv",
    out_csv           = OUT_DIR / "closed_loop_result.csv",
    dt                = 1.0,
    gp_variant        = "v7",
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

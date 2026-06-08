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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_ARX     = PROJECT_ROOT / "meta_arx"
for _p in [str(PROJECT_ROOT), str(META_ARX)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion.run_closed_loop import run_closed_loop_from_config

OUT_DIR = PROJECT_ROOT / "fusion" / "results" / "example_pid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1000 steps. El1 steps up by 0.06 mOhm at t=400 and returns at t=700.
# El2 and El3 hold at their operating points throughout.
N   = 1000
OP  = [1.08, 1.07, 1.07]

ref_arr = np.full((N, 3), OP, dtype=float)
ref_arr[400:700, 0] = OP[0] + 0.06
ref_arr[700:,   0] = OP[0]

R0 = OP
R1 = [OP[0] + 0.06, OP[1], OP[2]]

pd.DataFrame({"r1": ref_arr[:, 0], "r2": ref_arr[:, 1], "r3": ref_arr[:, 2]}).to_csv(
    OUT_DIR / "reference.csv", index=False
)

# kp=-0.05, ki=0: moderate proportional, no integral (avoids windup from
# cross-electrode coupling drift in independent-controller mode).
pd.DataFrame({"kp": [-0.05], "ki": [0.0], "kd": [0.0]}).to_csv(
    OUT_DIR / "PID_params.csv", index=False
)

df = run_closed_loop_from_config(
    ref_csv           = OUT_DIR / "reference.csv",
    controller_name   = "pid",
    controller_config = OUT_DIR / "PID_params.csv",
    out_csv           = OUT_DIR / "closed_loop_result.csv",
    dt                = 1.0,
)

COLORS = ["C0", "C1", "C2"]

# Plot 1: arc resistance with reference overlay
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for i, (el, c, ax) in enumerate(zip([1, 2, 3], COLORS, axes), start=1):
    ax.plot(df["t_s"], df[f"y{el}"], color=c, lw=0.8, alpha=0.5)
    ax.plot(df["t_s"],
            df[f"y{el}"].rolling(20, center=True, min_periods=1).mean(),
            color=c, lw=2.0, label=f"El{el}")
    ax.axhline(R0[i-1], color="grey", ls=":",  lw=0.9, label=f"R0={R0[i-1]:.3f}")
    ax.axhline(R1[i-1], color="k",    ls="--", lw=0.9, label=f"R1={R1[i-1]:.3f}")
    for vt in [400, 700]:
        ax.axvline(vt, color="grey", ls="--", lw=0.6, alpha=0.5)
    ax.set_ylabel("R (mOhm)")
    ax.set_title(f"Electrode {i}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("PID closed-loop: arc resistance", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "resistance.pdf")
plt.close(fig)

# Plot 2: electrode positions
fig, ax = plt.subplots(figsize=(13, 4))
for el, c in zip([1, 2, 3], COLORS):
    ax.plot(df["t_s"], df[f"u{el}"], color=c, lw=1.2, label=f"El{el}")
ax.axvline(400, color="grey", ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (m)")
ax.set_title("Electrode positions")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "positions.pdf")
plt.close(fig)

print(f"\nSteady-state (steps 900-1000):")
tail = df.tail(100)
for el in [1, 2, 3]:
    rmse = ((tail[f"y{el}"] - R0[el-1])**2).mean()**0.5
    print(f"  El{el}: mean={tail[f'y{el}'].mean():.4f}  ref={R0[el-1]:.4f}  RMSE={rmse:.4f} mOhm")
print(f"\nDone. Outputs in {OUT_DIR}")

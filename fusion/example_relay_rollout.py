"""
Relay controller example using the rollout SVGP + step9 VARX plant.

The relay controller matches the training distribution of the step9 VARX
(which was identified on relay-controlled data) and avoids the cross-electrode
oscillations that arise when three independent PID controllers are used.

Usage
    cd C:\\MASTER\\SAF_data_to_control
    python fusion/example_relay_rollout.py

Outputs to fusion/results/example_relay/:
    closed_loop_result.csv
    resistance.pdf
    positions.pdf
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

from fusion.run_closed_loop_rollout import run_closed_loop_from_config

OUT_DIR = PROJECT_ROOT / "fusion" / "results" / "example_relay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Operating point (absolute R, mOhm)
OP = [1.08, 1.07, 1.07]

# 2000-step simulation:
#   Phase 1 (t=0-600):   hold at OP, let trend settle inside deadband
#   Phase 2 (t=600-1200): step to new setpoint (+0.06 / -0.06 / +0.05 mOhm)
#   Phase 3 (t=1200-2000): return to OP, test disturbance rejection
N   = 2000
R0  = OP
R1  = [op + d for op, d in zip(OP, [+0.06, -0.06, +0.05])]

ref_arr = np.array([[R0[i]] * N for i in range(3)], dtype=float).T
ref_arr[600:1200] = R1
ref_arr[1200:]    = R0

pd.DataFrame({
    "r1": ref_arr[:, 0],
    "r2": ref_arr[:, 1],
    "r3": ref_arr[:, 2],
}).to_csv(OUT_DIR / "reference.csv", index=False)

# Relay params, identical to the real-plant relay tuning
# r_dead: half-deadband in mOhm (R_tilde space, same units as absolute R)
# step_size: electrode move per actuation [m] (matches _DPOS_MAX in simulation)
# wait_normal / wait_slow: steps to wait between moves
# escalate_after: consecutive steps before switching to slow mode
pd.DataFrame({
    "deadband":         [0.07],
    "step_size":        [0.01],
    "wait_normal":      [4],
    "wait_escalated":   [20],
    "escalation_count": [10],
}).to_csv(OUT_DIR / "relay_params.csv", index=False)

df = run_closed_loop_from_config(
    ref_csv           = OUT_DIR / "reference.csv",
    controller_name   = "relay",
    controller_config = OUT_DIR / "relay_params.csv",
    out_csv           = OUT_DIR / "closed_loop_result.csv",
    dt                = 1.0,
)

COLORS = ["C0", "C1", "C2"]

# Plot 1: absolute resistance with reference overlay
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for i, (el, c, ax) in enumerate(zip([1, 2, 3], COLORS, axes), start=1):
    r_dead = 0.07
    ref_col = f"r{el}"
    # R_abs uses the rolling-median trend added back inside the simulation
    ax.fill_between(
        df["t_s"],
        df[f"R_abs{el}"].shift(1).bfill() - r_dead,
        df[f"R_abs{el}"].shift(1).bfill() + r_dead,
        alpha=0.0,
    )
    # Deadband band around reference
    ref_abs_col = pd.concat([
        pd.Series([R0[i-1]] * 1),
        pd.Series(ref_arr[:, i-1]),
    ]).reset_index(drop=True)
    ax.fill_between(
        df["t_s"],
        ref_abs_col.values[:len(df)] - r_dead,
        ref_abs_col.values[:len(df)] + r_dead,
        color="green", alpha=0.12, label="deadband",
    )
    ax.plot(df["t_s"], df[f"R_abs{el}"], color=c, lw=0.7, alpha=0.5)
    ax.plot(
        df["t_s"],
        df[f"R_abs{el}"].rolling(20, center=True, min_periods=1).mean(),
        color=c, lw=2.0, label=f"El{el}",
    )
    ax.axhline(R0[i-1], color="grey", ls=":",  lw=0.9, label=f"R0={R0[i-1]:.3f}")
    ax.axhline(R1[i-1], color="k",    ls="--", lw=0.9, label=f"R1={R1[i-1]:.3f}")
    for vt in [600, 1200]:
        ax.axvline(vt, color="grey", ls="--", lw=0.6, alpha=0.5)
    ax.set_ylabel("R (mOhm)")
    ax.set_title(f"Electrode {i}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Rollout SVGP + Relay: absolute resistance (2000 steps)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "resistance.pdf")
plt.close(fig)

# Plot 2: electrode positions
fig, ax = plt.subplots(figsize=(13, 4))
for el, c in zip([1, 2, 3], COLORS):
    ax.plot(df["t_s"], df[f"u{el}"], color=c, lw=1.2, label=f"El{el}")
for vt in [600, 1200]:
    ax.axvline(vt, color="grey", ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (m)")
ax.set_ylim(0, 2.0)
ax.set_title("Electrode positions")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "positions.pdf")
plt.close(fig)

# Steady-state summary
print(f"\nSteady-state (steps 1800-2000, back at R0):")
f = df.tail(200)
for el in [1, 2, 3]:
    rmse = ((f[f"R_abs{el}"] - R0[el-1]) ** 2).mean() ** 0.5
    print(f"  El{el}: R_abs={f[f'R_abs{el}'].mean():.4f}  ref={R0[el-1]:.4f}  RMSE={rmse:.4f} mOhm")

print(f"\nDone. Outputs in {OUT_DIR}")
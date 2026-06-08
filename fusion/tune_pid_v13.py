"""
tune_pid_v13.py
Automatic PID tuning for the V13 decoupled ARX + GP model.

Runs an open-loop step test on El1, fits a first-order model, and computes PI
parameters using the IMC tuning rule. Then validates with a locked single-mover
PID simulation on all three electrodes.

Run with: python fusion/tune_pid_v13.py

Outputs to fusion/results/pid_tuning_v13/:
  step_response.pdf    open-loop step response with fitted model
  validation.pdf       closed-loop validation (3 electrodes)
  pid_params.csv       tuned kp, ki, kd for run_locked_closed_loop_from_config
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_META_ARX     = _PROJECT_ROOT / "meta_arx"
for _p in [str(_PROJECT_ROOT), str(_META_ARX)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Check V13 models are present
_MODELS = _HERE / "models"
_V15_FILES = [_MODELS / f"gp_el{i}_v15.pt" for i in (1, 2, 3)] + \
             [_MODELS / "arx_joint_v15.joblib"]

for _f in _V15_FILES:
    if not _f.exists():
        raise FileNotFoundError(
            f"V15 model not found: {_f}\n"
            "Run 'python fusion/train_gp_v15.py' first."
        )

print("[tune_pid] V15 models found. Starting PID tuning.")

OUT = _HERE / "results" / "pid_tuning_v15"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = ["C0", "C1", "C2"]

# Operating point
OP_R   = {1: 1.08, 2: 1.07, 3: 1.07}
OP_POS = {1: 1.04, 2: 1.03, 3: 1.04}
OP_KA  = 118.0
OP_RX  = 0.88
OP_V   = 165.0

STEP_MAG  = 0.06   # mOhm reference step for validation
WARMUP    = 100    # steps to settle before step
RESPONSE  = 300    # steps to observe after step
N_VAL     = 1500   # steps for closed-loop validation

# Step 1: Open-loop step response
def open_loop_step_test(el_active: int = 1) -> tuple[np.ndarray, np.ndarray, float]:
    """Move el_active by +0.05m at t=WARMUP, record R response.

    Returns (t, R_el_active - R_nom, pos_step_size).
    """
    from fusion.run_closed_loop import _build_sim_and_gps, _gp_corrected_r, _RollingFeatures

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim, gp_bundles, linear_models = _build_sim_and_gps("v15")

    # Apply V13 cross-R freeze
    import joblib
    _arx_bun = joblib.load(_MODELS / "arx_joint_v15.joblib")
    _rcm     = _arx_bun.get("r_cross_mean", {})

    plant_cache: dict = {}
    rolling     = _RollingFeatures()
    rolling.update(sim._row)
    rolling.inject(sim._row)

    pos_step = 0.05   # m
    u_now    = {i: float(OP_POS[i]) for i in (1, 2, 3)}

    R_trace = []
    u_trace = []

    total = WARMUP + RESPONSE
    for k in range(total):
        rolling.inject(sim._row)
        if _rcm:
            for _j in (1,2,3):
                _mj = _rcm.get(_j, 0.0)
                for _kk in (1,2,3):
                    _col = f"El{_j}_y_filt_lag{_kk}"
                    if _col in sim._row.index:
                        sim._row[_col] = _mj

        R_now = {}
        for i in (1, 2, 3):
            R_now[i], _, _, _ = _gp_corrected_r(
                sim, gp_bundles, i, plant_cache, step=k,
                linear_models=linear_models, gp_variant="v15")
        plant_cache["step"] = k

        # Apply step at WARMUP
        if k == WARMUP:
            u_now[el_active] = OP_POS[el_active] + pos_step

        R_trace.append(R_now[el_active])
        u_trace.append(u_now[el_active])

        y_arx_vec = {i: sim._predict_r(i) for i in (1, 2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec=u_now, y_new_vec=y_arx_vec)
        rolling.update(sim._row)
        for i in (1,2,3):
            plant_cache[f"r{i}_lag2"] = plant_cache.get(f"r{i}", y_arx_vec[i])
            plant_cache[f"r{i}"]      = y_arx_vec[i]

    R_arr  = np.array(R_trace)
    R_nom  = float(R_arr[:WARMUP].mean())
    dR     = R_arr - R_nom
    t      = np.arange(total) - WARMUP
    return t, dR, pos_step

print("\n[tune_pid] Running open-loop step test on El1 ...")
t_step, dR_step, pos_step = open_loop_step_test(el_active=1)

K_peak = float(dR_step[t_step >= 0].max()) / pos_step
t_peak = int(t_step[t_step >= 0][np.argmax(dR_step[t_step >= 0])])
print(f"  Peak gain K_peak = {K_peak:.4f} mOhm/m  t_peak = {t_peak} s")

# Step 2+3: Gain derivation for V14
# V14's delta ARX has a transient-only response: R peaks after a step then
# returns to its prior level (sustained gain is near zero). IMC from K_ss does
# not apply. Instead, derive gains from the relay parameters that are known to
# work. The relay fires step_size=0.01m every wait_normal=4 steps, giving an
# effective position rate of 0.0025 m/step when tracking within the deadband.
# For e_typical ~ 0.04 mOhm: kp = step_size / (wait_normal * e_typical).
# Sign: negative because the velocity-form PID with e = ref - y needs negative
# kp for a plant where higher position gives higher R.
_RELAY_STEP   = 0.01    # m
_RELAY_WAIT   = 4       # steps
_E_TYPICAL    = 0.04    # mOhm
_KP_BASE      = _RELAY_STEP / (_RELAY_WAIT * _E_TYPICAL)    # +0.0625

tunings = {
    "conservative": {"kp": _KP_BASE * 0.4,  "ki": _KP_BASE * 0.4  * 0.01, "kd": 0.0},
    "moderate":     {"kp": _KP_BASE * 0.8,  "ki": _KP_BASE * 0.8  * 0.01, "kd": 0.0},
    "aggressive":   {"kp": _KP_BASE * 1.5,  "ki": _KP_BASE * 1.5  * 0.01, "kd": 0.0},
}
for label, d in tunings.items():
    print(f"  {label:>12}:  kp={d['kp']:.5f}  ki={d['ki']:.6f}")

best_kp, best_ki = tunings["moderate"]["kp"], tunings["moderate"]["ki"]

# Plot step response
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

post_mask = t_step >= 0
t_pos = t_step[post_mask]
dR_pos = dR_step[post_mask]

ax1.axhline(0, color="grey", ls=":", lw=0.8)
ax1.plot(t_step, dR_step, color=COLORS[0], lw=1.5, label="El1 dR")
ax1.axvline(0, color="grey", ls="--", lw=0.8)
ax1.set_xlabel("Time after step (s)")
ax1.set_ylabel("dR from OP (mOhm)")
ax1.set_title(f"Open-loop step (+{pos_step*100:.0f} cm)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.95, f"K_peak={K_peak:.3f} mOhm/m\nt_peak={t_peak} s",
         transform=ax1.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round", fc="white", alpha=0.7))

ax2.axis("off")
rows = [["Tuning", "kp", "ki"]]
for lbl, d in tunings.items():
    rows.append([lbl, f"{d['kp']:.5f}", f"{d['ki']:.6f}"])
tbl = ax2.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)
ax2.set_title("PID gains (relay-derived)", pad=20)

fig.suptitle("V15 plant step response and PID gains", fontsize=11)
fig.tight_layout()
fig.savefig(str(OUT / "step_response.pdf"))
plt.close(fig)
print(f"\n[tune_pid] Step response plot: {OUT}/step_response.pdf")

# Step 4: Validation, locked PID on all 3 electrodes
print("\n[tune_pid] Running closed-loop PID validation with moderate tuning (V15) ...")

from fusion.run_locked_closed_loop import run_locked_closed_loop_from_config

# The cold-start state has all dpos lags = 0 which is outside the training
# distribution. warmup_hold_steps freezes all electrodes for the first N steps
# so the GP predictions can settle before the PID fires. After the hold, we
# read back the settled R and use it as the operating point for the step test.
WARMUP_HOLD = 150
STEP_VAL    = 0.02

# Run a single simulation: hold for WARMUP_HOLD steps, then PID for the rest.
# Reference during hold is the nominal OP; the PID steps begin after hold.
N = N_VAL + WARMUP_HOLD
ref = np.array([[OP_R[i+1]] * N for i in range(3)], dtype=float).T

# El1 reference step at PID step 400: +0.05 mOhm for 300 steps, then back
_STEP_START = WARMUP_HOLD + 400
_STEP_END   = WARMUP_HOLD + 700
ref[_STEP_START:_STEP_END, 0] += 0.05

pd.DataFrame({"r1": ref[:, 0], "r2": ref[:, 1], "r3": ref[:, 2]}).to_csv(
    OUT / "ref_val.csv", index=False)
pd.DataFrame({"kp": [best_kp], "ki": [best_ki], "kd": [0.0]}).to_csv(
    OUT / "pid_params.csv", index=False)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_full = run_locked_closed_loop_from_config(
        str(OUT / "ref_val.csv"),
        str(OUT / "pid_params.csv"),
        str(OUT / "validation_result_full.csv"),
        controller_name="pid",
        gp_variant="v15",
        deadband=0.0,
        warmup_hold_steps=WARMUP_HOLD,
        gp_scale=0.0,
    )

# Settled OP from the end of the hold phase
hold_end = df_full.iloc[WARMUP_HOLD - 10 : WARMUP_HOLD + 1]
op_settled = {1: float(hold_end["y1"].mean()),
              2: float(hold_end["y2"].mean()),
              3: float(hold_end["y3"].mean())}
print(f"[tune_pid] Settled after {WARMUP_HOLD}-step hold:  "
      f"El1={op_settled[1]:.4f}  El2={op_settled[2]:.4f}  El3={op_settled[3]:.4f} mOhm")

# Trim the warm-up from output for plotting and reporting
df_val = df_full.iloc[WARMUP_HOLD:].copy().reset_index(drop=True)
df_val["t_s"] = df_val["t_s"] - df_val["t_s"].iloc[0]
df_val.to_csv(OUT / "validation_result.csv", index=False)

# Plot validation
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
SETPOINTS = {1: OP_R[1], 2: OP_R[2], 3: OP_R[3]}
ref_trimmed_all = ref[WARMUP_HOLD:]   # shape (N_VAL, 3), trimmed to PID phase
t_val = df_val["t_s"].values
for el_idx, (el, ax) in enumerate(zip([1, 2, 3], axes), start=1):
    r_op = op_settled[el_idx]
    ref_el = ref_trimmed_all[:, el_idx - 1]
    # Step reference trajectory (prepend nan to align with y[0] initial row)
    ax.step(t_val, np.r_[np.nan, ref_el], color="k", ls="--", lw=1.0,
            where="post", label="ref")
    # ARX equilibrium marker (only if meaningfully different from base setpoint)
    if abs(r_op - SETPOINTS[el]) > 0.005:
        ax.axhline(r_op, color="grey", ls=":", lw=0.8, label=f"ARX eq={r_op:.3f}")
    y = pd.Series(df_val[f"y{el}"].values).rolling(10, center=True, min_periods=1).mean()
    ax.plot(t_val, y, color=COLORS[el_idx - 1], lw=1.5, label=f"El{el}")
    ax.set_ylabel("R (mOhm)")
    ax.set_title(f"Electrode {el}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")

tail = df_val.tail(300)
title_parts = []
for el in [1, 2, 3]:
    ref_tail = ref_trimmed_all[-300:, el - 1]
    rmse = ((tail[f"y{el}"].values - ref_tail) ** 2).mean() ** 0.5
    title_parts.append(f"El{el} RMSE={rmse:.3f}")
fig.suptitle(
    f"V15 locked PID validation (moderate: kp={best_kp:.4f} ki={best_ki:.5f})\n"
    + "  ".join(title_parts),
    fontsize=10,
)
fig.tight_layout()
fig.savefig(str(OUT / "validation.pdf"))
plt.close(fig)
print(f"[tune_pid] Validation plot: {OUT}/validation.pdf")

# Save all PID params for VRFT handoff
rows = []
for lbl, d in tunings.items():
    rows.append({"label": lbl, "kp": d["kp"], "ki": d["ki"], "kd": 0.0,
                 "K_peak_mOhm_per_m": K_peak})
pd.DataFrame(rows).to_csv(OUT / "all_tunings.csv", index=False)

print(f"\n[tune_pid] PID params saved: {OUT}/pid_params.csv")
print(f"[tune_pid] All tunings:       {OUT}/all_tunings.csv")
print(f"\nRecommended starting point for VRFT:")
print(f"  kp = {best_kp:.5f}  ki = {best_ki:.6f}  kd = 0.0")

# ── Relay comparison ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[tune_pid] Running relay comparisons ...")

# Relay params tuned for V15 dynamics (K_peak = 8.8 mOhm/m).
# step_size=0.005 m keeps each R step within the deadband;
# wait_normal=6 allows the 5-lag ARX to settle between steps.
RELAY_PARAMS = {"deadband": 0.05, "step_size": 0.005,
                "wait_normal": 6, "wait_escalated": 20, "escalation_count": 10}
pd.DataFrame([RELAY_PARAMS]).to_csv(OUT / "relay_params.csv", index=False)

relay_configs = [
    ("relay_v9",   "v9",   1.0, "Relay + V9 GP   "),
    ("relay_arx",  "v15",  0.0, "Relay + V15 ARX "),
    ("relay_gp",   "v15",  1.0, "Relay + V15 GP  "),
]

summary_rows = []

# PID result already computed - add to summary
pid_tail = df_val.tail(300)
for el, col, sp in [(1,"y1",1.08),(2,"y2",1.07),(3,"y3",1.07)]:
    ref_tail = ref_trimmed_all[-300:, el - 1]
    rmse = ((pid_tail[col].values - ref_tail)**2).mean()**0.5
    summary_rows.append({"config": "PID ARX-only", "electrode": el, "RMSE": rmse})

for tag, variant, scale, label in relay_configs:
    print(f"\n  {label} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_r = run_locked_closed_loop_from_config(
            str(OUT / "ref_val.csv"),
            str(OUT / "relay_params.csv"),
            str(OUT / f"{tag}_result_full.csv"),
            controller_name="relay",
            gp_variant=variant,
            warmup_hold_steps=WARMUP_HOLD,
            gp_scale=scale,
        )
    df_r_pid = df_r.iloc[WARMUP_HOLD:].reset_index(drop=True)
    tail_r = df_r_pid.tail(300)
    for el, col, sp in [(1,"y1",1.08),(2,"y2",1.07),(3,"y3",1.07)]:
        ref_tail = ref_trimmed_all[-300:, el - 1]
        rmse = ((tail_r[col].values - ref_tail)**2).mean()**0.5
        summary_rows.append({"config": label.strip(), "electrode": el, "RMSE": rmse})

# Print summary table
print("\n" + "="*60)
print(f"{'Configuration':<22} {'El1 RMSE':>10} {'El2 RMSE':>10} {'El3 RMSE':>10}")
print("-"*52)
configs_seen = []
for row in summary_rows:
    if row["config"] not in configs_seen:
        configs_seen.append(row["config"])
for cfg in configs_seen:
    el_rmse = {r["electrode"]: r["RMSE"] for r in summary_rows if r["config"] == cfg}
    print(f"{cfg:<22} {el_rmse[1]:>10.4f} {el_rmse[2]:>10.4f} {el_rmse[3]:>10.4f}")
print("="*60)

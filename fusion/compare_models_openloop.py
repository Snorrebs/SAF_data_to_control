"""
compare_models_openloop.py
Open-loop rollout comparison: V9+GP vs V15 ARX on historical relay data.

Runs N_WINDOWS random segments from the PI data forward using the historical
dpos inputs and compares predicted R against measured R at multiple horizons.
Quantifies which model better captures plant dynamics in relay-controlled
operating conditions.

Run with: python fusion/compare_models_openloop.py
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
for _p in [str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion.run_closed_loop import _build_sim_and_gps, _gp_corrected_r, _RollingFeatures
from fusion.train_gp_v15 import load_pi_data, PI_ROW_START, PI_ROW_END

ROLLOUT_H   = 200     # steps per evaluation window
N_WINDOWS   = 50      # number of random windows to evaluate
SEED        = 42
OUT         = _HERE / "results" / "model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

ELECTRODES  = [1, 2, 3]
VARIANTS    = [("v9", 1.0), ("v15", 0.0)]    # (gp_variant, gp_scale)
LABELS      = {"v9": "V9 + GP", "v15": "V15 ARX-only"}


def run_rollout(variant: str, gp_scale: float,
                df_seg: pd.DataFrame) -> dict[int, np.ndarray]:
    """Simulate ROLLOUT_H steps for one segment using historical dpos.
    Returns predicted R per electrode."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim, gp_bundles, linear_models = _build_sim_and_gps(variant)

    # Seed simulator state from the first row of the segment
    for col in sim._row.index:
        if col in df_seg.columns:
            sim._row[col] = float(df_seg[col].iloc[0])

    rolling = _RollingFeatures()
    rolling.update(sim._row)
    rolling.inject(sim._row)

    plant_cache: dict = {}
    r_pred = {el: [] for el in ELECTRODES}

    for t in range(min(ROLLOUT_H, len(df_seg) - 1)):
        for el in ELECTRODES:
            r, _, _, _ = _gp_corrected_r(
                sim, gp_bundles, el, plant_cache, step=t,
                linear_models=linear_models, gp_variant=variant,
                gp_scale=gp_scale)
            r_pred[el].append(r)
        plant_cache["step"] = t

        # Use historical dpos to advance (replay actual plant inputs)
        u_new = {}
        for el in ELECTRODES:
            pos_col = f"El{el}_pos_m_lag1"
            if pos_col in df_seg.columns and t + 1 < len(df_seg):
                u_new[el] = float(df_seg[pos_col].iloc[t + 1])
            else:
                u_new[el] = float(sim._row.get(f"El{el}_pos_m_lag1", 1.04))

        y_arx = {el: sim._predict_r(el) for el in ELECTRODES}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec=u_new, y_new_vec=y_arx)
        rolling.update(sim._row)
        rolling.inject(sim._row)
        for el in ELECTRODES:
            plant_cache[f"r{el}_lag2"] = plant_cache.get(f"r{el}", y_arx[el])
            plant_cache[f"r{el}"]      = y_arx[el]

    return {el: np.array(r_pred[el]) for el in ELECTRODES}


def main():
    print("Loading PI data ...")
    df = load_pi_data()
    n = len(df)

    rng = np.random.default_rng(SEED)
    starts = rng.choice(n - ROLLOUT_H - 10, N_WINDOWS, replace=False)

    # Accumulate errors per variant per electrode
    errors: dict[str, dict[int, list]] = {v: {el: [] for el in ELECTRODES}
                                           for v, _ in VARIANTS}

    print(f"Running {N_WINDOWS} windows x {ROLLOUT_H} steps ...")
    for wi, t0 in enumerate(starts):
        seg = df.iloc[t0: t0 + ROLLOUT_H + 1].reset_index(drop=True)
        r_true = {el: seg[f"El{el}_R_true"].values[:ROLLOUT_H]
                  for el in ELECTRODES}

        for variant, scale in VARIANTS:
            r_hat = run_rollout(variant, scale, seg)
            for el in ELECTRODES:
                length = min(len(r_hat[el]), len(r_true[el]))
                err = r_hat[el][:length] - r_true[el][:length]
                errors[variant][el].append(err)

        if (wi + 1) % 10 == 0:
            print(f"  {wi+1}/{N_WINDOWS} windows done")

    # Compute MAE and RMSE per horizon bucket
    horizons = [(1, 10), (11, 50), (51, 100), (101, 200)]
    print("\n=== Open-loop rollout comparison ===")
    print(f"{'Variant':<18} {'Electrode':<12} "
          + "  ".join(f"MAE h={a}-{b}" for a,b in horizons))
    print("-" * 80)

    summary = {}
    for variant, scale in VARIANTS:
        summary[variant] = {}
        for el in ELECTRODES:
            all_err = np.concatenate(errors[variant][el])   # flat
            mae_by_h = []
            for a, b in horizons:
                h_errs = np.concatenate([e[a-1:b] for e in errors[variant][el]
                                         if len(e) >= b])
                mae_by_h.append(np.abs(h_errs).mean())
            summary[variant][el] = mae_by_h
            print(f"{LABELS[variant]:<18} El{el:<10} "
                  + "  ".join(f"{m:.5f}" for m in mae_by_h))
        print()

    # Overall MAE (all steps, all electrodes)
    print("Overall MAE (all horizons, all electrodes):")
    for variant, scale in VARIANTS:
        all_err = np.concatenate([
            np.concatenate(errors[variant][el]) for el in ELECTRODES])
        print(f"  {LABELS[variant]:<18}: MAE={np.abs(all_err).mean():.5f}  "
              f"RMSE={np.sqrt((all_err**2).mean()):.5f} mOhm")

    # Plot MAE vs horizon
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    h_mids = [(a+b)//2 for a,b in horizons]
    colors = {"v9": "C1", "v15": "C0"}
    for el_idx, el in enumerate(ELECTRODES):
        ax = axes[el_idx]
        for variant, scale in VARIANTS:
            ax.plot(h_mids, summary[variant][el],
                    marker="o", label=LABELS[variant],
                    color=colors[variant], lw=1.5)
        ax.set_title(f"Electrode {el}")
        ax.set_xlabel("Rollout horizon (steps)")
        ax.set_ylabel("MAE (mOhm)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Open-loop rollout prediction accuracy\n"
        f"{N_WINDOWS} windows x {ROLLOUT_H} steps on historical relay data",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(str(OUT / "model_comparison.pdf"))
    plt.close(fig)
    print(f"\nPlot saved: {OUT}/model_comparison.pdf")


if __name__ == "__main__":
    main()

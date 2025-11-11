#!/usr/bin/env python3
# plot_pctrl_sim.py
#
# Plot ARX P-controller simulation results.
# Works with output from arx_p_controller_sim.py
#
# Example:
#   python plot_pctrl_sim.py --csv models/arx_pctrl_sim.csv --setpoint 0.65

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot ARX P-controller simulation output.")
    parser.add_argument("--csv", type=Path, required=True, help="Simulation result CSV (from arx_p_controller_sim.py)")
    parser.add_argument("--setpoint", type=float, default=None, help="Resistance setpoint (optional)")
    parser.add_argument("--save", type=Path, default=None, help="Optional output file (e.g. plots/sim_plot.png)")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    assert args.csv.exists(), f"Missing CSV: {args.csv}"

    df = pd.read_csv(args.csv, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    cols = df.columns

    # Auto-detect electrode position columns
    pos_cols = [c for c in cols if "El" in c and "pos" in c]
    if not pos_cols:
        raise ValueError("No electrode position columns found in CSV.")
    print(f"[info] Found position columns: {pos_cols}")

    # --- Plot setup ---
    fig, ax1 = plt.subplots(figsize=(10,6))
    t = (df.index - df.index[0]).total_seconds()

    # Resistance
    ax1.plot(t, df["y_pred_mOhm"], label="Predicted Resistance", color="tab:red", lw=2)
    if args.setpoint is not None:
        ax1.axhline(args.setpoint, color="k", ls="--", lw=1.5, label=f"Setpoint = {args.setpoint:.3f} mΩ")
    ax1.set_ylabel("Total Resistance [mΩ]", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Second y-axis for electrode positions
    ax2 = ax1.twinx()
    for c in pos_cols:
        ax2.plot(t, df[c], label=c, lw=1.5)
    ax2.set_ylabel("Electrode position [m]")
    ax2.grid(True, axis="both", alpha=0.3)

    # Legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    ax1.set_xlabel("Time [s]")
    ax1.set_title("ARX P-controller closed-loop simulation")

    fig.tight_layout()

    # Save or show
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=300)
        print(f"[save] {args.save}")
    if args.show or not args.save:
        plt.show()

if __name__ == "__main__":
    print("her")
    main()

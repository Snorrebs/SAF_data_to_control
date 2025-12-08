# run_simulation/scripts/plot_closed_loop_vrft_comparison.py
# Usage (from repo root):
#   python -m run_simulation.scripts.plot_closed_loop_vrft_comparison

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    base_csv = Path("run_simulation/closed_loop/closed_loop_sim.csv")
    vrft_csv = Path("run_simulation/closed_loop/closed_loop_sim_vrft_online.csv")

    if not base_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {base_csv}")
    if not vrft_csv.exists():
        raise FileNotFoundError(f"VRFT CSV not found: {vrft_csv}")

    df_base = pd.read_csv(base_csv)
    df_vrft = pd.read_csv(vrft_csv)

    # ---- 1) Output vs reference ----
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    ax = axs[0]
    ax.plot(df_base["t_s"], df_base["y_pred_mOhm"], label="Baseline y", alpha=0.9)
    ax.plot(df_vrft["t_s"], df_vrft["y_pred_mOhm"], label="VRFT y", alpha=0.9)
    # use ref from baseline (they should be identical)
    if "r" in df_base.columns:
        ax.plot(df_base["t_s"], df_base["r"], "k--", label="Reference r", alpha=0.7)
    ax.set_ylabel("y [mΩ]")
    ax.set_title("Closed-loop output vs reference")
    ax.grid(True)
    ax.legend()

    # ---- 2) Control signal ----
    ax = axs[1]
    ax.plot(df_base["t_s"], df_base["u_El1_pos_m"], label="Baseline u", alpha=0.9)
    ax.plot(df_vrft["t_s"], df_vrft["u_El1_pos_m"], label="VRFT u", alpha=0.9)
    ax.set_ylabel("u (electrode pos) [m]")
    ax.set_title("Control signal")
    ax.grid(True)
    ax.legend()

    # ---- 3) VRFT gains over time ----
    ax = axs[2]
    if all(c in df_vrft.columns for c in ["Kp", "Ki", "Kd"]):
        ax.plot(df_vrft["t_s"], df_vrft["Kp"], label="Kp")
        ax.plot(df_vrft["t_s"], df_vrft["Ki"], label="Ki")
        ax.plot(df_vrft["t_s"], df_vrft["Kd"], label="Kd")
        ax.set_ylabel("Gain value")
        ax.set_title("VRFT PID gains")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No Kp/Ki/Kd columns in VRFT CSV",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("time [s]")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

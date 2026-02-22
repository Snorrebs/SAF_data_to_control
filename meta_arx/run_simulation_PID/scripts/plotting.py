#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Adjust if needed
    CSV = Path("run_simulation_PID/history/closed_loop_sim.csv")

    assert CSV.exists(), f"Missing file: {CSV}"

    df = pd.read_csv(CSV)

    # --- time series ---
    t = df["t_s"]
    y = df["y_pred_mOhm"]
    r = df["r"]
    u = df["u_El2_pos_m"]
    e = df["e"]

    # --- figure ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Closed-loop ARX Simulation", fontsize=14)

    # --- resistance plot ---
    axs[0].plot(t, y, label="Resistance (pred)", lw=2)
    axs[0].plot(t, r, "--", label="Reference", lw=2)
    axs[0].set_ylabel("Resistance [mΩ]")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # --- control signal ---
    axs[1].plot(t, u, label="El2 position", color="C1", lw=2)
    axs[1].set_ylabel("Electrode Position [m]")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # --- error ---
    axs[2].plot(t, e, label="Error (r - y)", color="C2", lw=2)
    axs[2].set_ylabel("Error [mΩ]")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

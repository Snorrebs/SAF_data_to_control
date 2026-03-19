#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path("run_simulation/history/closed_loop_sim_test.csv")


def get_col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    raise KeyError(f"None of these columns were found: {names}")


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    t = get_col(df, "t_s")
    y = get_col(df, "y_pred")
    r = get_col(df, "reference", "r")
    u = get_col(df, "u_cmd")
    e = get_col(df, "error", "e")

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Closed-loop ARX Simulation", fontsize=14)

    axs[0].plot(t, y, label="Resistance (pred)", lw=2)
    axs[0].plot(t, r, "--", label="Reference", lw=2)
    axs[0].set_ylabel("Resistance [mΩ]")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, u, label="El1 position", lw=2)
    axs[1].set_ylabel("Electrode Position [m]")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t, e, label="Error (r - y)", lw=2)
    axs[2].set_ylabel("Error [mΩ]")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

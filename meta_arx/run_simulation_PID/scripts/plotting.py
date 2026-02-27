#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("run_simulation_PID/history/closed_loop_sim_varx.csv")


def _pick_cols(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    # time column (support old/new logs)
    t_col = _pick_cols(df, ["t_s", "t", "time_s"])
    t = df[t_col]

    # References: either r (scalar) or r1/r2/r3
    if all(c in df.columns for c in ["r1", "r2", "r3"]):
        r = df[["r1", "r2", "r3"]]
    elif "r" in df.columns:
        r = pd.DataFrame({"r1": df["r"], "r2": df["r"], "r3": df["r"]})
    else:
        r = None

    # Currents: prefer new naming, fallback to older single-output logs
    if all(c in df.columns for c in ["y1", "y2", "y3"]):
        y = df[["y1", "y2", "y3"]]
        y_label = "Current [kA]/Resistance [mOhm]"
        y_title = "Electrode currents (pred)"
    else:
        raise KeyError("Could not find current/resistance columns to plot.")

    # Positions: prefer u1/u2/u3, else u_cmd (legacy)
    if all(c in df.columns for c in ["u1", "u2", "u3"]):
        u = df[["u1", "u2", "u3"]]
    elif "u_cmd" in df.columns:
        u = pd.DataFrame({"u1": df["u_cmd"]})
    else:
        u = None

    # Errors: prefer e1/e2/e3, else e
    if all(c in df.columns for c in ["e1", "e2", "e3"]):
        e = df[["e1", "e2", "e3"]]
    elif "e" in df.columns:
        e = pd.DataFrame({"e1": df["e"]})
    else:
        e = None

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Closed-loop VARX Simulation", fontsize=14)

    # --- Y plot ---
    for col in y.columns:
        axs[0].plot(t, y[col], lw=2, label=col)
    if r is not None:
        for col in r.columns:
            axs[0].plot(t, r[col], "--", lw=2, label=f"{col} (ref)")
    axs[0].set_ylabel(y_label)
    axs[0].set_title(y_title)
    axs[0].legend(ncol=3)
    axs[0].grid(True, alpha=0.3)

    # --- U plot ---
    if u is not None:
        for col in u.columns:
            axs[1].plot(t, u[col], lw=2, label=col)
        axs[1].set_ylabel("Electrode Position [m]")
        axs[1].legend(ncol=3)
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5, 0.5, "No position columns found", ha="center", va="center")
        axs[1].axis("off")

    # --- Error plot ---
    if e is not None:
        for col in e.columns:
            axs[2].plot(t, e[col], lw=2, label=col)
        axs[2].set_ylabel("Error")
        axs[2].legend(ncol=3)
        axs[2].grid(True, alpha=0.3)
    else:
        axs[2].text(0.5, 0.5, "No error columns found", ha="center", va="center")
        axs[2].axis("off")

    axs[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
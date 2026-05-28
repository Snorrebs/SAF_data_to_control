#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("run_simulation/history/closed_loop_sim_varx.csv")


def _pick_cols(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def main(path: str | Path) -> None:
    CSV_PATH = Path(path)
    df = pd.read_csv(CSV_PATH)

    t_col = _pick_cols(df, ["t_s", "t", "time_s"])
    t = df[t_col]

    if all(c in df.columns for c in ["r1", "r2", "r3"]):
        r = df[["r1", "r2", "r3"]]
    elif "r" in df.columns:
        r = pd.DataFrame({"r1": df["r"], "r2": df["r"], "r3": df["r"]})
    else:
        r = None

    if all(c in df.columns for c in ["y1", "y2", "y3"]):
        y = df[["y1", "y2", "y3"]]
    else:
        raise KeyError("Could not find resistance columns to plot.")
    # t_s,y1,y2,y3,r1,r2,r3,du1,du2,du3,u1_cumsum,u2_cumsum,u3_cumsum,e1,e2,e3,kA1,kA2,kA3
    if all(c in df.columns for c in ["du1", "du2", "du3"]):
        u = df[["du1", "du2", "du3"]]
    elif "u_cmd" in df.columns:
        u = pd.DataFrame({"u1": df["u_cmd"]})
    else:
        u = None

    if all(c in df.columns for c in ["e1", "e2", "e3"]):
        e = df[["e1", "e2", "e3"]]
    elif "e" in df.columns:
        e = pd.DataFrame({"e1": df["e"]})
    else:
        e = None

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Closed-loop VARX Simulation — Resistance control", fontsize=14)

    # --- Resistance plot ---
    for col in y.columns:
        axs[0].plot(t, y[col], lw=2, label=col)
    if r is not None:
        for col in r.columns:
            axs[0].plot(t, r[col], "--", lw=2, label=f"{col} (ref)")
    axs[0].set_ylabel("Resistance [mΩ]")
    axs[0].set_title("Electrode resistances (pred)")
    axs[0].legend(ncol=3)
    axs[0].grid(True, alpha=0.3)

    # --- Position plot ---
    if u is not None:
        for col in u.columns:
            axs[1].plot(t, u[col], lw=2, label=col)
        axs[1].set_ylabel("Electrode speed [m/s]")
        axs[1].legend(ncol=3)
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5, 0.5, "No position columns found", ha="center", va="center")
        axs[1].axis("off")

    # # --- Error plot ---
    # if e is not None:
    #     for col in e.columns:
    #         axs[2].plot(t, e[col], lw=2, label=col)
    #     axs[2].set_ylabel("Error [mΩ]")
    #     axs[2].legend(ncol=3)
    #     axs[2].grid(True, alpha=0.3)
    # else:
    #     axs[2].text(0.5, 0.5, "No error columns found", ha="center", va="center")
    #     axs[2].axis("off")

    # --- Disturbance plot ---
    # run_mpc.py writes kA1, kA2, kA3; run_closed_loop.py writes v_transformer
    ka_cols = [c for c in ["kA1", "kA2", "kA3"] if c in df.columns]
    if ka_cols:
        colors = ["steelblue", "darkorange", "forestgreen"]
        for col, color in zip(ka_cols, colors):
            axs[2].plot(t, df[col], lw=1.5, color=color, label=col)
        axs[2].set_ylabel("Current [kA]")
        axs[2].legend(ncol=3)
        axs[2].grid(True, alpha=0.3)
    elif "v_transformer" in df.columns:
        axs[2].plot(t, df["v_transformer"], lw=1.5, color="steelblue", label="V transformer")
        axs[2].set_ylabel("Voltage [V]")
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
    else:
        axs[2].text(0.5, 0.5, "No disturbance column found (expected kA1/kA2/kA3 or v_transformer)",
                    ha="center", va="center", fontsize=9)
        axs[2].axis("off")

    axs[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(CSV_PATH.with_suffix(".png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main(path=CSV_PATH)
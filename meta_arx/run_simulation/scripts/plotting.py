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


def main(path: str | Path = CSV_PATH) -> None:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    t_col = _pick_cols(df, ["t_s", "t", "time_s"])
    t = df[t_col]

    # Absolute resistance outputs and references
    if all(c in df.columns for c in ["R_abs1", "R_abs2", "R_abs3"]):
        y = df[["R_abs1", "R_abs2", "R_abs3"]]
    else:
        raise KeyError("Could not find absolute resistance columns R_abs1..R_abs3.")

    if all(c in df.columns for c in ["R_ref_abs1", "R_ref_abs2", "R_ref_abs3"]):
        r = df[["R_ref_abs1", "R_ref_abs2", "R_ref_abs3"]]
    else:
        raise KeyError("Could not find absolute reference columns R_ref_abs1..R_ref_abs3.")

    # Control inputs
    if all(c in df.columns for c in ["dpos1", "dpos2", "dpos3"]):
        dpos = df[["dpos1", "dpos2", "dpos3"]]
    elif all(c in df.columns for c in ["du1", "du2", "du3"]):
        dpos = df[["du1", "du2", "du3"]]
    else:
        dpos = None

    # Errors
    if all(c in df.columns for c in ["e1", "e2", "e3"]):
        e = df[["e1", "e2", "e3"]]
    elif "e" in df.columns:
        e = pd.DataFrame({"e1": df["e"]})
    else:
        e = None

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Closed-loop VARX Simulation — Resistance control", fontsize=14)

    # --- Resistance plot ---
    colors = ["tab:blue", "tab:orange", "tab:green"]
    y_cols = ["R_abs1", "R_abs2", "R_abs3"]
    r_cols = ["R_ref_abs1", "R_ref_abs2", "R_ref_abs3"]

    for y_col, r_col, color in zip(y_cols, r_cols, colors):
        # Reference in background as dotted line
        axs[0].plot(
            t, r[r_col],
            linestyle=":",
            linewidth=2.5,
            color=color,
            alpha=0.7,
            zorder=1,
            label=f"{r_col} ref",
        )
        axs[0].plot(
            t, y[y_col],
            linewidth=2.0,
            color=color,
            zorder=2,
            label=y_col,
        )

    axs[0].set_ylabel("Resistance [mΩ]")
    axs[0].set_title("Absolute electrode resistances")
    axs[0].legend(ncol=3)
    axs[0].grid(True, alpha=0.3)

    # --- Movement plot ---
    if dpos is not None:
        for col in dpos.columns:
            axs[1].plot(t, dpos[col], lw=2, label=col)
        axs[1].set_ylabel("Holder movement [m/step]")
        axs[1].set_title("Electrode movement commands")
        axs[1].legend(ncol=3)
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5, 0.5, "No movement columns found", ha="center", va="center")
        axs[1].axis("off")

    # --- Current plot ---
    ka_cols = [c for c in ["kA1", "kA2", "kA3"] if c in df.columns]
    if ka_cols:
        for col in ka_cols:
            axs[2].plot(t, df[col], lw=1.8, label=col)
        axs[2].set_ylabel("Current [kA]")
        axs[2].set_title("Predicted electrode currents")
        axs[2].legend(ncol=3)
        axs[2].grid(True, alpha=0.3)
    elif e is not None:
        for col in e.columns:
            axs[2].plot(t, e[col], lw=1.8, label=col)
        axs[2].set_ylabel("Error [mΩ]")
        axs[2].set_title("Tracking errors")
        axs[2].legend(ncol=3)
        axs[2].grid(True, alpha=0.3)
    else:
        axs[2].text(0.5, 0.5, "No current or error columns found", ha="center", va="center")
        axs[2].axis("off")

    axs[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(csv_path.with_suffix(".png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main(path=CSV_PATH)
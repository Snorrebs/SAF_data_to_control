#!/usr/bin/env python3
"""
Check actuator-rate saturation in a closed-loop VARX/MPC simulation.

Expected CSV columns:
t_s,y1,y2,y3,r1,r2,r3,du1,du2,du3,u1_cumsum,u2_cumsum,u3_cumsum,e1,e2,e3,kA1,kA2,kA3

Example:
python check_saturation.py --csv run_mpc/history/closed_loop_sim.csv --du-limit 0.01
python run_simulation/history/check_satuartion.py --csv run_simulation/history/closed_loop_sim_varx.csv --du-limit 0.01 --plot
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DU_COLS = ["du1", "du2", "du3"]


def classify_activity(max_abs_du: float, du_limit: float, tol: float) -> str:
    """Return a short text verdict for one electrode."""
    if max_abs_du >= du_limit - tol:
        return "limit reached"
    elif max_abs_du >= 0.9 * du_limit:
        return "close to limit"
    else:
        return "not active"


def compute_saturation_summary(
    df: pd.DataFrame,
    du_limit: float,
    tol: float = 1e-9,
) -> pd.DataFrame:
    rows = []

    for col in DU_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

        du = df[col].astype(float).to_numpy()
        t = df["t_s"].astype(float).to_numpy() if "t_s" in df.columns else np.arange(len(du))

        abs_du = np.abs(du)

        # Saturated means the command is equal to the allowed boundary within tolerance.
        saturated = abs_du >= du_limit - tol

        # Near saturation can be useful if numerical optimisation gives values just below the bound.
        near_saturated_95 = abs_du >= 0.95 * du_limit
        near_saturated_90 = abs_du >= 0.90 * du_limit

        max_idx = int(np.nanargmax(abs_du))
        max_abs_du = float(abs_du[max_idx])

        n = int(np.sum(~np.isnan(du)))
        n_sat = int(np.sum(saturated))
        n_near_95 = int(np.sum(near_saturated_95))
        n_near_90 = int(np.sum(near_saturated_90))

        rows.append(
            {
                "electrode": col.replace("du", "El"),
                "max_abs_du_m_per_s": max_abs_du,
                "du_limit_m_per_s": du_limit,
                "max_abs_du_percent_of_limit": 100.0 * max_abs_du / du_limit,
                "time_of_max_s": float(t[max_idx]),
                "n_samples": n,
                "n_saturated_samples": n_sat,
                "percent_saturated": 100.0 * n_sat / n,
                "n_samples_above_95_percent_limit": n_near_95,
                "percent_above_95_percent_limit": 100.0 * n_near_95 / n,
                "n_samples_above_90_percent_limit": n_near_90,
                "percent_above_90_percent_limit": 100.0 * n_near_90 / n,
                "verdict": classify_activity(max_abs_du, du_limit, tol),
            }
        )

    return pd.DataFrame(rows)


def make_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    thesis_table = summary[
        [
            "electrode",
            "max_abs_du_m_per_s",
            "du_limit_m_per_s",
            "percent_saturated",
            "percent_above_95_percent_limit",
            "time_of_max_s",
            "verdict",
        ]
    ].copy()

    thesis_table = thesis_table.rename(
        columns={
            "electrode": "Electrode",
            "max_abs_du_m_per_s": r"$\max |\Delta u_i|$ [m/s]",
            "du_limit_m_per_s": r"Limit [m/s]",
            "percent_saturated": r"Saturated samples [\%]",
            "percent_above_95_percent_limit": r"Samples above 95\% limit [\%]",
            "time_of_max_s": r"Time of max [s]",
            "verdict": "Verdict",
        }
    )

    latex = thesis_table.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4g}",
        escape=False,
        caption=(
            "Actuator-rate saturation statistics for the closed-loop simulation. "
            "A sample is counted as saturated when "
            r"$|\Delta u_i|$ reaches the imposed rate limit."
        ),
        label="tab:closed_loop_saturation",
    )

    output_path.write_text(latex, encoding="utf-8")


def plot_du_with_limits(df: pd.DataFrame, du_limit: float, output_path: Path) -> None:
    if "t_s" in df.columns:
        t = df["t_s"]
    else:
        t = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 4))

    for col in DU_COLS:
        ax.plot(t, df[col], label=col)

    ax.axhline(du_limit, linestyle="--", linewidth=1.2, label="upper rate limit")
    ax.axhline(-du_limit, linestyle="--", linewidth=1.2, label="lower rate limit")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Electrode movement $\Delta u$ [m/s]")
    ax.set_title("Actuator-rate commands and imposed rate limits")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="Path to closed-loop simulation CSV.")
    parser.add_argument(
        "--du-limit",
        type=float,
        default=0.01,
        help="Actuator rate limit in m/s. Default: 0.01 m/s.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="Tolerance for detecting exact saturation.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a plot of du1/du2/du3 with rate limits.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same directory as input CSV.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_saturation_summary(df, du_limit=args.du_limit, tol=args.tol)

    summary_csv = out_dir / "saturation_summary.csv"
    latex_path = out_dir / "saturation_summary_latex.txt"

    summary.to_csv(summary_csv, index=False)
    make_latex_table(summary, latex_path)

    print("\nActuator-rate saturation summary")
    print("=" * 40)
    print(summary.to_string(index=False))

    print(f"\nSaved summary CSV to: {summary_csv}")
    print(f"Saved LaTeX table to: {latex_path}")

    max_overall = summary["max_abs_du_m_per_s"].max()
    percent_sat_overall = summary["percent_saturated"].max()

    print("\nThesis sentence suggestion:")
    if max_overall >= args.du_limit - args.tol:
        print(
            "The actuator-rate constraint becomes active during the simulation. "
            f"The largest command reaches {max_overall:.4g} m/s, equal to the imposed "
            f"limit of {args.du_limit:.4g} m/s, and the maximum saturation fraction "
            f"across electrodes is {percent_sat_overall:.2f}%."
        )
    else:
        print(
            "The actuator-rate constraint does not become active during the simulation. "
            f"The largest command is {max_overall:.4g} m/s, corresponding to "
            f"{100.0 * max_overall / args.du_limit:.1f}% of the imposed limit "
            f"of {args.du_limit:.4g} m/s."
        )

    if args.plot:
        plot_path = out_dir / "du_saturation_check.png"
        plot_du_with_limits(df, args.du_limit, plot_path)
        print(f"Saved saturation plot to: {plot_path}")


if __name__ == "__main__":
    main()
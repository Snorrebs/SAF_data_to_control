#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf

MODEL_NAME = "arx_el1res_5321_07"
PRED_CSV   = Path("arx/models/pred_csv") / f"{MODEL_NAME}.csv"

TRAIN_RATIO = 0.7  # same as ARX_fit


def main():
    df = pd.read_csv(PRED_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute residual
    df["residual_mOhm"] = df["y_true_mOhm"] - df["y_pred_mOhm"]

    # Train/test separation
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    df["split"] = "test"
    df.loc[:n_train-1, "split"] = "train"

    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)

    # Extract residual vector
    r = df_test["residual_mOhm"].values

    # --- ACF Plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(r, ax=ax, lags=40)
    ax.set_title("Residual ACF (Test Set)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

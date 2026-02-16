#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_NAME = "arx_el1res_2321_07"
PRED_CSV   = Path("arx/models/pred_csv") / f"{MODEL_NAME}.csv"
TRAIN_RATIO = 0.7

def main():
    df = pd.read_csv(PRED_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["k"] = df.index

    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    df["split"] = "test"
    df.loc[: n_train - 1, "split"] = "train"

    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
    df_test["k_test"] = df_test.index

    df_test["residual_mOhm"] = df_test["y_true_mOhm"] - df_test["y_pred_mOhm"]

    fig, axes = plt.subplots(figsize=(10, 7))
    ax = axes
    ax.hist(df_test["residual_mOhm"], bins=50, density=True, alpha=0.7)
    ax.set_title("Residual histogram (test set)")
    ax.set_xlabel("Residual [mΩ]")
    ax.set_ylabel("Density")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

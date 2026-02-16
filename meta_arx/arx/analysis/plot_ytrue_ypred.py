#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_NAME = "arx_el1res_2321_07"
PRED_CSV   = Path("arx/models/pred_csv") / f"{MODEL_NAME}.csv"

# how big a window to show (in samples)
WINDOW_LEN = 100  # e.g. 3000 s = 50 min


def main():
    df = pd.read_csv(PRED_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["k"] = df.index

    # if you want a simple train/test split by ratio:
    TRAIN_RATIO = 0.7
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    df["split"] = "test"
    df.loc[: n_train - 1, "split"] = "train"

    # focus on TEST region only, then take a window from the end
    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
    n_test = len(df_test)

    if n_test <= WINDOW_LEN:
        df_win = df_test
    else:
        df_win = df_test.iloc[-WINDOW_LEN:]  # last WINDOW_LEN samples of test

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df_win["timestamp"], df_win["y_true_mOhm"], label="True resistance", lw=1.0)
    ax.plot(df_win["timestamp"], df_win["y_pred_mOhm"], label="ARX predicted", lw=1.0)

    ax.set_title(f"One-step-ahead ARX prediction)")
    ax.set_ylabel("Resistance [mΩ]")
    ax.set_xlabel("Timestamp (test region)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    # plt.savefig(f"Bilder/ARX_identification_quality/arx_analysis_ytrue_ypred_{MODEL_NAME}.png", dpi=300)
    plt.show()
    


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
from pathlib import Path

MODEL = "arx_el1res_2321_07_hz01"   # <-- change to your model
MODEL_IN = "arx_el1res_15654_07_hz01"   # <-- change to your model
BUNDLE_PATH = Path("arx/models/model_meta") / f"{MODEL}.meta.joblib"
PREP_PATH   = Path("arx/arx_prep_data") / f"{MODEL_IN}.csv"

N_STEPS = 200   # free-run simulation length


def main():
    bundle = load(BUNDLE_PATH)
    df = pd.read_csv(PREP_PATH)

    # pick a starting row (from test set ideally)
    start_idx = int(len(df) * 0.7) + 5
    row = df.iloc[start_idx].copy()

    X_cols = bundle["X_cols"]
    X_scaler = bundle["X_scaler"]
    y_scaler = bundle["y_scaler"]
    model = bundle["model"]

    y_free = []
    y_true = []

    # initial truth for comparison
    y0 = row[bundle["y_col"]]
    y_free.append(y0)

    for k in range(N_STEPS):
        # Build feature vector
        x_raw = row[X_cols].to_numpy(dtype=float).reshape(1, -1)
        x_z = X_scaler.transform(x_raw)
        y_z = model.predict(x_z).reshape(-1, 1)
        y_hat = y_scaler.inverse_transform(y_z)[0, 0]

        y_free.append(y_hat)

        # shift state lags manually
        for col in X_cols:
            if "_lag" in col and col.startswith("y_filt"):
                lag = int(col.split("lag")[1])
                if lag == 1:
                    row[col] = y_hat
                else:
                    prev = f"y_filt_lag{lag-1}"
                    if prev in row:
                        row[col] = row[prev]

    # True measured future for comparison
    future = df.iloc[start_idx:start_idx+N_STEPS+1]
    y_true = future[bundle["y_col"]].values[:len(y_free)]

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(y_true, label="True y (future)")
    plt.plot(y_free, label="Free-run ARX prediction")
    plt.title("Multi-Step Free-Run Prediction (ARX)")
    plt.xlabel("Step")
    plt.ylabel("Resistance (mΩ)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

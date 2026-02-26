from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# --------------------------- CONFIG ---------------------------

MODEL_NAME_IN  = "varx_curr_2321_07"
MODEL_NAME_OUT = "varx_curr_2321_07_ridge"

META_PATH = Path("arx/arx_prep_meta") / f"{MODEL_NAME_IN}.meta.joblib"
IN_CSV    = Path("arx/arx_prep_data") / f"{MODEL_NAME_IN}.csv"

MODEL_OUT    = Path("arx/models/model_meta") / f"{MODEL_NAME_OUT}.meta.joblib"
COEF_CSV_OUT = Path("arx/models/model_coef") / f"{MODEL_NAME_OUT}.csv"
PRED_CSV_OUT = Path("arx/models/pred_csv") / f"{MODEL_NAME_OUT}.csv"

TRAIN_FRAC = 0.7

# RidgeCV hyperparameters
USE_RIDGECV = True
ALPHAS    = np.logspace(-4, 4, 25)
TS_SPLITS = 5
SEED      = 0

# If you want fixed ridge instead of CV:
RIDGE_ALPHA = 1.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ---------- load data + meta ----------
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta file not found: {META_PATH}")

    df = pd.read_csv(IN_CSV, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    meta: Dict[str, Any] = load(META_PATH)

    # NEW: multi-output
    y_cols: List[str] = meta.get("y_cols", [])
    X_cols: List[str] = meta.get("X_cols", [])

    if not y_cols:
        raise KeyError("meta['y_cols'] missing or empty in meta file")
    if not X_cols:
        raise KeyError("meta['X_cols'] missing or empty in meta file")

    missing_cols = [c for c in X_cols + y_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns missing from input CSV: {missing_cols}")

    print(f"[info] Loaded {len(df)} samples from {IN_CSV}")
    print(f"[info] Targets (Y): {y_cols}")
    print(f"[info] Features (X): {len(X_cols)} columns")

    # drop rows with NaNs in Y or X
    df = df.dropna(subset=y_cols + X_cols)
    print(f"[info] After dropna: {len(df)} samples")
    if len(df) < 50:
        raise RuntimeError("Too few samples after dropna for VARX training.")

    # ---------- chronological train/test split ----------
    n = len(df)
    split_idx = int(n * TRAIN_FRAC)
    if split_idx <= 0 or split_idx >= n:
        raise RuntimeError("Invalid TRAIN_FRAC; gives empty train or test split.")

    df_tr = df.iloc[:split_idx].copy()
    df_te = df.iloc[split_idx:].copy()
    print(f"[split] Train: {len(df_tr)} samples, Test: {len(df_te)} samples")

    # ---------- build raw arrays ----------
    Y_tr_raw = df_tr[y_cols].to_numpy(dtype=np.float64)  # (N, 3)
    Y_te_raw = df_te[y_cols].to_numpy(dtype=np.float64)

    X_tr_raw = df_tr[X_cols].to_numpy(dtype=np.float64)
    X_te_raw = df_te[X_cols].to_numpy(dtype=np.float64)

    # ---------- standardize ----------
    # 1) scale Y (multi-column)
    Y_scaler = StandardScaler()
    Y_tr_z = Y_scaler.fit_transform(Y_tr_raw)
    Y_te_z = Y_scaler.transform(Y_te_raw)

    # 2) drop zero-variance features on training set, then scale X
    stds_raw = X_tr_raw.std(axis=0)
    keep_mask = stds_raw > 0.0
    if not np.any(keep_mask):
        raise RuntimeError("All features have zero variance; cannot train VARX.")

    X_tr_raw_keep = X_tr_raw[:, keep_mask]
    X_te_raw_keep = X_te_raw[:, keep_mask]
    X_cols_final = [c for c, k in zip(X_cols, keep_mask) if k]

    X_scaler = StandardScaler()
    X_tr_z = X_scaler.fit_transform(X_tr_raw_keep)
    X_te_z = X_scaler.transform(X_te_raw_keep)

    print(f"[info] Kept {len(X_cols_final)} / {len(X_cols)} features after variance filter")

    # ---------- fit model ----------
    if USE_RIDGECV:
        print(f"[fit] Fitting RidgeCV with {len(ALPHAS)} alphas, {TS_SPLITS} time-series splits")
        tscv = TimeSeriesSplit(n_splits=TS_SPLITS)
        model = RidgeCV(
            alphas=ALPHAS,
            cv=tscv,
            scoring="neg_mean_squared_error",
        )
    else:
        print(f"[fit] Fitting Ridge(alpha={RIDGE_ALPHA})")
        model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)

    model.fit(X_tr_z, Y_tr_z)  # multi-output y is 2D
    print(f"[fit] Best alpha: {getattr(model, 'alpha_', None)}")

    # ---------- predictions ----------
    Yhat_tr_z = model.predict(X_tr_z)  # (N, 3)
    Yhat_te_z = model.predict(X_te_z)

    # back to physical units
    Yhat_tr = Y_scaler.inverse_transform(Yhat_tr_z)
    Yhat_te = Y_scaler.inverse_transform(Yhat_te_z)

    # ---------- metrics (per-electrode + overall) ----------
    rmses_tr = [rmse(Y_tr_raw[:, i], Yhat_tr[:, i]) for i in range(Y_tr_raw.shape[1])]
    rmses_te = [rmse(Y_te_raw[:, i], Yhat_te[:, i]) for i in range(Y_te_raw.shape[1])]

    for i, c in enumerate(y_cols):
        print(f"[metrics] {c}: Train RMSE = {rmses_tr[i]:.6f} | Test RMSE = {rmses_te[i]:.6f} (kA)")

    overall_tr = rmse(Y_tr_raw.ravel(), Yhat_tr.ravel())
    overall_te = rmse(Y_te_raw.ravel(), Yhat_te.ravel())
    print(f"[metrics] Overall: Train RMSE = {overall_tr:.6f} | Test RMSE = {overall_te:.6f} (kA, stacked)")

    # ---------- (optional) analyze AR poles per output (approx) ----------
    # NOTE: In multi-output Ridge, coef_ is shape (n_targets, n_features)
    # We estimate AR poles separately for each output using the coefficients
    # corresponding to that output’s current-lag features.
    coef_ = getattr(model, "coef_", None)
    if coef_ is not None and coef_.ndim == 2:
        for yi, yname in enumerate(y_cols):
            ar_terms = []
            for xname, coef_val in zip(X_cols_final, coef_[yi]):
                # current lags are named like "El1_kA_filt_lag1" etc. in your prep script
                if "_kA_filt_lag" in xname:
                    try:
                        lag = int(xname.split("lag")[-1])
                        ar_terms.append((lag, float(coef_val)))
                    except Exception:
                        pass
            if ar_terms:
                ar_terms.sort(key=lambda x: x[0])
                ar_coefs = np.array([c for _, c in ar_terms], dtype=float)
                ar_poly = np.r_[1.0, -ar_coefs]
                poles = np.roots(ar_poly)
                print(f"[ar] {yname}: AR coeffs (sorted) len={len(ar_coefs)} | poles={poles}")

    # ---------- save model bundle ----------
    ensure_parent(MODEL_OUT)
    bundle = {
        "model_name": MODEL_NAME_OUT,
        "y_cols": y_cols,
        "X_cols": X_cols_final,
        "meta_prep": meta,
        "model": model,
        "Y_scaler": Y_scaler,
        "X_scaler": X_scaler,
        "train_index": df_tr.index,
        "test_index": df_te.index,
        "train_rmse_per_y": dict(zip(y_cols, rmses_tr)),
        "test_rmse_per_y": dict(zip(y_cols, rmses_te)),
        "train_rmse_overall": overall_tr,
        "test_rmse_overall": overall_te,
    }
    dump(bundle, MODEL_OUT)
    print(f"[save] Model bundle -> {MODEL_OUT}")

    # ---------- save coefficients (z-space) ----------
    intercept_ = getattr(model, "intercept_", None)

    # Store coef matrix with y as rows, X as columns
    coef_df = pd.DataFrame(coef_, index=y_cols, columns=X_cols_final)

    if intercept_ is not None:
        # intercept_ is (n_targets,) for multi-output
        coef_df.insert(0, "_intercept_z", np.asarray(intercept_, dtype=float))

    ensure_parent(COEF_CSV_OUT)
    coef_df.to_csv(COEF_CSV_OUT)
    print(f"[save] Coefficients (z-space) -> {COEF_CSV_OUT}")

    # ---------- save predictions (physical units, test set) ----------
    pred_df = pd.DataFrame(index=df_te.index)
    for i, c in enumerate(y_cols):
        pred_df[f"{c}_true"] = Y_te_raw[:, i]
        pred_df[f"{c}_pred"] = Yhat_te[:, i]

    ensure_parent(PRED_CSV_OUT)
    pred_df.to_csv(PRED_CSV_OUT)
    print(f"[save] Test predictions -> {PRED_CSV_OUT}")


if __name__ == "__main__":
    main()
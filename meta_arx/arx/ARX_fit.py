from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# --------------------------- CONFIG ---------------------------

MODEL_NAME_IN     = "arx_el1res_0321_07_pos"
MODEL_NAME_OUT = "arx_el1res_0321_07_pos"
META_PATH      = Path("arx/arx_prep_meta") / f"{MODEL_NAME_IN}.meta.joblib"
IN_CSV         = Path("arx/arx_prep_data") / f"{MODEL_NAME_IN}.csv"

MODEL_OUT      = Path("arx/models/model_meta") / f"{MODEL_NAME_OUT}.meta.joblib"
COEF_CSV_OUT   = Path("arx/models/model_coef") / f"{MODEL_NAME_OUT}.csv"
PRED_CSV_OUT   = Path("arx/models/pred_csv") / f"{MODEL_NAME_OUT}.csv"

# train/test split (chronological)
TRAIN_FRAC     = 0.7

# RidgeCV hyperparameters
alpha = 0.1
ALPHAS         = np.logspace(-4, 4, 20)
TS_SPLITS      = 5
SEED           = 0  # for any stochastic parts (not critical here)


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

    y_col: str = meta.get("y_col")
    X_cols: List[str] = meta.get("X_cols", [])

    if y_col is None:
        raise KeyError("meta['y_col'] missing in meta file")
    if not X_cols:
        raise KeyError("meta['X_cols'] missing or empty in meta file")

    missing_cols = [c for c in X_cols + [y_col] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns missing from input CSV: {missing_cols}")

    print(f"[info] Loaded {len(df)} samples from {IN_CSV}")
    print(f"[info] Target (y): {y_col}")
    print(f"[info] Features (X): {len(X_cols)} columns")

    # drop rows with NaNs in y or X
    df = df.dropna(subset=[y_col] + X_cols)
    print(f"[info] After dropna: {len(df)} samples")

    if len(df) < 10:
        raise RuntimeError("Too few samples after dropna for ARX training.")

    # ---------- chronological train/test split ----------
    n = len(df)
    split_idx = int(n * TRAIN_FRAC)
    if split_idx <= 0 or split_idx >= n:
        raise RuntimeError("Invalid TRAIN_FRAC; gives empty train or test split.")

    df_tr = df.iloc[:split_idx].copy()
    df_te = df.iloc[split_idx:].copy()

    print(f"[split] Train: {len(df_tr)} samples, Test: {len(df_te)} samples")

    # ---------- build raw arrays ----------
    y_tr_raw = df_tr[y_col].to_numpy(dtype=np.float64)
    y_te_raw = df_te[y_col].to_numpy(dtype=np.float64)

    X_tr_raw = df_tr[X_cols].to_numpy(dtype=np.float64)
    X_te_raw = df_te[X_cols].to_numpy(dtype=np.float64)

    # ---------- standardize ----------
    # 1) scale y (target)
    y_scaler = StandardScaler()
    y_tr_z = y_scaler.fit_transform(y_tr_raw.reshape(-1, 1)).ravel()
    y_te_z = y_scaler.transform(y_te_raw.reshape(-1, 1)).ravel()

    # 2) drop features with zero variance on training set, then scale X
    stds_raw = X_tr_raw.std(axis=0)
    keep_mask = stds_raw > 0.0
    if not np.any(keep_mask):
        raise RuntimeError("All features have zero variance; cannot train ARX.")

    X_tr_raw_keep = X_tr_raw[:, keep_mask]
    X_te_raw_keep = X_te_raw[:, keep_mask]
    X_cols_final = [c for c, k in zip(X_cols, keep_mask) if k]

    X_scaler = StandardScaler()
    X_tr_z = X_scaler.fit_transform(X_tr_raw_keep)
    X_te_z = X_scaler.transform(X_te_raw_keep)

    print(f"[info] Kept {len(X_cols_final)} / {len(X_cols)} features after variance filter")

    # ---------- fit RidgeCV on (X_z → y_z) ----------
    print(f"[fit] Fitting RidgeCV with {len(ALPHAS)} alphas, {TS_SPLITS} time-series splits")
    tscv = TimeSeriesSplit(n_splits=TS_SPLITS)
    # model = RidgeCV(
    #     alphas=ALPHAS,
    #     cv=tscv,
    #     scoring="neg_mean_squared_error",
    #     gcv_mode="svd",
    # )
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(X_tr_z, y_tr_z)
    print(f"[fit] Best alpha: {getattr(model, 'alpha_', None)}")

    # ---------- predictions ----------
    yhat_tr_z = model.predict(X_tr_z)
    yhat_te_z = model.predict(X_te_z)

    # back to physical units
    yhat_tr = y_scaler.inverse_transform(yhat_tr_z.reshape(-1, 1))[:, 0]
    yhat_te = y_scaler.inverse_transform(yhat_te_z.reshape(-1, 1))[:, 0]

    # ---------- metrics ----------
    tr_rmse = rmse(y_tr_raw, yhat_tr)
    te_rmse = rmse(y_te_raw, yhat_te)

    print(f"[metrics] Train RMSE = {tr_rmse:.6f} (physical units)")
    print(f"[metrics] Test  RMSE = {te_rmse:.6f} (physical units)")

    # ---------- analyze AR poles ----------
    ar_terms = []
    for name, coef in zip(X_cols_final, model.coef_):   # <-- IMPORTANT: X_cols_final
        if "y_filt_lag" in name:
            lag = int(name.split("lag")[-1])
            ar_terms.append((lag, coef))
            

    ar_terms.sort(key=lambda x: x[0])                  # <-- IMPORTANT: sort by lag
    ar_coefs = np.array([c for _, c in ar_terms], dtype=float)

    ar_poly = np.r_[1.0, -ar_coefs]                    # 1 - a1 z^-1 - a2 z^-2 - ...
    poles = np.roots(ar_poly)
    print("AR coefficients (sorted):", ar_coefs)
    print("AR poles:", poles)
    coef = model.coef_
    print(coef)
    u_idx = np.array([("El1_pos_m_lag" in c) for c in X_cols_final], dtype=bool)
    y_idx = np.array([("y_filt_lag" in c) for c in X_cols_final], dtype=bool)

    u_norm = float(np.linalg.norm(coef[u_idx])) if u_idx.any() else 0.0
    y_norm = float(np.linalg.norm(coef[y_idx])) if y_idx.any() else 0.0
    all_norm = float(np.linalg.norm(coef))

    print(f"[coef] ||coef_u|| = {u_norm:.4f}, ||coef_y|| = {y_norm:.4f}, ||coef_all|| = {all_norm:.4f}")
    # ---------- save model bundle ----------
    ensure_parent(MODEL_OUT)
    bundle = {
        "model_name": MODEL_NAME_OUT,
        "y_col": y_col,
        "X_cols": X_cols_final,
        "meta_prep": meta,
        "model": model,
        "y_scaler": y_scaler,
        "X_scaler": X_scaler,
        "train_index": df_tr.index,
        "test_index": df_te.index,
        "train_rmse": tr_rmse,
        "test_rmse": te_rmse,
    }
    dump(bundle, MODEL_OUT)
    print(f"[save] Model bundle -> {MODEL_OUT}")

    # ---------- save coefficients (z-space) ----------
    coef_z = getattr(model, "coef_", None)
    intercept_z = getattr(model, "intercept_", 0.0)

    coef_df = pd.DataFrame(
        {
            "coef_z": coef_z,
        },
        index=X_cols_final,
    )
    coef_df.loc["_intercept_z"] = intercept_z

    ensure_parent(COEF_CSV_OUT)
    coef_df.to_csv(COEF_CSV_OUT)
    print(f"[save] Coefficients (z-space) -> {COEF_CSV_OUT}")

    # ---------- save predictions (physical units, test set only) ----------
    pred_df = pd.DataFrame(
        {
            "timestamp": df_te.index,
            "y_true_mOhm": y_te_raw,
            "y_pred_mOhm": yhat_te,
        }
    ).set_index("timestamp")

    ensure_parent(PRED_CSV_OUT)
    pred_df.to_csv(PRED_CSV_OUT)
    
    print(f"[save] Test predictions -> {PRED_CSV_OUT}")


if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import TimeSeriesSplit

# --------------------------- CONFIG ---------------------------
META_PATH     = Path("arx/arx_prep/model_arx_scalers_30_5_5.meta.joblib")  # from prep
IN_CSV        = Path("arx/arx_prep/model_arx_30_5_5.csv")                  # from data_processing

MODEL_OUT     = Path("models/arx_linear_ridge.joblib")
COEF_CSV_OUT  = Path("models/arx_linear_ridge_coefficients_zspace.csv")
PRED_CSV_OUT  = Path("models/arx_linear_ridge_predictions.csv")

TRAIN_FRAC    = 0.80
USE_CV        = True
ALPHAS        = np.logspace(-6, 6, 25)
RIDGE_ALPHA   = 1.0
TS_SPLITS     = 5
SEED          = 7

# Guards
VAR_EPS       = 1e-6
CORR_CUTOFF   = 0.99  # features with |corr(X, y)| above this on TRAIN are dropped. For testing leakage.
MIN_SAFE_GAP  = 60    # extra safety against filter/feature window bleed

# --------------------------------------------------------------

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def main():
    np.random.seed(SEED)

    assert IN_CSV.exists(), f"Missing dataset: {IN_CSV}"
    assert META_PATH.exists(), f"Missing meta: {META_PATH}"

    df   = pd.read_csv(IN_CSV, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    meta = load(META_PATH)
    df["target_time"] = pd.to_datetime(df["target_time"], errors="coerce")
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    if df["target_time"].dt.tz is None:
        df["target_time"] = df["target_time"].dt.tz_localize("UTC")
    else:
        df["target_time"] = df["target_time"].dt.tz_convert("UTC")

    df = df.dropna(subset=["target_time"])

    y_col      = meta["y_col"]        # "y_target" (H>0) or "Tot_Resistance_mOhm_filt" (H=0)
    X_cols_all = meta["X_cols"]
    cfg        = meta.get("config", {})
    H          = int(cfg.get("horizon", 0))
    max_ar     = int(cfg.get("max_ar_lag", 0))
    max_x      = int(cfg.get("max_x_lag", 0))

    # basic sanity
    need_cols = ["target_time", y_col] + X_cols_all
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in IN_CSV: {missing}")

    #should be clean already
    df = df.dropna(subset=need_cols).copy()

    # ---------- Time split with GAP + boundary trim ----------
    n = len(df)
    split_idx = int(TRAIN_FRAC * n)

    # a safe temporal gap: H (label lead) + max lag window + manual cushion
    gap_feat = max(max_ar, max_x)
    gap = max(H + gap_feat, MIN_SAFE_GAP)

    # indices
    train_end_idx = df.index[split_idx - 1]
    te_start_idx = split_idx + gap
    if te_start_idx >= n:
        raise RuntimeError(f"Empty test set: split={split_idx}, gap={gap}, n={n}. "
                           f"Reduce TRAIN_FRAC or gap, or use more data.")

    # raw chunks
    df_tr = df.iloc[:split_idx].copy()
    df_te = df.iloc[te_start_idx:].copy()
    if len(df_te) == 0:
        raise RuntimeError(f"Empty test set after applying gap={gap}.")

    # trim train rows whose labels occur after the train window
    df_tr = df_tr[df_tr["target_time"] <= train_end_idx]
    if len(df_tr) == 0:
        raise RuntimeError("All training rows trimmed by target_time; decrease H/gap or increase TRAIN_FRAC/data.")

    # ---------- Build train/test arrays ----------
    X_cols = list(X_cols_all)  # will shrink with guards
    ytr = df_tr[[y_col]].to_numpy(dtype=np.float64).ravel()
    yte = df_te[[y_col]].to_numpy(dtype=np.float64).ravel()

    Xtr_raw = df_tr[X_cols].to_numpy(dtype=np.float64)
    Xte_raw = df_te[X_cols].to_numpy(dtype=np.float64)

    # ---------- Train-only scaling ----------
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr = X_scaler.fit_transform(Xtr_raw)
    Xte = X_scaler.transform(Xte_raw)

    ytr_z = y_scaler.fit_transform(ytr.reshape(-1, 1)).ravel()
    yte_z = y_scaler.transform(yte.reshape(-1, 1)).ravel()

    # ---------- Guards (computed on TRAIN) ----------
    # 1) Variance filter
    var = Xtr.var(axis=0)
    keep = var > VAR_EPS
    if not np.any(keep):
        raise RuntimeError("All features removed by variance threshold. Lower VAR_EPS or inspect inputs.")
    X_cols = [c for c, k in zip(X_cols, keep) if k]
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]

    # 2) High |corr| with y (suspicious signal — often leakage or order mismatch). Check for leakage.
    if Xtr.shape[1] > 1:
        # corr(X_i, y) across train rows
        corrs = np.corrcoef(np.c_[Xtr, ytr_z].T)[-1, :-1]
        suspicious = np.where(np.abs(corrs) > CORR_CUTOFF)[0]
        if suspicious.size > 0:
            suspicious_cols = [X_cols[i] for i in suspicious]
            print(f"[guard] Dropping {len(suspicious_cols)} high-|corr| features (>|{CORR_CUTOFF}|) vs y on TRAIN: "
                  f"{suspicious_cols[:8]}{'...' if len(suspicious_cols) > 8 else ''}")
            mask = np.ones(len(X_cols), dtype=bool)
            mask[suspicious] = False
            X_cols = [c for c, m in zip(X_cols, mask) if m]
            Xtr, Xte = Xtr[:, mask], Xte[:, mask]

    # ---------- Fit ----------
    if USE_CV:
        tscv = TimeSeriesSplit(n_splits=TS_SPLITS)
        model = RidgeCV(alphas=ALPHAS, cv=tscv, scoring="neg_mean_squared_error", gcv_mode="svd")
    else:
        model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)

    model.fit(Xtr, ytr_z)
    if USE_CV:
        print(f"[ridge] Chosen alpha = {getattr(model, 'alpha_', None)}")

    # ---------- Predict ----------
    yhat_tr_z = model.predict(Xtr)
    yhat_te_z = model.predict(Xte)

    # Reverse scaling
    yhat_tr = y_scaler.inverse_transform(yhat_tr_z.reshape(-1, 1))[:, 0]
    yhat_te = y_scaler.inverse_transform(yhat_te_z.reshape(-1, 1))[:, 0]

    tr_rmse = rmse(ytr, yhat_tr)
    te_rmse = rmse(yte, yhat_te)
    tr_r2   = r2_score(ytr, yhat_tr)
    te_r2   = r2_score(yte, yhat_te)

    print(f"[Train] RMSE={tr_rmse:.6f} mΩ, R²={tr_r2:.3f}")
    print(f"[ Test] RMSE={te_rmse:.6f} mΩ, R²={te_r2:.3f}")
    print(f"[Info ] Samples: train={len(df_tr)}, gap={gap}, test={len(df_te)}")
    print(f"[Info ] Kept features: {len(X_cols)}")

    # ---------- Save ----------
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "X_cols": X_cols,
        "y_col": y_col,
        "alpha": getattr(model, "alpha_", getattr(model, "alpha", None)),
        "train_frac": TRAIN_FRAC,
        "seed": SEED,
        "gap": gap,
        "ts_splits": TS_SPLITS,
        "var_eps": VAR_EPS,
        "corr_cutoff": CORR_CUTOFF,
        "scalers": {
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
        },
        "prep_config": cfg,
    }
    dump(bundle, MODEL_OUT)
    print(f"[save] Model bundle -> {MODEL_OUT}")

    # Coefficients in z-space (standardized feature space)
    coef = np.asarray(model.coef_, dtype=float).ravel()
    coef_df = (pd.Series(coef, index=X_cols, name="coef_zspace")
                 .sort_values(key=np.abs, ascending=False)
                 .to_frame())
    coef_df.loc["(intercept)", "coef_zspace"] = float(model.intercept_)
    coef_df.to_csv(COEF_CSV_OUT)
    print(f"[save] Coefficients (z-space) -> {COEF_CSV_OUT}")

    # Test predictions (physical units)
    out_df = pd.DataFrame({
        "timestamp": df_te.index,
        "y_true_mOhm": yte,
        "y_pred_mOhm": yhat_te,
    }).set_index("timestamp")
    out_df.to_csv(PRED_CSV_OUT)
    print(f"[save] Predictions -> {PRED_CSV_OUT}")

if __name__ == "__main__":
    main()

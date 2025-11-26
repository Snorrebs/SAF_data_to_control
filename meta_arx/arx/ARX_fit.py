from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import TimeSeriesSplit

# --------------------------- CONFIG ---------------------------

META_PATH     = Path("arx/arx_prep/model_arx_scalers_1_5_5.meta.joblib")
IN_CSV        = Path("arx/arx_prep/model_arx_1_5_5.csv")

MODEL_OUT     = Path("models/arx_linear_ridge_stable_yonly.joblib")
COEF_CSV_OUT  = Path("models/arx_linear_ridge_stable_yonly_coefficients_zspace.csv")
PRED_CSV_OUT  = Path("models/arx_linear_ridge_stable_yonly_predictions.csv")

TRAIN_FRAC    = 0.80
USE_CV        = True
ALPHAS        = np.logspace(-6, 6, 25)
RIDGE_ALPHA   = 1.0
TS_SPLITS     = 5
SEED          = 7

VAR_EPS       = 1e-6
CORR_CUTOFF   = 1.0
MIN_SAFE_GAP  = 60


# --------------------------- utils ---------------------------

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def stabilize_ar_coeffs(a, eps=1e-3):
    """
    Stabilize AR coeffs for:
        y_t = sum a_i y_{t-i}
    by reflecting roots of 1 - sum a_i z^{-i} inside the unit circle.
    """
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        return a
    poly = np.r_[1.0, -a]
    roots = np.roots(poly)
    changed = False
    for i, r in enumerate(roots):
        mag = np.abs(r)
        if mag >= 1.0:
            roots[i] = r / (mag + eps)
            changed = True
    if not changed:
        return a
    poly_stab = np.poly(roots)
    a_stab = -poly_stab[1:].real
    return a_stab


# --------------------------- main ---------------------------

def main():
    np.random.seed(SEED)

    assert IN_CSV.exists(), f"Missing dataset: {IN_CSV}"
    assert META_PATH.exists(), f"Missing meta: {META_PATH}"

    df   = pd.read_csv(IN_CSV, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    meta = load(META_PATH)

    df["target_time"] = pd.to_datetime(df["target_time"], errors="coerce")
    # normalize timezones
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    if df["target_time"].dt.tz is None:
        df["target_time"] = df["target_time"].dt_localize("UTC")
    else:
        df["target_time"] = df["target_time"].dt.tz_convert("UTC")

    df = df.dropna(subset=["target_time"])

    y_col      = meta["y_col"]
    X_cols_all = meta["X_cols"]
    cfg        = meta.get("config", {})
    H          = int(cfg.get("horizon", 0))
    max_ar     = int(cfg.get("max_ar_lag", 0))
    max_x      = int(cfg.get("max_x_lag", 0))

    p = max_ar  # AR order
    print(f"[info] Using AR order p = {p}")

    need_cols = ["target_time", y_col] + X_cols_all
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in IN_CSV: {missing}")

    df = df.dropna(subset=need_cols).copy()

    # ---------- time split with gap ---------- #
    n = len(df)
    split_idx = int(TRAIN_FRAC * n)

    gap_feat = max(max_ar, max_x)
    gap = max(H + gap_feat, MIN_SAFE_GAP)

    train_end_idx = df.index[split_idx - 1]
    te_start_idx = split_idx + gap
    if te_start_idx >= n:
        raise RuntimeError(f"Empty test set: split={split_idx}, gap={gap}, n={n}.")

    df_tr = df.iloc[:split_idx].copy()
    df_te = df.iloc[te_start_idx:].copy()
    if len(df_te) == 0:
        raise RuntimeError(f"Empty test set after applying gap={gap}.")

    df_tr = df_tr[df_tr["target_time"] <= train_end_idx]
    if len(df_tr) == 0:
        raise RuntimeError("All training rows trimmed by target_time; adjust H/gap.")

    # ---------- extract y arrays ---------- #
    y_tr_raw_all = df_tr[y_col].to_numpy(dtype=np.float64)
    y_te_raw_all = df_te[y_col].to_numpy(dtype=np.float64)

    # ---------- scale y ---------- #
    y_scaler = StandardScaler()
    y_tr_z_all = y_scaler.fit_transform(y_tr_raw_all.reshape(-1, 1)).ravel()
    y_te_z_all = y_scaler.transform(y_te_raw_all.reshape(-1, 1)).ravel()

    # ---------- build AR design on y_z only ---------- #
    if p > 0:
        n_tr = len(y_tr_z_all)
        n_te = len(y_te_z_all)
        if n_tr <= p or n_te <= p:
            raise RuntimeError(f"Not enough samples for AR order p={p}: n_tr={n_tr}, n_te={n_te}")

        # TRAIN effective samples (we drop the first p rows)
        n_tr_eff = n_tr - p
        y_tr_z = y_tr_z_all[p:]              # target y_z(t), t = p..n_tr-1
        y_tr_raw = y_tr_raw_all[p:]

        X_tr_ar = np.zeros((n_tr_eff, p))
        for i in range(1, p + 1):
            X_tr_ar[:, i-1] = y_tr_z_all[p-i : n_tr - i]   # y_z(t-i)

        # TEST effective samples
        n_te_eff = n_te - p
        y_te_z = y_te_z_all[p:]
        y_te_raw = y_te_raw_all[p:]

        X_te_ar = np.zeros((n_te_eff, p))
        for i in range(1, p + 1):
            X_te_ar[:, i-1] = y_te_z_all[p-i : n_te - i]

    else:
        # No AR part
        y_tr_z = y_tr_z_all
        y_te_z = y_te_z_all
        y_tr_raw = y_tr_raw_all
        y_te_raw = y_te_raw_all
        X_tr_ar = np.zeros((len(y_tr_z), 0))
        X_te_ar = np.zeros((len(y_te_z), 0))

    # ---------- build exogenous matrices ---------- #
    # Exogenous = all X_cols_all that are NOT y-like (no direct target leakage)
    def is_y_like(name: str) -> bool:
        """
        Heuristic to identify columns that directly represent the target
        or target-derived quantities. These should not be used as exogenous
        inputs in the ARX model.
        """
        base = y_col  # e.g. "Tot_Resistance_mOhm_filt" or "y_target"
        lname = name.lower()
        return (
            name == y_col
            or name.startswith("y_")
            or "residual" in lname
            or base in name
        )

    # keep everything that is NOT y-like
    exog_cols = [c for c in X_cols_all if not is_y_like(c)]

    if not exog_cols:
        print("[warn] No exogenous features left after filtering; pure AR model.")
        X_tr_ex_raw = None
        X_te_ex_raw = None
    else:
        X_tr_full = df_tr[exog_cols].to_numpy(dtype=np.float64)
        X_te_full = df_te[exog_cols].to_numpy(dtype=np.float64)

        # align with effective indices (drop first p rows)
        if p > 0:
            X_tr_ex_raw = X_tr_full[p:, :]
            X_te_ex_raw = X_te_full[p:, :]
        else:
            X_tr_ex_raw = X_tr_full
            X_te_ex_raw = X_te_full


    # ---------- AR part: fit + stabilize ---------- #
    if p > 0:
        # Fit y_tr_z = X_tr_ar @ a + eps (OLS)
        a_hat, *_ = np.linalg.lstsq(X_tr_ar, y_tr_z, rcond=None)
        a_hat = a_hat.ravel()
        a_stab = stabilize_ar_coeffs(a_hat)
        print(f"[AR] raw coeffs: {a_hat}")
        print(f"[AR] stabilized coeffs: {a_stab}")

        y_tr_ar = X_tr_ar @ a_stab
        y_te_ar = X_te_ar @ a_stab
    else:
        a_stab = np.array([])
        y_tr_ar = np.zeros_like(y_tr_z)
        y_te_ar = np.zeros_like(y_te_z)

    # ---------- exogenous part: fit on residuals ---------- #
    r_tr = y_tr_z - y_tr_ar
    r_te = y_te_z - y_te_ar  # for diagnostics only

    if exog_cols and X_tr_ex_raw is not None:
        # variance filter
        var = X_tr_ex_raw.var(axis=0)
        keep_var = var > VAR_EPS
        if not np.any(keep_var):
            raise RuntimeError("All exogenous features removed by variance threshold.")

        exog_cols_var = [c for c, k in zip(exog_cols, keep_var) if k]
        X_tr_ex_var = X_tr_ex_raw[:, keep_var]
        X_te_ex_var = X_te_ex_raw[:, keep_var]

        # correlation guard vs y_tr_z
        X_ex_cols_final = exog_cols_var
        X_tr_ex_guard = X_tr_ex_var
        X_te_ex_guard = X_te_ex_var

        if X_tr_ex_guard.shape[1] > 1:
            corrs = np.corrcoef(np.c_[X_tr_ex_guard, y_tr_z].T)[-1, :-1]
            suspicious = np.where(np.abs(corrs) > CORR_CUTOFF)[0]
            if suspicious.size > 0:
                suspicious_cols = [exog_cols_var[i] for i in suspicious]
                print(f"[guard] Dropping {len(suspicious_cols)} high-|corr| exog features: "
                      f"{suspicious_cols[:8]}{'...' if len(suspicious_cols) > 8 else ''}")
                mask_corr = np.ones(X_tr_ex_guard.shape[1], dtype=bool)
                mask_corr[suspicious] = False

                X_ex_cols_final = [c for c, m in zip(exog_cols_var, mask_corr) if m]
                X_tr_ex_guard = X_tr_ex_guard[:, mask_corr]
                X_te_ex_guard = X_te_ex_guard[:, mask_corr]

        # scale exog
        X_scaler_exog = StandardScaler()
        X_tr_ex = X_scaler_exog.fit_transform(X_tr_ex_guard)
        X_te_ex = X_scaler_exog.transform(X_te_ex_guard)

        # fit exog model
        if USE_CV:
            tscv = TimeSeriesSplit(n_splits=TS_SPLITS)
            exog_model = RidgeCV(
                alphas=ALPHAS,
                cv=tscv,
                scoring="neg_mean_squared_error",
                gcv_mode="svd",
            )
            exog_model.fit(X_tr_ex, r_tr)
            print(f"[exog] Chosen alpha = {getattr(exog_model, 'alpha_', None)}")
        else:
            exog_model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
            exog_model.fit(X_tr_ex, r_tr)

        rhat_tr = exog_model.predict(X_tr_ex)
        rhat_te = exog_model.predict(X_te_ex)
    else:
        X_scaler_exog = None
        exog_model = None
        X_ex_cols_final = []
        rhat_tr = np.zeros_like(r_tr)
        rhat_te = np.zeros_like(r_te)

    # ---------- full preds (z-space) & back to physical ---------- #
    yhat_tr_z = y_tr_ar + rhat_tr
    yhat_te_z = y_te_ar + rhat_te

    yhat_tr = y_scaler.inverse_transform(yhat_tr_z.reshape(-1, 1))[:, 0]
    yhat_te = y_scaler.inverse_transform(yhat_te_z.reshape(-1, 1))[:, 0]

    # metrics (on effective segments)
    tr_rmse = rmse(y_tr_raw, yhat_tr)
    te_rmse = rmse(y_te_raw, yhat_te)
    tr_r2   = r2_score(y_tr_raw, yhat_tr)
    te_r2   = r2_score(y_te_raw, yhat_te)

    print(f"[Train] RMSE={tr_rmse:.6f} mΩ, R²={tr_r2:.3f}")
    print(f"[ Test] RMSE={te_rmse:.6f} mΩ, R²={te_r2:.3f}")
    print(f"[Info ] Samples: train_eff={len(y_tr_raw)}, test_eff={len(y_te_raw)}, gap={gap}")
    print(f"[Info ] AR order: {p}, exog cols: {len(X_ex_cols_final)}")

    # ---------- save bundle ---------- #
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "ar_order": p,
        "ar_coeffs": a_stab,
        "exog_model": exog_model,
        "exog_cols": X_ex_cols_final,
        "y_col": y_col,
        "alpha": getattr(exog_model, "alpha_", getattr(exog_model, "alpha", None)) if exog_model else None,
        "train_frac": TRAIN_FRAC,
        "seed": SEED,
        "gap": gap,
        "ts_splits": TS_SPLITS,
        "var_eps": VAR_EPS,
        "corr_cutoff": CORR_CUTOFF,
        "scalers": {
            "y_scaler": y_scaler,
            "X_scaler_exog": X_scaler_exog,
        },
        "prep_config": cfg,
    }
    print(bundle["exog_cols"])
    dump(bundle, MODEL_OUT)
    print(f"[save] Model bundle -> {MODEL_OUT}")

    # ---------- coefficients & predictions CSV ---------- #
    coef_records = []

    if exog_model is not None:
        coef_ex = np.asarray(exog_model.coef_, dtype=float).ravel()
        for name, val in zip(X_ex_cols_final, coef_ex):
            coef_records.append({"feature": name, "coef_zspace": val})

        coef_records.append({"feature": "(intercept_exog)", "coef_zspace": float(exog_model.intercept_)})

    # AR coeffs
    if p > 0:
        for i, val in enumerate(a_stab, start=1):
            coef_records.append({"feature": f"AR_y_z_lag{i}", "coef_zspace": float(val)})

    coef_df = pd.DataFrame(coef_records).set_index("feature").sort_values(
        "coef_zspace", key=np.abs, ascending=False
    )
    coef_df.to_csv(COEF_CSV_OUT)
    print(f"[save] Coefficients (z-space) -> {COEF_CSV_OUT}")

    # predictions on test (physical units)
    out_df = pd.DataFrame({
        "timestamp": df_te.index[p:],    # align with y_te_raw
        "y_true_mOhm": y_te_raw,
        "y_pred_mOhm": yhat_te,
    }).set_index("timestamp")
    out_df.to_csv(PRED_CSV_OUT)
    print(f"[save] Predictions -> {PRED_CSV_OUT}")


if __name__ == "__main__":
    main()

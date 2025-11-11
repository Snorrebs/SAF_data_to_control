# ==== CONFIG (edit here, no argparse) ====
CFG_FILE = "config.yaml"
FS_HZ = 1.0
SAVE_SCALER = True
# ========================================

import pandas as pd
from pathlib import Path
from saf_dynamify.paths import Config
from saf_dynamify.utils import ensure_dir
from saf_dynamify.filtering import filter_dataframe
from saf_dynamify.normalize import fit_save_scaler, transform
from saf_dynamify.metamodel import predict_meta
from saf_dynamify.residuals import compute_residual_targets, build_lagged_matrix

cfg = Config(CFG_FILE)
raw = pd.read_csv(cfg.raw_csv, parse_dates=[cfg.p("columns", "timestamp")]).set_index(cfg.p("columns", "timestamp"))

# 1) Filter first (plant signals)
if cfg.p("filter", "enabled"):
    cols_to_filter = list(set(cfg.p("columns", "y") + cfg.p("columns", "X")))
    raw = filter_dataframe(
        raw,
        cols=cols_to_filter,
        cutoff_Hz=cfg.p("filter", "cutoff_Hz"),
        fs_Hz=FS_HZ,
        order=cfg.p("filter", "order"),
    )

# 2) Metamodel predictions (UNfiltered meta)
meta = predict_meta(raw, cfg.p("columns", "y"))
df = raw.join(meta, how="left")

# 3) Residual targets: y_filt - meta
Yres = compute_residual_targets(df, cfg.p("columns", "y"))

# 4) Build exog (filtered & z-scored later)
Xcols_f = [f"{c}_filt" if f"{c}_filt" in df.columns else c for c in cfg.p("columns", "X")]
X = df[Xcols_f].copy()

# 5) Normalize X (optionally also Yres if you want)
if cfg.p("normalize", "enabled"):
    if SAVE_SCALER:
        scaler = fit_save_scaler(X, X.columns.tolist(), cfg.models_dir)
    else:
        from src.saf_dynamify.normalize import StandardScaler
        scaler = StandardScaler().fit(X)
    Xz = transform(X, X.columns.tolist(), scaler)
else:
    Xz = X.copy()

# 6) Lagged design
lags = cfg.p("arx", "lags")
Xlag = build_lagged_matrix(Xz, Xz.columns.tolist(), lags)

# 7) Align with targets (drop NaNs from shifting)
Z = pd.concat([Yres, Xlag], axis=1).dropna()

# 8) Save interim & processed
ensure_dir(cfg.interim_csv)
ensure_dir(cfg.processed_parquet)
raw.reset_index().to_csv(cfg.interim_csv, index=False)
Z.to_parquet(cfg.processed_parquet)
print(f"Saved: {cfg.interim_csv} and {cfg.processed_parquet}")
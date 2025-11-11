# ==== CONFIG ====
CFG_FILE = "config.yaml"
SAVE_PATH = "models/arx_sklearn.joblib"
EVAL_PLOTS = True
# ================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.saf_dynamify.paths import Config
from src.saf_dynamify.arx import ARX, ARXConfig

cfg = Config(CFG_FILE)
Z = pd.read_parquet(cfg.processed_parquet)

# Split back into Y (residual targets) and Xlag
y_cols = [c for c in Z.columns if c.startswith("res_")]
X_cols = [c for c in Z.columns if c not in y_cols]
Y = Z[y_cols]
Xlag = Z[X_cols]

model = ARX(ARXConfig(lags=cfg.p("arx", "lags"), ridge=cfg.p("arx", "ridge")))
model.fit(Xlag, Y)
Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
model.save(Path(SAVE_PATH))
print(f"Saved ARX to {SAVE_PATH}")

# One-step prediction fit
Yhat = model.predict(Xlag)
resid = Y - Yhat
R2 = 1 - (resid.pow(2).sum() / (Y.sub(Y.mean()).pow(2).sum()))
print("One-step R²:")
print(R2)

# Multi-horizon rollout on a stride of windows (free-run with true exog)
H = cfg.p("arx", "horizons")
stride = cfg.p("arx", "stride")

rmseH, r2H = {}, {}
for h in H:
    errs = []
    fits = []
    # simple rolling windows
    for i in range(0, len(Z) - h, stride):
        Xwin = Xlag.iloc[i:i+1]  # one-step features at t
        Ytrue = Y.iloc[i+h:i+h+1]  # target at t+h (still residual)
        # naive: reuse same features for demo; for full recursion, stack lagged predictions
        Ypred = model.predict(Xwin)
        errs.append(float(((Ytrue - Ypred).pow(2)).mean(axis=1)))
        fits.append(1 - float(((Ytrue - Ypred).pow(2)).sum() / ((Ytrue - Y.mean()).pow(2)).sum()))
    rmseH[h] = np.sqrt(np.mean(errs))
    r2H[h] = np.mean(fits)

print("\n[FAST H] RMSE / R²:")
for h in H:
    print(f"H={h:2d}: RMSE={rmseH[h]:.4f}  R²={r2H[h]:.3f}")

if EVAL_PLOTS:
    plt.figure(figsize=(10,4))
    plt.plot(Y.iloc[:1000, 0].to_numpy(), label="res target")
    plt.plot(Yhat.iloc[:1000, 0].to_numpy(), label="1-step pred")
    plt.legend(); plt.title(y_cols[0]); plt.tight_layout()
    Path(cfg.figs_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(cfg.figs_dir)/"arx_fit_example.png", dpi=150)
    print(f"Saved plot to {Path(cfg.figs_dir)/'arx_fit_example.png'}")
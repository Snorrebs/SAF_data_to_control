# ==== CONFIG ====
CFG_FILE = "config.yaml"
SAVE_PATH = "models/arx_torch.pt"
# ================

import pandas as pd
import torch
from pathlib import Path
from src.saf_dynamify.paths import Config
from src.saf_dynamify.torch_arx import TorchARX, TorchARXConfig

cfg = Config(CFG_FILE)
Z = pd.read_parquet(cfg.processed_parquet)

# Split
y_cols = [c for c in Z.columns if c.startswith("res_")]
X_cols = [c for c in Z.columns if c not in y_cols]
Y = Z[y_cols]
Xlag = Z[X_cols]

model = TorchARX(TorchARXConfig(lags=cfg.p("torch_arx", "lags"), ridge=cfg.p("torch_arx", "ridge"), device=cfg.p("torch_arx", "device")))
model.fit(Xlag, Y)

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
torch.save({"W": model.W, "y_names": model.y_names, "columns": model.columns}, SAVE_PATH)
print(f"Saved Torch ARX to {SAVE_PATH}")
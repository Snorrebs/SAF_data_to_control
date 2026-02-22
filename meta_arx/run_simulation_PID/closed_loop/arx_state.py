from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple, Sequence

import numpy as np
import pandas as pd
from joblib import load


@dataclass
class ArxState:
    row: pd.Series

    # ---------- Convenience accessors ----------

    def current_y(self, y_col: str = "El1_Resistance_mOhm_filt") -> float:
        if y_col in self.row.index:
            return float(self.row[y_col])
        if "y_target" in self.row.index:
            return float(self.row["y_target"])
        if "y_filt_lag1" in self.row.index:
            return float(self.row["y_filt_lag1"])
        raise KeyError(f"No {y_col}, y_target, or y_filt_lag1 in state")

    def current_u_el1(self, u_base: str = "El1_dpos_mps_filt") -> float:
        if f"{u_base}_lag1" in self.row.index:
            return float(self.row[f"{u_base}_lag1"])
        if u_base in self.row.index:
            return float(self.row[u_base])
        return 0.0

    # ---------- Core prediction ----------

    def _get_y_lag_cols(self) -> list[str]:
        """
        Find y-lag columns in the row, e.g. ['y_raw_lag1', 'y_raw_lag2', ...]
        sorted by lag index (1, 2, 3, ...).
        """
        lag_cols = [c for c in self.row.index if c.startswith("y_raw_lag")]
        # sort by the trailing integer
        def lag_idx(name: str) -> int:
            # "y_raw_lag3" -> 3
            return int(name.replace("y_raw_lag", ""))

        return sorted(lag_cols, key=lag_idx)

    def predict_next_y(
        self,
        bundle: dict,
        clip_z: float = 10.0,
        fillna_with_mean: bool = True,
    ) -> float:
        """
        One-step prediction:
          physical x -> X_scaler -> model -> y_z -> y_scaler^{-1} -> physical y

        Returns physical y (mΩ).
        """
        model = bundle["model"]
        X_scaler = bundle["X_scaler"]
        y_scaler = bundle["y_scaler"]
        X_cols: List[str] = bundle["X_cols"]

        # physical feature row in model feature order
        x_raw = self.row.reindex(X_cols).to_numpy(dtype=float)[None, :]

        if np.isnan(x_raw).any():
            if fillna_with_mean and hasattr(X_scaler, "mean_"):
                mu = X_scaler.mean_[None, :]
                x_raw = np.where(np.isnan(x_raw), mu, x_raw)
            else:
                x_raw = np.nan_to_num(x_raw, nan=0.0)

        x_z = X_scaler.transform(x_raw)
        y_z = model.predict(x_z).reshape(-1, 1)
        y_z = np.clip(y_z, -clip_z, clip_z)

        y_phys = float(y_scaler.inverse_transform(y_z)[0, 0])
        return y_phys

    # ---------- State update (advance one step) ----------

    def advance(self, u_El2_new: float, y_new: float, max_lag: int = 5) -> None:
        """
        Advance the ARX state one step:

        - Update y-lag columns y_raw_lag1..lagp with the new y (physical units).
        - Update El2_pos_m_filt_lag* with the new electrode command.
        - For all other *_lagk bases, shift the lag window and keep lag1 frozen
          (i.e. use previous lag1 as new lag1).

        This assumes:
            - y_raw_lag1 holds y(t-1), y_raw_lag2=y(t-2), ...
            - El2_pos_m_filt_lag1 holds the most recent commanded El2 position.
        """
        # ---- generic base detection for *_lag1..lagN ----
        lag1_suffix = "_lag1"
        lag_bases = sorted({
            col[:-len(lag1_suffix)]
            for col in self.row.index
            if col.endswith(lag1_suffix)
        })

        for base in lag_bases:
            # build the full list of lag columns for this base, up to max_lag
            cols = [f"{base}_lag{k}" for k in range(1, max_lag + 1)]
            # keep only those that actually exist
            cols = [c for c in cols if c in self.row.index]
            if len(cols) < 2:
                continue  # nothing to shift

            old = self.row[cols].to_numpy(dtype=float)
            new = np.empty_like(old)

            # shift lags: lag(k+1) := old lag(k)
            new[1:] = old[:-1]

            if base == "El2_pos_m_filt":
                # newest electrode position is the new command
                new[0] = u_El2_new
            elif base == "y_raw":
                # newest y is the freshly predicted/measured y_new
                new[0] = y_new
            else:
                # keep the previous lag1 (freeze)
                new[0] = old[0]

            self.row.loc[cols] = new

        if "El2_pos_m_filt" in self.row.index:
            self.row["El2_pos_m_filt"] = u_El2_new

        if "y_target" in self.row.index:
            self.row["y_target"] = y_new


# ---------- Bundle / CSV helpers ----------

def load_arx_bundle(model_path: str) -> dict:
    """
    Load the new 'AR-on-y-only' stable ARX bundle with keys:
      - ar_order, ar_coeffs
      - exog_model, exog_cols
      - scalers: y_scaler, X_scaler_exog
    """
    bundle = load(model_path)
    required = ["model", "X_scaler", "y_scaler", "X_cols", "y_col"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(f"ARX bundle is missing required keys: {missing}")
    return bundle
    


def load_initial_state(
    csv_path: Union[str, Path],
    bundle: dict,
    idx: int | None = None,
    ) -> ArxState:

    df = pd.read_csv(csv_path)

    X_cols: List[str] = bundle["X_cols"]
    y_col: str = bundle["y_col"]

    needed = X_cols + [y_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"History CSV {csv_path} is missing required columns: {missing}")

    df_valid = df.dropna(subset=needed)
    if df_valid.empty:
        raise ValueError(f"No valid (non-NaN) rows in {csv_path} for columns {needed}")

    if idx is None:
        row = df_valid.iloc[-1].copy()
    else:
        row = df.iloc[idx].copy()

    return ArxState(row=row)
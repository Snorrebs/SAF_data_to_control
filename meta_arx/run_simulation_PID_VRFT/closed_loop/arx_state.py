from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import load


@dataclass
class ArxState:
    row: pd.Series

    # ---------- Convenience accessors ----------

    def current_y(self) -> float:
        """
        Return the most recent y in physical units (mΩ),
        either from 'y_target' or from 'y_raw_lag1' as fallback.
        """
        if "y_target" in self.row.index:
            return float(self.row["y_target"])
        elif "y_raw_lag1" in self.row.index:
            return float(self.row["y_raw_lag1"])
        raise KeyError("No y_target or y_raw_lag1 in ARX state")

    def current_u_el1(self) -> float:
        """
        Return the most recent El1 position (m) from the state row.
        """
        for c in ["El1_pos_m_filt_lag1", "El1_pos_m_filt"]:
            if c in self.row.index:
                return float(self.row[c])
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

    def predict_next_y(self, bundle: dict, clip_z: float = 10.0) -> float:
        """
        Predict next y using the arx.

        Model:
            y_z(t) = sum_{i=1..p} a_i * y_z(t-i) + f_exog(X_exog(t))
            y(t)   = inverse_transform_y(y_z(t))

        Where:
            - a_i are in bundle["ar_coeffs"]
            - p   = bundle["ar_order"]
            - y_z(t-i) are obtained by z-scaling the physical y-lags in the row
              (y_raw_lag1..p)
            - exog features are bundle["exog_cols"], scaled by X_scaler_exog
        """
        a          = np.asarray(bundle["ar_coeffs"], dtype=float).ravel()
        p          = int(bundle["ar_order"])
        exog_cols  = bundle["exog_cols"]
        exog_model = bundle["exog_model"]

        y_scaler      = bundle["scalers"]["y_scaler"]
        Xs_exog       = bundle["scalers"]["X_scaler_exog"]

        # ---- 1) AR part on y_z lags ----
        if p > 0:
            y_lag_cols = self._get_y_lag_cols()
            if len(y_lag_cols) < p:
                raise ValueError(
                    f"Not enough y-lag columns in state. "
                    f"Need at least {p}, have {len(y_lag_cols)}: {y_lag_cols}"
                )
            # Take the first p lags: y(t-1)..y(t-p) in physical units
            y_lags_phys = self.row[y_lag_cols[:p]].to_numpy(dtype=float)  # shape (p,)

            # z-scale using the same scaler as during training
            mu  = y_scaler.mean_[0]
            sig = y_scaler.scale_[0]
            y_lags_z = (y_lags_phys - mu) / sig        

            y_ar_z = float(a @ y_lags_z)
        else:
            y_ar_z = 0.0

        # ---- 2) Exogenous part on X_exog(t) ----
        if exog_model is not None and exog_cols:
            X_ex = self.row[exog_cols].to_numpy(dtype=float)[None, :]  # (1, n_ex)
            X_ex_z = Xs_exog.transform(X_ex)
            r_hat = float(exog_model.predict(X_ex_z))
        else:
            r_hat = 0.0

        # ---- 3) Combine and inverse-transform ----
        y_z = y_ar_z + r_hat

        #safety clip
        if not np.isfinite(y_z):
            y_z = 0.0
        else:
            y_z = float(np.clip(y_z, -clip_z, clip_z))

        y = y_scaler.inverse_transform([[y_z]])[0, 0]
        return float(y)

    # ---------- State update (advance one step) ----------

    def advance(self, u_el2_new: float, y_new: float, max_lag: int = 5) -> None:
        """
        Advance the ARX state one step:

        - Update y-lag columns y_raw_lag1..lagp with the new y (physical units).
        - Update El1_pos_m_filt_lag* with the new electrode command.
        - For all other *_lagk bases, shift the lag window and keep lag1 frozen
          (i.e. use previous lag1 as new lag1).

        This assumes:
            - y_raw_lag1 holds y(t-1), y_raw_lag2=y(t-2), ...
            - El1_pos_m_filt_lag1 holds the most recent commanded El1 position.
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
                new[0] = u_el2_new
            elif base == "y_raw":
                # newest y is the freshly predicted/measured y_new
                new[0] = y_new
            else:
                # keep the previous lag1 (freeze)
                new[0] = old[0]

            self.row.loc[cols] = new

        if "El2_pos_m_filt" in self.row.index:
            self.row["El2_pos_m_filt"] = u_el2_new

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
    b = load(model_path)
    required_top = ["ar_order", "ar_coeffs", "exog_model", "exog_cols", "scalers"]
    for k in required_top:
        if k not in b:
            raise KeyError(f"Stable ARX bundle is missing key '{k}'")

    required_scalers = ["y_scaler", "X_scaler_exog"]
    for k in required_scalers:
        if k not in b["scalers"]:
            raise KeyError(f"Stable ARX bundle missing scalers['{k}']")

    return b


def load_initial_state(csv_path: str, bundle: dict) -> "ArxState":
    """
    Use the last row of the history CSV as initial state.

    Requirements:
      - Must contain all exog_cols from the bundle.
      - Should contain y_raw_lag1..p (for AR lags) and optionally y_target.
    """
    df = pd.read_csv(csv_path)
    exog_cols = list(bundle["exog_cols"])

    missing_exog = [c for c in exog_cols if c not in df.columns]
    if missing_exog:
        raise ValueError(f"History CSV missing exogenous columns needed for ARX: {missing_exog}")

    # y-lag columns are checked lazily in predict_next_y; we just warn here
    p = int(bundle["ar_order"])
    y_lag_names = [f"y_raw_lag{i}" for i in range(1, p + 1)]
    missing_y_lags = [c for c in y_lag_names if c not in df.columns]
    if missing_y_lags:
        print(f"[warn] History CSV missing some y-lag columns {missing_y_lags}. "
              f"predict_next_y() will fail if it needs them.")

    last_row = df.iloc[-1].copy()
    return ArxState(row=last_row)

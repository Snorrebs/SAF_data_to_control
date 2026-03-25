from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


@dataclass
class ArxState:
    """Lagged VARX state stored as a single pandas Series (row)."""

    row: pd.Series

    def current_y(self, y_cols: list[str] | None = None) -> np.ndarray:
        """Return current output vector.

        For VARX-current this is typically [El1_kA_filt, El2_kA_filt, El3_kA_filt].
        Falls back to lag1 if non-lagged columns are not present.
        """

        if y_cols is None:
            y_cols = ["El1_kA_filt", "El2_kA_filt", "El3_kA_filt"]

        vals: list[float] = []
        for c in y_cols:
            if c in self.row.index:
                vals.append(float(self.row[c]))
                continue

            lag1 = f"{c}_lag1"
            if lag1 in self.row.index:
                vals.append(float(self.row[lag1]))
                continue

            # legacy SISO fallback
            if c == "El1_kA_filt" and "y_filt_lag1" in self.row.index:
                vals.append(float(self.row["y_filt_lag1"]))
                continue

            raise KeyError(f"No '{c}' (or '{c}_lag1') found in state")

        return np.asarray(vals, dtype=float)

    def current_u(self, u_bases: list[str] | None = None) -> np.ndarray:
        """Return current input vector from lag1 values.

        For VARX-current this is typically electrode positions:
        [El1_pos_m_filt, El2_pos_m_filt, El3_pos_m_filt].
        """

        if u_bases is None:
            u_bases = ["El1_pos_m_filt", "El2_pos_m_filt", "El3_pos_m_filt"]

        vals: list[float] = []
        for base in u_bases:
            lag1 = f"{base}_lag1"
            if lag1 in self.row.index:
                vals.append(float(self.row[lag1]))
            elif base in self.row.index:
                vals.append(float(self.row[base]))
            else:
                vals.append(0.0)

        return np.asarray(vals, dtype=float)

    def predict_next_y(
        self,
        bundle: dict,
        clip_z: float = 10.0,
        fillna_with_mean: bool = True,
    ) -> np.ndarray:
        """One-step prediction y(t) from current lagged row."""

        model = bundle["model"]
        x_scaler = bundle["X_scaler"]
        # new bundles use Y_scaler; keep backwards compatibility
        y_scaler = bundle.get("Y_scaler", bundle.get("y_scaler"))
        if y_scaler is None:
            raise KeyError("Bundle missing Y_scaler (or legacy y_scaler)")

        x_cols: list[str] = bundle["X_cols"]

        x_raw = self.row.reindex(x_cols).to_numpy(dtype=float)[None, :]
        if np.isnan(x_raw).any():
            if fillna_with_mean and hasattr(x_scaler, "mean_"):
                x_raw = np.where(np.isnan(x_raw), x_scaler.mean_[None, :], x_raw)
            else:
                x_raw = np.nan_to_num(x_raw, nan=0.0)

        x_z = x_scaler.transform(x_raw)
        y_z = np.clip(model.predict(x_z), -clip_z, clip_z)
        y = y_scaler.inverse_transform(y_z)
        return y.reshape(-1)

    def advance(self, u_new: np.ndarray, y_new: np.ndarray) -> None:
        """Advance all *_lagk columns by one step.

        - Updates *_kA_filt_lag* with y_new (vector)
        - Updates *_pos_m_filt_lag* with u_new (vector)
        - Keeps other exogenous lags (voltages/resistances) frozen by default
        """

        u_new = np.asarray(u_new, dtype=float).reshape(-1)
        y_new = np.asarray(y_new, dtype=float).reshape(-1)

        lag_bases = sorted(col[:-5] for col in self.row.index if col.endswith("_lag1"))

        for base in lag_bases:
            # detect available lags for this base
            cols: list[str] = []
            k = 1
            while True:
                c = f"{base}_lag{k}"
                if c not in self.row.index:
                    break
                cols.append(c)
                k += 1

            if len(cols) < 2:
                continue

            old = self.row[cols].to_numpy(dtype=float)
            new = np.empty_like(old)
            new[1:] = old[:-1]

            if base.endswith("pos_m_filt"):
                # expects El1/El2/El3 naming
                try:
                    idx = int(base[2]) - 1
                except Exception:
                    idx = 0
                new[0] = u_new[idx] if idx < len(u_new) else float(u_new[-1])

            elif base.endswith("kA_filt"):
                try:
                    idx = int(base[2]) - 1
                except Exception:
                    idx = 0
                new[0] = y_new[idx] if idx < len(y_new) else float(y_new[-1])

            else:
                # exogenous: freeze (extend here later if you replay trajectories)
                new[0] = old[0]

            self.row.loc[cols] = new

        # keep non-lagged columns consistent if they exist
        for i, base in enumerate(["El1_pos_m_filt", "El2_pos_m_filt", "El3_pos_m_filt"]):
            if base in self.row.index and i < len(u_new):
                self.row[base] = float(u_new[i])

        for i, base in enumerate(["El1_kA_filt", "El2_kA_filt", "El3_kA_filt"]):
            if base in self.row.index and i < len(y_new):
                self.row[base] = float(y_new[i])


def load_arx_bundle(model_path: str) -> dict:
    bundle = load(model_path)

    # common required
    required_any = ["model", "X_scaler", "X_cols"]
    missing_any = [k for k in required_any if k not in bundle]
    if missing_any:
        raise KeyError(f"Model bundle is missing required keys: {missing_any}")

    # VARX vs legacy
    has_varx = ("y_cols" in bundle) and ("Y_scaler" in bundle or "y_scaler" in bundle)
    has_siso = ("y_col" in bundle) and ("y_scaler" in bundle or "Y_scaler" in bundle)

    if not (has_varx or has_siso):
        raise KeyError("Bundle must contain either (y_cols + Y_scaler) or (y_col + y_scaler)")

    return bundle


def load_initial_state(csv_path: str | Path, bundle: dict, idx: int | None = None) -> ArxState:
    df = pd.read_csv(csv_path)
    x_cols: list[str] = bundle["X_cols"]

    # VARX bundles store y_cols; legacy stores y_col
    y_cols = bundle.get("y_cols")
    if y_cols is None:
        y_cols = [bundle["y_col"]]

    needed = x_cols + list(y_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"History CSV {csv_path} is missing required columns: {missing}")

    df_valid = df.dropna(subset=needed)
    if df_valid.empty:
        raise ValueError(f"No valid (non-NaN) rows in {csv_path} for columns {needed}")

    row = df_valid.iloc[-1].copy() if idx is None else df.iloc[idx].copy()
    return ArxState(row=row)

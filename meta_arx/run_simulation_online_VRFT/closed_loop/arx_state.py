from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from joblib import load


Number = Union[float, int]


@dataclass
class ArxState:
    row: pd.Series

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

    def _get_lag_bases(self, lag1_suffix: str = "_lag1") -> list[str]:
        return sorted({
            col[:-len(lag1_suffix)]
            for col in self.row.index
            if col.endswith(lag1_suffix)
        })

    def predict_next_y(self, bundle: dict, clip_z: float = 10.0) -> float:
        model = bundle["model"]
        X_scaler = bundle["X_scaler"]
        y_scaler = bundle["y_scaler"]
        X_cols = bundle["X_cols"]

        x_raw = self.row[X_cols].to_numpy(dtype=float)[None, :]

        if np.isnan(x_raw).any():
            mu = getattr(X_scaler, "mean_", None)
            if mu is not None:
                x_raw = np.where(np.isnan(x_raw), mu, x_raw)
            else:
                x_raw = np.nan_to_num(x_raw, nan=0.0)

        x_z = X_scaler.transform(x_raw)
        y_z = model.predict(x_z).reshape(-1, 1)
        y_z = np.clip(y_z, -clip_z, clip_z)
        return float(y_scaler.inverse_transform(y_z)[0, 0])

    def advance(
        self,
        u_el2_new: Optional[Number] = None,
        y_new: Optional[Number] = None,
        *,
        u_el1_new: Optional[Number] = None,
        max_lag: int = 5,
        y_base: str = "y_filt",
        u_base: str = "El1_dpos_mps_filt",
        y_col: str = "El1_Resistance_mOhm_filt",
    ) -> None:
        if y_new is None:
            raise ValueError("y_new must be provided.")

        if u_el1_new is not None:
            u_new = float(u_el1_new)
        elif u_el2_new is not None:
            u_new = float(u_el2_new)
        else:
            u_new = None

        bases = self._get_lag_bases()

        for base in bases:
            cols = [
                f"{base}_lag{k}"
                for k in range(1, max_lag + 1)
                if f"{base}_lag{k}" in self.row.index
            ]
            if len(cols) < 2:
                continue

            old = self.row[cols].to_numpy(dtype=float)
            new = np.empty_like(old)
            new[1:] = old[:-1]

            if base == u_base and u_new is not None:
                new[0] = u_new
            elif base == y_base:
                new[0] = float(y_new)
            else:
                new[0] = old[0]

            self.row.loc[cols] = new

        if u_new is not None and u_base in self.row.index:
            self.row[u_base] = u_new

        if y_col in self.row.index:
            self.row[y_col] = float(y_new)

        if "y_target" in self.row.index:
            self.row["y_target"] = float(y_new)


# ---------------- Bundle / Initial State --------------------


def load_arx_bundle(model_path: Union[str, Path]) -> dict:
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

    X_cols = bundle["X_cols"]
    y_col = bundle["y_col"]

    needed = X_cols + [y_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"History CSV {csv_path} is missing required columns: {missing}"
        )

    df_valid = df.dropna(subset=needed)
    if df_valid.empty:
        raise ValueError(
            f"No valid (non-NaN) rows in {csv_path} for columns {needed}"
        )

    if idx is None:
        row = df_valid.iloc[-1].copy()
    else:
        row = df.iloc[idx].copy()

    return ArxState(row=row)
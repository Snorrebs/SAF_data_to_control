from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from joblib import load


@dataclass(frozen=True)
class ModelIOConfig:
    input_base: str
    output_col: str
    output_lag_base: str = "y_filt"


def _lag_base(col: str) -> str | None:
    if "_lag" not in col:
        return None
    return col.rsplit("_lag", 1)[0]


def infer_model_io(bundle: dict, *, input_base: str | None = None, output_lag_base: str | None = None) -> ModelIOConfig:
    x_cols: list[str] = bundle["X_cols"]
    output_col: str = bundle["y_col"]

    lag_bases = [_lag_base(col) for col in x_cols if _lag_base(col) is not None]
    lag_bases = [base for base in lag_bases if base is not None]

    if output_lag_base is None:
        if "y_filt_lag1" in x_cols:
            output_lag_base = "y_filt"
        elif f"{output_col}_lag1" in x_cols:
            output_lag_base = output_col
        else:
            output_lag_base = "y_filt"

    if input_base is None:
        counts = Counter(base for base in lag_bases if base != output_lag_base)
        if not counts:
            raise ValueError("Could not infer input_base from bundle['X_cols']")
        input_base = counts.most_common(1)[0][0]

    return ModelIOConfig(
        input_base=input_base,
        output_col=output_col,
        output_lag_base=output_lag_base,
    )


@dataclass
class ArxState:
    row: pd.Series
    io: ModelIOConfig

    def current_y(self) -> float:
        for col in (self.io.output_col, "y_target", f"{self.io.output_lag_base}_lag1"):
            if col in self.row.index:
                return float(self.row[col])
        raise KeyError(
            f"No {self.io.output_col}, y_target, or {self.io.output_lag_base}_lag1 in state"
        )

    def current_u(self) -> float:
        lag_col = f"{self.io.input_base}_lag1"
        if lag_col in self.row.index:
            return float(self.row[lag_col])
        if self.io.input_base in self.row.index:
            return float(self.row[self.io.input_base])
        return 0.0

    def predict_next_y(
        self,
        bundle: dict,
        clip_z: float = 10.0,
        fillna_with_mean: bool = True,
    ) -> float:
        model = bundle["model"]
        x_scaler = bundle["X_scaler"]
        y_scaler = bundle["y_scaler"]
        x_cols: list[str] = bundle["X_cols"]

        x_raw = self.row.reindex(x_cols).to_numpy(dtype=float)[None, :]
        if np.isnan(x_raw).any():
            if fillna_with_mean and hasattr(x_scaler, "mean_"):
                x_raw = np.where(np.isnan(x_raw), x_scaler.mean_[None, :], x_raw)
            else:
                x_raw = np.nan_to_num(x_raw, nan=0.0)

        x_z = x_scaler.transform(x_raw)
        y_z = np.clip(model.predict(x_z).reshape(-1, 1), -clip_z, clip_z)
        return float(y_scaler.inverse_transform(y_z)[0, 0])

    def advance(self, u_new: float, y_new: float, max_lag: int = 5) -> None:
        lag_bases = sorted(col[:-5] for col in self.row.index if col.endswith("_lag1"))

        for base in lag_bases:
            cols = [f"{base}_lag{k}" for k in range(1, max_lag + 1)]
            cols = [c for c in cols if c in self.row.index]
            if len(cols) < 2:
                continue

            old = self.row[cols].to_numpy(dtype=float)
            new = np.empty_like(old)
            new[1:] = old[:-1]

            if base == self.io.input_base:
                new[0] = u_new
            elif base == self.io.output_lag_base:
                new[0] = y_new
            else:
                new[0] = old[0]

            self.row.loc[cols] = new

        if self.io.input_base in self.row.index:
            self.row[self.io.input_base] = u_new
        if self.io.output_col in self.row.index:
            self.row[self.io.output_col] = y_new
        if "y_target" in self.row.index:
            self.row["y_target"] = y_new


def load_arx_bundle(model_path: str) -> dict:
    bundle = load(model_path)
    required = ["model", "X_scaler", "y_scaler", "X_cols", "y_col"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(f"ARX bundle is missing required keys: {missing}")
    return bundle


def load_initial_state(
    csv_path: str | Path,
    bundle: dict,
    idx: int | None = None,
    *,
    io_cfg: ModelIOConfig | None = None,
) -> ArxState:
    df = pd.read_csv(csv_path)
    x_cols: list[str] = bundle["X_cols"]
    y_col: str = bundle["y_col"]

    needed = x_cols + [y_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"History CSV {csv_path} is missing required columns: {missing}")

    df_valid = df.dropna(subset=needed)
    if df_valid.empty:
        raise ValueError(f"No valid (non-NaN) rows in {csv_path} for columns {needed}")

    row = df_valid.iloc[-1].copy() if idx is None else df.iloc[idx].copy()
    if io_cfg is None:
        io_cfg = infer_model_io(bundle)
    return ArxState(row=row, io=io_cfg)

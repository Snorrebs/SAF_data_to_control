from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import load

Number = Union[float, int]


def _infer_max_lag_from_cols(cols: List[str]) -> int:
    lags = []
    for c in cols:
        if "_lag" in c:
            try:
                lags.append(int(c.split("_lag")[-1]))
            except Exception:
                pass
    return max(lags) if lags else 1


def _extract_base_and_lag(col: str) -> Optional[Tuple[str, int]]:
    """Parse '<base>_lag<k>' -> (base, k). Return None if not a lag column."""
    if "_lag" not in col:
        return None
    try:
        base, lag_str = col.rsplit("_lag", 1)
        k = int(lag_str)
        return base, k
    except Exception:
        return None


@dataclass
class ArxState:
    """
    State holds *physical* values (raw units), not standardized.

    Invariant:
      - self.row contains physical values for all model features (X_cols) and y_col.
      - predict_next_y() standardizes using X_scaler, predicts y_z, then returns y in physical units.
      - advance() shifts lag columns in physical units.
    """
    row: pd.Series

    # --------------------- convenience getters ---------------------

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

    # --------------------- core: prediction ---------------------

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

    # --------------------- core: state advance ---------------------

    def advance(
        self,
        *,
        bundle: dict,
        y_new: Number,
        u_el1_new: Optional[Number] = None,
        u_el2_new: Optional[Number] = None,
        y_col: str = "El1_Resistance_mOhm_filt",
        u_base: str = "El1_dpos_mps_filt",
        y_base: str = "y_filt",
        # If you ever want to feed y_new in z-space explicitly:
        y_new_is_z: bool = False,
        clip_z: float = 10.0,
    ) -> None:
        """
        Shift lagged variables forward by 1 step and insert new values.

        IMPORTANT:
          - State is stored in physical units.
          - y_new must be physical unless y_new_is_z=True (then it will be converted).
          - u_new is assumed physical.

        This advances only the lag columns that the model actually uses (bundle["X_cols"]).
        """
        X_scaler = bundle["X_scaler"]
        y_scaler = bundle["y_scaler"]
        X_cols: List[str] = bundle["X_cols"]

        # Determine u_new (physical)
        if u_el1_new is not None:
            u_new = float(u_el1_new)
        elif u_el2_new is not None:
            u_new = float(u_el2_new)
        else:
            u_new = None

        # Convert y_new if user passes z-space
        if y_new_is_z:
            y_z = float(np.clip(float(y_new), -clip_z, clip_z))
            y_new_phys = float(y_scaler.inverse_transform([[y_z]])[0, 0])
        else:
            y_new_phys = float(y_new)

        # Guardrail: catch accidental z-values being fed as physical
        # (Typical y in mΩ will NOT be order 1e-2; z-values usually are).
        if not y_new_is_z:
            # soft heuristic; adjust bounds if your y is very small in magnitude
            if abs(y_new_phys) < 1e-2 and hasattr(y_scaler, "scale_") and y_scaler.scale_[0] > 1e-2:
                # This indicates y_new might be standardized rather than physical.
                # We don't hard-fail because some signals might legitimately be tiny,
                # but we warn loudly.
                # (Replace with raise ValueError(...) if you want strict behavior.)
                print(
                    "[warn] y_new is very close to 0 in physical units. "
                    "If you accidentally passed y in z-space, set y_new_is_z=True."
                )

        # Build mapping: base -> list of lag columns used by the model
        # We only advance what the model uses, which avoids mismatched max_lag.
        lag_cols_by_base: Dict[str, List[Tuple[int, str]]] = {}
        for c in X_cols:
            parsed = _extract_base_and_lag(c)
            if parsed is None:
                continue
            base, k = parsed
            lag_cols_by_base.setdefault(base, []).append((k, c))

        # For each base, sort by lag increasing: lag1, lag2, ...
        for base, items in lag_cols_by_base.items():
            items.sort(key=lambda t: t[0])  # (k, col)

            # Only meaningful if we have at least lag1 and lag2
            if len(items) < 2:
                continue

            ks = [k for k, _ in items]
            cols = [col for _, col in items]

            # Read current physical values
            old = self.row.reindex(cols).to_numpy(dtype=float)

            # Shift: lag2 <- lag1, lag3 <- lag2, ...
            new = old.copy()
            new[1:] = old[:-1]

            # Insert new[0] depending on which base this is
            if base == u_base and u_new is not None:
                new[0] = u_new
            elif base == y_base:
                new[0] = y_new_phys
            else:
                # hold last value (or could set to mean)
                new[0] = old[0]

            # Write back to row
            self.row.loc[cols] = new

        # Also keep non-lag raw columns in sync if present
        if u_new is not None and u_base in self.row.index:
            self.row[u_base] = u_new

        if y_col in self.row.index:
            self.row[y_col] = y_new_phys

        if "y_target" in self.row.index:
            self.row["y_target"] = y_new_phys

    # --------------------- optional helper: make equilibrium row ---------------------

    def set_to_equilibrium_means(self, bundle: dict, y_col: str = "El1_Resistance_mOhm_filt") -> None:
        """
        Set state to the training equilibrium (z=0):
          - X_cols set to X_scaler.mean_
          - y_col set to y_scaler.mean_
        Useful for clean step/impulse tests in closed-loop infrastructure.
        """
        X_scaler = bundle["X_scaler"]
        y_scaler = bundle["y_scaler"]
        X_cols: List[str] = bundle["X_cols"]

        # Ensure all X_cols exist
        for c in X_cols:
            if c not in self.row.index:
                self.row[c] = np.nan

        self.row.loc[X_cols] = X_scaler.mean_
        if y_col in self.row.index:
            self.row[y_col] = float(y_scaler.mean_[0])
        if "y_target" in self.row.index:
            self.row["y_target"] = float(y_scaler.mean_[0])


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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from joblib import load


_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<lag>\d+)(?P<suffix>.*)$")


@dataclass
class ArxState:
    """Lagged ARX/VARX state stored as a single pandas Series."""

    row: pd.Series

    def current_y(self, y_cols: list[str] | None = None) -> np.ndarray:
        """Return the current output vector.

        If a non-lagged output column is unavailable, the corresponding
        lag-1 column is used.
        """
        if y_cols is None:
            y_cols = [
                "El1_Resistance_mOhm_filt",
                "El2_Resistance_mOhm_filt",
                "El3_Resistance_mOhm_filt",
            ]

        vals: list[float] = []
        for col in y_cols:
            if col in self.row.index:
                vals.append(float(self.row[col]))
                continue

            lag1 = f"{col}_lag1"
            if lag1 in self.row.index:
                vals.append(float(self.row[lag1]))
                continue

            raise KeyError(f"No '{col}' or '{lag1}' found in state")

        return np.asarray(vals, dtype=float)

    def current_values(self, bases: list[str], default: float = 0.0) -> np.ndarray:
        """Return current values for a list of signal bases."""
        vals: list[float] = []
        for base in bases:
            if base in self.row.index:
                vals.append(float(self.row[base]))
            elif f"{base}_lag1" in self.row.index:
                vals.append(float(self.row[f"{base}_lag1"]))
            else:
                vals.append(float(default))
        return np.asarray(vals, dtype=float)

    def current_u(self, u_bases: list[str] | None = None) -> np.ndarray:
        """Return the latest movement input values from the state."""
        if u_bases is None:
            u_bases = [
                "El1_dpos_m_filt",
                "El2_dpos_m_filt",
                "El3_dpos_m_filt",
            ]
            legacy = [
                "El1_pos_m_filt",
                "El2_pos_m_filt",
                "El3_pos_m_filt",
            ]
            if not any(f"{base}_lag1" in self.row.index or base in self.row.index for base in u_bases):
                u_bases = legacy

        return self.current_values(u_bases, default=0.0)

    @staticmethod
    def _x_cols_for_bundle(bundle: dict, equation_index: int | None = None) -> list[str]:
        if equation_index is not None and "X_cols_per_eq" in bundle:
            return list(bundle["X_cols_per_eq"][equation_index])
        if "X_cols" in bundle:
            return list(bundle["X_cols"])
        if "X_cols_flat" in bundle:
            return list(bundle["X_cols_flat"])
        if "X_cols_per_eq" in bundle:
            return [col for cols in bundle["X_cols_per_eq"] for col in cols]
        raise KeyError("Bundle missing X column definitions")

    def _input_row(self, cols: list[str], overrides: dict[str, float] | None) -> np.ndarray:
        row = self.row
        if overrides:
            row = row.copy()
            for col, val in overrides.items():
                row.loc[col] = float(val)
        return row.reindex(cols).to_numpy(dtype=float)[None, :]

    def predict_next_y(
        self,
        bundle: dict,
        clip_z: float = 10.0,
        fillna_with_mean: bool = True,
        overrides: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Predict one step ahead from the current lagged state.

        Values in ``overrides`` are applied only while constructing the model
        input vector and are not written to the stored state.
        """
        y_scaler = bundle.get("Y_scaler", bundle.get("y_scaler"))
        if y_scaler is None:
            raise KeyError("Bundle missing Y_scaler or y_scaler")

        if "models" in bundle and "X_scalers" in bundle and "X_cols_per_eq" in bundle:
            models = bundle["models"]
            x_scalers = bundle["X_scalers"]
            x_col_lists = bundle["X_cols_per_eq"]

            y_z = np.zeros(len(models), dtype=float)
            for ei, model in enumerate(models):
                cols = list(x_col_lists[ei])
                x_raw = self._input_row(cols, overrides)

                if np.isnan(x_raw).any():
                    if fillna_with_mean:
                        x_raw = np.where(np.isnan(x_raw), x_scalers[ei].mean_[None, :], x_raw)
                    else:
                        x_raw = np.nan_to_num(x_raw, nan=0.0)

                x_z = x_scalers[ei].transform(x_raw)
                y_z[ei] = float(np.clip(model.predict(x_z), -clip_z, clip_z))

            return (y_z * y_scaler.scale_ + y_scaler.mean_).reshape(-1)

        model = bundle["model"]
        x_scaler = bundle["X_scaler"]
        x_cols = self._x_cols_for_bundle(bundle)
        x_raw = self._input_row(x_cols, overrides)

        if np.isnan(x_raw).any():
            if fillna_with_mean and hasattr(x_scaler, "mean_"):
                x_raw = np.where(np.isnan(x_raw), x_scaler.mean_[None, :], x_raw)
            else:
                x_raw = np.nan_to_num(x_raw, nan=0.0)

        x_z = x_scaler.transform(x_raw)
        y_z = np.clip(model.predict(x_z), -clip_z, clip_z)

        if hasattr(y_scaler, "inverse_transform"):
            return y_scaler.inverse_transform(y_z).reshape(-1)
        return (y_z * y_scaler.scale_ + y_scaler.mean_).reshape(-1)

    def _lag_groups(self) -> dict[tuple[str, str], list[tuple[int, str]]]:
        groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for col in self.row.index:
            match = _LAG_RE.match(str(col))
            if not match:
                continue
            base = match.group("base")
            lag = int(match.group("lag"))
            suffix = match.group("suffix")
            groups.setdefault((base, suffix), []).append((lag, str(col)))

        for key in groups:
            groups[key].sort(key=lambda item: item[0])
        return groups

    def advance_lags(
        self,
        updates: dict[str, float],
        freeze_missing: bool = True,
    ) -> None:
        """Advance all lag columns and insert supplied values at lag 1.

        Lag groups are matched by signal base name. Columns with suffixes after
        the lag number, such as cross-equation columns, are shifted together
        with the same base signal.
        """
        updates = {str(base): float(value) for base, value in updates.items()}

        for (base, _suffix), lag_cols in self._lag_groups().items():
            cols = [col for _, col in lag_cols]
            old = self.row.loc[cols].to_numpy(dtype=float)
            new = np.empty_like(old)

            if len(old) > 1:
                new[1:] = old[:-1]

            if base in updates:
                new[0] = updates[base]
            elif freeze_missing:
                new[0] = old[0]
            else:
                new[0] = np.nan

            self.row.loc[cols] = new

        for base, value in updates.items():
            if base in self.row.index:
                self.row.loc[base] = value

    def advance(
        self,
        u_new: np.ndarray,
        y_new: np.ndarray,
        exog_new: dict[str, float] | None = None,
    ) -> None:
        """Advance a resistance-model state by one step."""
        u_new = np.asarray(u_new, dtype=float).reshape(-1)
        y_new = np.asarray(y_new, dtype=float).reshape(-1)

        updates: dict[str, float] = {}

        for i, base in enumerate([
            "El1_Resistance_mOhm_filt",
            "El2_Resistance_mOhm_filt",
            "El3_Resistance_mOhm_filt",
        ]):
            if i < len(y_new):
                updates[base] = float(y_new[i])

        dpos_bases = [
            "El1_dpos_m_filt",
            "El2_dpos_m_filt",
            "El3_dpos_m_filt",
        ]
        legacy_pos_bases = [
            "El1_pos_m_filt",
            "El2_pos_m_filt",
            "El3_pos_m_filt",
        ]

        available_bases = {base for base, _suffix in self._lag_groups().keys()} | set(self.row.index)
        if not any(base in available_bases for base in dpos_bases):
            dpos_bases = legacy_pos_bases

        for i, base in enumerate(dpos_bases):
            if i < len(u_new):
                updates[base] = float(u_new[i])

        if exog_new:
            updates.update({str(base): float(value) for base, value in exog_new.items()})

        self.advance_lags(updates, freeze_missing=True)


def _all_x_cols(bundle: dict) -> list[str]:
    if "X_cols" in bundle:
        return list(bundle["X_cols"])
    if "X_cols_flat" in bundle:
        return list(bundle["X_cols_flat"])
    if "X_cols_per_eq" in bundle:
        return [col for cols in bundle["X_cols_per_eq"] for col in cols]
    raise KeyError("Bundle missing X_cols, X_cols_flat, or X_cols_per_eq")


def _y_cols(bundle: dict) -> list[str]:
    if "y_cols" in bundle:
        return list(bundle["y_cols"])
    if "y_col" in bundle:
        return [bundle["y_col"]]
    return []


def load_arx_bundle(model_path: str) -> dict:
    bundle = load(model_path)

    has_new_varx = all(k in bundle for k in ["models", "X_scalers", "X_cols_per_eq"])
    has_legacy = all(k in bundle for k in ["model", "X_scaler"])
    if not (has_new_varx or has_legacy):
        raise KeyError(
            "Bundle must contain either ('models' + 'X_scalers' + 'X_cols_per_eq') "
            "or legacy ('model' + 'X_scaler')"
        )

    if "Y_scaler" not in bundle and "y_scaler" not in bundle:
        raise KeyError("Bundle missing Y_scaler or y_scaler")

    _all_x_cols(bundle)
    return bundle


def load_initial_state(
    csv_path: str | Path,
    bundle: dict,
    idx: int | None = None,
    allow_missing_lag0: bool = True,
) -> ArxState:
    df = pd.read_csv(csv_path)

    needed = _all_x_cols(bundle) + _y_cols(bundle)
    if allow_missing_lag0:
        needed = [col for col in needed if not re.search(r"_lag0($|[^0-9])", str(col))]

    needed = list(dict.fromkeys(needed))
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"History CSV {csv_path} is missing required columns: {missing}")

    df_valid = df.dropna(subset=needed)
    if df_valid.empty:
        raise ValueError(f"No valid non-NaN rows in {csv_path} for columns {needed}")

    row = df_valid.iloc[-1].copy() if idx is None else df.iloc[idx].copy()
    return ArxState(row=row)

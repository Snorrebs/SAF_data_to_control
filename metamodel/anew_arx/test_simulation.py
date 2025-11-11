#!/usr/bin/env python3
"""
Test the MetamodelARXForecaster with synthetic data.

What this does:
- Loads your saved models:
    models/meta/plsr_with_terms.joblib
    models/arx/arx_model.joblib
- Parses x_terms to figure out which *raw/base* columns the metamodel needs
- Builds a synthetic history_df (180 s @ 1 Hz) with those base columns + ARX exog
- Synthesizes Tot_Resistance_mOhm = meta + small noise (to seed AR residuals)
- Runs two forecasts:
    A) Hold-last 30 s (no future_df)
    B) True 30-row future_df with gentle ramps in inputs
"""

import re
import os
import numpy as np
import pandas as pd
import joblib

# --- import your forecaster classes/functions ---
# Ensure these files are importable (same folder or in PYTHONPATH):
# - forecaster.py (MetamodelARXForecaster)
# - forecaster_meta_adapter.py (MetaPLSWithTerms)
from forecaster import MetamodelARXForecaster

# ------------------------------
# Term parsing / builders (must match your training logic)
# ------------------------------
_INTERACTION_RE = re.compile(r"^(.*?, ge\d+)\*(.*?, ge\d+)$")

def _series_for_term(term: str, db: pd.DataFrame) -> pd.Series:
    if term == "B_0":
        return pd.Series(1.0, index=db.index)
    if term.endswith("**2"):
        base = term[:-3]
        return db[base] ** 2
    m = _INTERACTION_RE.match(term)
    if m:
        a, b = m.group(1), m.group(2)
        return db[a] * db[b]
    return db[term]

def build_x_from_terms(db: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    return pd.DataFrame({t: _series_for_term(t, db) for t in terms}, index=db.index)

def extract_base_columns_from_terms(terms: list[str]) -> list[str]:
    """Collect the raw/base column names required by x_terms."""
    bases = set()
    for t in terms:
        if t == "B_0":
            continue
        if t.endswith("**2"):
            bases.add(t[:-3])
            continue
        m = _INTERACTION_RE.match(t)
        if m:
            bases.add(m.group(1))
            bases.add(m.group(2))
            continue
        # plain base term
        bases.add(t)
    return sorted(bases)

# ------------------------------
# Paths (edit if your layout differs)
# ------------------------------
META_ARTIFACT = "models/meta/plsr_with_terms.joblib"
ARX_BUNDLE    = "models/arx/arx_model.joblib"

def main():
    assert os.path.exists(META_ARTIFACT), f"Missing {META_ARTIFACT}"
    assert os.path.exists(ARX_BUNDLE), f"Missing {ARX_BUNDLE}"

    # Peek at what the metamodel needs (x_terms) and ARX exogenous columns
    meta_art = joblib.load(META_ARTIFACT)
    x_terms = list(meta_art["x_terms"])
    base_cols = extract_base_columns_from_terms(x_terms)

    arx = joblib.load(ARX_BUNDLE)
    exog_cols = list(arx["exog_cols"])
    p = int(arx["AR_ORDER"])

    print(f"[info] Metamodel x_terms: {len(x_terms)} terms")
    print(f"[info] Base/raw columns needed by metamodel: {len(base_cols)}")
    print(f"[info] ARX exog_cols: {exog_cols} (count={len(exog_cols)})")
    print(f"[info] AR order p = {p}")

    # ------------------------------
    # Build synthetic HISTORY (180 s @ 1 Hz)
    # ------------------------------
    rng = np.random.default_rng(42)
    HIST_SEC = 180
    ts = pd.date_range("2025-01-01 12:00:00", periods=HIST_SEC, freq="s")

    hist = pd.DataFrame(index=ts)

    # Fill required base/raw columns (for metamodel term builder)
    for c in base_cols:
        # simple bounded random walk to look "process-like"
        steps = rng.normal(0, 0.01, size=HIST_SEC)
        hist[c] = np.cumsum(steps) + rng.normal(0, 0.1)

    # Ensure ARX exog columns exist (add if not already among base columns)
    for c in exog_cols:
        if c not in hist.columns:
            hist[c] = rng.normal(0, 1, size=HIST_SEC)

    # Instantiate the combined forecaster (uses your build_x_from_terms)
    fcast = MetamodelARXForecaster(
        meta_artifact_path=META_ARTIFACT,
        arx_joblib_path=ARX_BUNDLE,
        build_x_from_terms=build_x_from_terms
    )

    # Compute metamodel on the history to synthesize a plausible plant signal
    meta_hist = fcast.meta_model.predict(hist)
    # plant = meta + small noise (so residual seeds are realistic)
    hist["Tot_Resistance_mOhm"] = meta_hist + 0.01 * rng.standard_normal(HIST_SEC)

    # ------------------------------
    # A) HOLD-LAST FORECAST (no future_df)
    # ------------------------------
    print("\n[run] 30 s forecast — hold-last mode")
    out_hold = fcast.forecast(
        history_df=hist,          # already has DatetimeIndex
        future_df=None,           # hold last meta/exog
        horizon=30,
        time_col="timestamp"      # not needed since index is datetime, but harmless
    )
    print(out_hold.head(8))

    # ------------------------------
    # B) TRUE FUTURE_DF (30 rows) with gentle ramps
    # ------------------------------
    print("\n[run] 30 s forecast — with true future_df")
    FUT_SEC = 30
    future_idx = pd.date_range(ts[-1] + pd.Timedelta(seconds=1), periods=FUT_SEC, freq="s")
    future = pd.DataFrame(index=future_idx)

    # Start from last history row and add small linear ramps
    last_hist = hist.iloc[-1]

    # Exog ramps
    for c in exog_cols:
        start = float(last_hist[c])
        future[c] = np.linspace(start, start + 0.2, FUT_SEC)

    # Base/raw columns for metamodel ramps
    for c in base_cols:
        start = float(last_hist[c])
        future[c] = np.linspace(start, start + 0.1, FUT_SEC)

    out_true = fcast.forecast(
        history_df=hist,
        future_df=future,
        horizon=30,
        time_col="timestamp"
    )
    print(out_true.head(8))

    # Optional: quick sanity plots (uncomment if you want)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,4))
    # plt.plot(out_hold["timestamp"], out_hold["Tot_Resistance_dyn"], label="hold-last dyn")
    # plt.plot(out_true["timestamp"], out_true["Tot_Resistance_dyn"], label="true-future dyn")
    # plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

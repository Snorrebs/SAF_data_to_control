"""
tap_lookup.py
Maps a target arc resistance to the transformer tap changer setting
that places that operating point within reach of the electrode controller.

Background
The arc resistance R in a submerged arc furnace is determined by three things:
the transformer voltage (set by the tap changer), the arc current kA (a result
of voltage and total circuit impedance), and the electrode position which
controls the arc length.

The tap changer provides coarse control of the voltage level and therefore
the kA-R operating point. The electrode position relay or PID then provides
fine adjustment of R around that tap-set level.

This module builds a lookup table from stationary periods in the PI historian
and provides the function get_tap_for_target(R_target) which returns the
recommended tap setting for each electrode.

Usage
    from fusion.training.tap_lookup import TapLookup, build_tap_lookup

    # Build from PI data (run once, then save/load)
    lookup = build_tap_lookup(df_pi)
    lookup.save("fusion/models/tap_lookup.json")

    # Use in simulation
    lookup = TapLookup.load("fusion/models/tap_lookup.json")
    tca = lookup.get_tap(el=1, r_target=1.01)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Minimum number of stationary samples required for a tap value to be trusted.
_MIN_SAMPLES = 500

# A stationary period requires all three electrodes to have had position
# changes below this threshold for at least this many consecutive seconds.
_DPOS_THRESH  = 0.002   # m
_MIN_STATIONARY_STEPS = 30

# Tap changer column for each electrode
_TC_COL = {1: "TCA", 2: "TCB", 3: "TCC"}

class TapLookup:
    """Per-electrode tap changer lookup table.

    Stores mean R per tap value (from stationary periods) and provides
    nearest-tap queries for a target R.
    """

    def __init__(self, tables: dict[int, dict[float, float]]) -> None:
        # tables[el][tap_val] = mean_R
        self._tables = {int(k): v for k, v in tables.items()}

    # Query interface
    def get_tap(self, el: int, r_target: float) -> float:
        """Return the tap value whose mean R is closest to r_target.

        If r_target is outside the range covered by the data the nearest
        extreme value is returned.
        """
        tbl = self._tables.get(el)
        if not tbl:
            raise ValueError(f"No tap table for electrode {el}")
        taps  = np.array(sorted(tbl.keys()))
        means = np.array([tbl[t] for t in taps])
        idx   = int(np.argmin(np.abs(means - r_target)))
        return float(taps[idx])

    def get_taps_for_refs(self, r_refs: dict[int, float]) -> dict[int, float]:
        """Return {el: tap_val} for a dict of {el: R_target}."""
        return {el: self.get_tap(el, r) for el, r in r_refs.items()}

    def achievable_range(self, el: int) -> tuple[float, float]:
        """Return (R_min, R_max) achievable by varying the tap for electrode el."""
        means = list(self._tables[el].values())
        return float(min(means)), float(max(means))

    def describe(self) -> None:
        """Print a summary of the tap table for each electrode."""
        for el in sorted(self._tables):
            tbl = self._tables[el]
            taps  = sorted(tbl)
            means = [tbl[t] for t in taps]
            print(f"El{el} ({_TC_COL[el]}): "
                  f"tap range [{taps[0]}, {taps[-1]}]  "
                  f"R range [{min(means):.4f}, {max(means):.4f}] mOhm")
            for tap, r in zip(taps, means):
                print(f"  tap={tap:.1f}  R={r:.4f} mOhm")

    # Persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({str(el): {str(t): r for t, r in tbl.items()}
                        for el, tbl in self._tables.items()},
                       indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "TapLookup":
        raw = json.loads(Path(path).read_text())
        tables = {int(el): {float(t): r for t, r in tbl.items()}
                  for el, tbl in raw.items()}
        return cls(tables)

def build_tap_lookup(df: pd.DataFrame) -> TapLookup:
    """Build a TapLookup from a pre-processed PI data frame.

    The data frame must contain the columns produced by load_pi_data() in
    train_gp_v11.py / train_gp_v13.py, including El{i}_dpos_raw,
    El{i}_R_true, TCA, TCB, TCC.

    Only tap values with at least _MIN_SAMPLES stationary observations are
    included so noisy transient readings do not corrupt the table.
    """
    # Identify stationary periods
    dpos_cols = [f"El{i}_dpos_raw" for i in (1, 2, 3)]
    dpos_max  = np.column_stack([np.abs(df[c].values) for c in dpos_cols]).max(axis=1)
    stationary = dpos_max < _DPOS_THRESH

    stat_window = (pd.Series(stationary.astype(int))
                   .rolling(_MIN_STATIONARY_STEPS, min_periods=_MIN_STATIONARY_STEPS)
                   .sum() >= _MIN_STATIONARY_STEPS)
    df_stat = df[stat_window].copy()

    n = int(stat_window.sum())
    pct = 100.0 * n / len(df)
    print(f"[tap_lookup] Stationary rows: {n:,} ({pct:.1f}%)")

    tables: dict[int, dict[float, float]] = {}
    for el in (1, 2, 3):
        tc_col = _TC_COL[el]
        r_col  = f"El{el}_R_true"
        grp    = df_stat.groupby(tc_col)[r_col].agg(["mean", "count"])
        grp    = grp[grp["count"] >= _MIN_SAMPLES]
        tables[el] = {float(tap): float(row["mean"])
                      for tap, row in grp.iterrows()}
        print(f"[tap_lookup] El{el}: {len(tables[el])} reliable tap values")

    return TapLookup(tables)

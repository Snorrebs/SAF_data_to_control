#!/usr/bin/env python3
# arx_p_controller_sim.py
#
# Closed-loop ARX simulation with a proportional (P) controller on Tot_Resistance_mOhm.
# The controller updates El1/2/3_pos_m every step to track a resistance setpoint.
#
# Example:
# python arx_p_controller_sim.py \
#   --model models/arx_linear_ridge.joblib \
#   --hist-csv arx/arx_prep/model_arx_30_5_5.csv \
#   --steps 300 --dt 1 \
#   --setpoint 0.650 \
#   --kp -0.08 \
#   --pos-min 2.5 --pos-max 3.8 --dpos-max 0.015 \
#   --save-csv models/arx_pctrl_sim.csv
#

#!/usr/bin/env python3
#arx_p_controller_sim.py — P-control on resistance using ARX bundle
from __future__ import annotations
from pathlib import Path
import argparse, re
import numpy as np
import pandas as pd
from joblib import load

def tz_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def parse_lag_map(x_cols: list[str]) -> dict[str, dict[int, str]]:
    """Return base -> {lag_k: colname}. Treat unsuffixed as lag0 if present."""
    pat = re.compile(r"^(?P<base>.+?)_lag(?P<k>\d+)$")
    lag_map: dict[str, dict[int, str]] = {}
    for c in x_cols:
        m = pat.match(c)
        if m:
            base, k = m.group("base"), int(m.group("k"))
            lag_map.setdefault(base, {})[k] = c
        else:
            lag_map.setdefault(c, {})[0] = c
    return lag_map

def detect_ar_cols(x_cols: list[str], y_col: str) -> dict[int,str]:
    pat = re.compile(rf"^{re.escape(y_col)}_lag(\d+)$")
    out = {}
    for c in x_cols:
        m = pat.match(c)
        if m: out[int(m.group(1))] = c
    return dict(sorted(out.items()))

def clip_rate(u_prev, u_cmd, dmax, umin, umax):
    if np.isfinite(dmax) and dmax>0:
        delta = np.clip(u_cmd - u_prev, -dmax, dmax)
        u = u_prev + delta
    else:
        u = u_cmd
    return np.clip(u, umin, umax)

class ARXBundle:
    def __init__(self, path: Path):
        b = load(path)
        self.model     = b["model"]
        self.X_cols    = list(b["X_cols"])
        self.y_col     = b["y_col"]
        self.X_scaler  = b["scalers"]["X_scaler"]
        self.y_scaler  = b["scalers"]["y_scaler"]
        self.cfg       = b.get("prep_config", {}) or {}
        self.H         = int(self.cfg.get("horizon", 0))
        self.max_ar    = int(self.cfg.get("max_ar_lag", 0))
        self.max_xlag  = int(self.cfg.get("max_x_lag", 0))

        self.lag_map   = parse_lag_map(self.X_cols)
        self.ar_cols   = detect_ar_cols(self.X_cols, self.y_col)

    def predict_row(self, xrow: pd.DataFrame) -> float:
        X = xrow[self.X_cols].to_numpy(dtype=np.float64)
        Xz = self.X_scaler.transform(X)
        yhat_z = self.model.predict(Xz)
        return float(self.y_scaler.inverse_transform(np.asarray(yhat_z).reshape(-1,1))[0,0])

def resolve_electrode_bases(lag_map: dict[str,dict[int,str]], hint_cols: list[str]|None=None) -> list[str]:
    """
    Try to find the 3 base names for electrodes among lag_map keys.
    Prefers names containing 'El1_pos', 'El2_pos', 'El3_pos'.
    Works with your file: 'El1_pos_m_filt', etc.
    """
    bases = list(lag_map.keys())
    def pick(tag):
        cand = [b for b in bases if tag in b]
        if not cand:
            # fallback: search by regex ElN.*pos
            m = [b for b in bases if re.search(tag.replace("_", ".*"), b)]
            return m[0] if m else None
        # prefer filtered if both exist
        cand.sort(key=lambda s: (("_filt" not in s), len(s)))
        return cand[0]
    el1 = pick("El1_pos")
    el2 = pick("El2_pos")
    el3 = pick("El3_pos")
    found = [c for c in (el1, el2, el3) if c is not None]
    if len(found) != 3:
        raise RuntimeError(f"Could not auto-detect 3 electrode bases from X columns. Found: {found}")
    return found

def build_one_step_features(
    B: ARXBundle,
    last_xrow: pd.Series,
    base_bufs: dict[str, list[float]],
    y_buf: list[float],
) -> pd.DataFrame:
    """
    Build a single feature row from buffers:
    - For each base var, set lag k using base_bufs[base][-k] and lag0 from last element
    - Overwrite AR(y) lag columns from y_buf if present
    """
    x = last_xrow.to_frame().T.copy()

    for base, kmap in B.lag_map.items():
        buf = base_bufs[base]
        need = max(kmap.keys()) if kmap else 0
        if len(buf) < need+1:
            pad = [buf[0]] * (need+1 - len(buf))
            buf_ext = pad + buf
        else:
            buf_ext = buf
        if 0 in kmap:
            x.loc[:, kmap[0]] = buf_ext[-1]
        for k, col in kmap.items():
            if k==0: continue
            x.loc[:, col] = buf_ext[-(k+1)]

    if B.ar_cols and y_buf:
        need = max(B.ar_cols.keys())
        yext = ([y_buf[0]] * (need+1 - len(y_buf)) + y_buf) if len(y_buf) < need+1 else y_buf
        for k, col in B.ar_cols.items():
            x.loc[:, col] = yext[-(k+1)]

    return x[B.X_cols]  # keep exact order

def simulate_closed_loop(
    model_path: Path,
    hist_csv: Path,
    steps: int,
    dt: int,
    setpoint: float,
    kp_vals: list[float],
    pos_min: float, pos_max: float, dpos_max: float,
    save_csv: Path,
):
    B = ARXBundle(model_path)

    # Load and prep history
    df = pd.read_csv(hist_csv, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = tz_utc(df)

    # Sanity: training X columns must exist in history CSV
    missing = [c for c in B.X_cols if c not in df.columns]
    if missing:
        raise ValueError(f"History CSV missing columns required by model: {missing}")

    have_y = B.y_col in df.columns
    tail_len = max(5, B.max_ar + B.max_xlag + 2)
    hist = df[B.X_cols + ([B.y_col] if have_y else [])].iloc[-tail_len:].copy()

    # Auto-detect electrode *base* names from lag map
    pos_bases = resolve_electrode_bases(B.lag_map)  # e.g. ['El1_pos_m_filt','El2_pos_m_filt','El3_pos_m_filt']

    # Seed base buffers for ALL bases from history using their smallest available lag (prefer 0 else 1)
    base_bufs: dict[str, list[float]] = {}
    for base, kmap in B.lag_map.items():
        # choose lag0 if present, else smallest k present
        if 0 in kmap:
            col = kmap[0]
        else:
            k_min = min(kmap.keys())
            col = kmap[k_min]
        series = hist[col].astype(float).to_list()
        base_bufs[base] = series

    # Seed y buffer
    y_buf: list[float] = hist[B.y_col].astype(float).to_list() if have_y else []

    # Initial electrode positions u: if lag0 exists for base use it, else use last lag1 as proxy
    u0 = []
    for base in pos_bases:
        kmap = B.lag_map[base]
        if 0 in kmap:
            u0.append(float(hist[kmap[0]].iloc[-1]))
        else:
            k1 = 1 if 1 in kmap else min(kmap.keys())
            u0.append(float(hist[kmap[k1]].iloc[-1]))
    u = np.array(u0, dtype=float)

    kp = np.array(kp_vals if len(kp_vals)==3 else [kp_vals[0]]*3, dtype=float)

    # Last row of X to use as template
    last_xrow = hist.iloc[-1][B.X_cols].copy()

    start_ts = hist.index[-1]
    idx = pd.date_range(start_ts + pd.Timedelta(seconds=dt), periods=steps, freq=f"{dt}s", tz="UTC")

    out_rows = []
    for i in range(steps):
        # Push commanded u into the *base* buffers for the three electrode bases (this defines future lagged values)
        for j, base in enumerate(pos_bases):
            base_bufs[base].append(float(u[j]))
            # keep buffers bounded
            if len(base_bufs[base]) > 5000:
                base_bufs[base] = base_bufs[base][-5000:]

        # Build features for this step from buffers
        xrow = build_one_step_features(B, last_xrow, base_bufs, y_buf)

        # Predict resistance (yhat = y(t+H) if model has H>0)
        yhat = B.predict_row(xrow)

        # P control
        e = setpoint - yhat
        u_cmd = u + kp * e
        u = clip_rate(u, u_cmd, dpos_max, pos_min, pos_max)

        # Roll y buffer with predicted y (for AR of y)
        if B.max_ar > 0:
            y_buf.append(float(yhat))
            if len(y_buf) > max(1, B.max_ar):
                y_buf = y_buf[-B.max_ar:]

        out_rows.append({
            "timestamp": idx[i],
            "y_pred_mOhm": float(yhat),
            "e": float(e),
            pos_bases[0]: float(u[0]),
            pos_bases[1]: float(u[1]),
            pos_bases[2]: float(u[2]),
        })

    out = pd.DataFrame(out_rows).set_index("timestamp")
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_csv)
    print(f"[save] {save_csv}")
    print(out.head(8))
    return out

# ---------------- CLI ----------------

def _floats3(s: str) -> list[float]:
    parts = [p for p in s.replace(",", " ").split() if p]
    vals = [float(v) for v in parts]
    if len(vals) not in (1,3):
        raise argparse.ArgumentTypeError("Provide one value or three values.")
    return vals

def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop ARX P-controller on Tot_Resistance_mOhm.")
    p.add_argument("--model", type=Path, required=True, help="models/arx_linear_ridge.joblib")
    p.add_argument("--hist-csv", type=Path, required=True, help="arx/arx_prep/model_arx_30_5_5.csv")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--dt", type=int, default=1)
    p.add_argument("--setpoint", type=float, required=True)
    p.add_argument("--kp", type=_floats3, required=True, help="Kp (one or three values). Sign sets control direction.")
    p.add_argument("--pos-min", type=float, default=-np.inf)
    p.add_argument("--pos-max", type=float, default= np.inf)
    p.add_argument("--dpos-max", type=float, default=np.inf)
    p.add_argument("--save-csv", type=Path, default=Path("models/arx_pctrl_sim.csv"))
    return p.parse_args()

def main():
    a = parse_args()
    simulate_closed_loop(
        model_path=a.model,
        hist_csv=a.hist_csv,
        steps=a.steps,
        dt=a.dt,
        setpoint=a.setpoint,
        kp_vals=a.kp,
        pos_min=a.pos_min, pos_max=a.pos_max, dpos_max=a.dpos_max,
        save_csv=a.save_csv,
    )

if __name__ == "__main__":
    main()

# forecaster.py
from __future__ import annotations
import numpy as np, pandas as pd, joblib
from typing import Optional, List
from forecaster_meta_adapter import MetaPLSWithTerms

class MetamodelARXForecaster:
    def __init__(self, meta_artifact_path: str, arx_joblib_path: str, build_x_from_terms):
        self.meta_model = MetaPLSWithTerms(meta_artifact_path, build_x_from_terms)
        bundle = joblib.load(arx_joblib_path)
        self.beta = np.array(bundle["beta"]).reshape(-1)
        self.p = int(bundle["AR_ORDER"])
        self.exog_cols: List[str] = list(bundle["exog_cols"])
        self.exog_scaler = bundle["scaler"]
        self.ar_scaler = bundle["ar_scaler"]

        self._b0 = self.beta[0]
        self._b_ar = self.beta[1:1+self.p]
        self._b_ex = self.beta[1+self.p:]
        if self._b_ex.size != len(self.exog_cols):
            raise ValueError("ARX exog size mismatch.")

        self._mu_ar = np.asarray(self.ar_scaler.mean_, dtype=np.float64)
        self._sd_ar = np.asarray(self.ar_scaler.scale_, dtype=np.float64)
        self._mu_ex = np.asarray(self.exog_scaler.mean_, dtype=np.float64)
        self._sd_ex = np.asarray(self.exog_scaler.scale_, dtype=np.float64)

    def _ensure(self, df, cols, ctx):
        miss = [c for c in cols if c not in df.columns]
        if miss: raise ValueError(f"Missing columns for {ctx}: {miss}")

    def _maybe_time_index(self, df: pd.DataFrame, time_col: Optional[str]):
        if isinstance(df.index, pd.DatetimeIndex): return df.sort_index(), True
        if time_col and time_col in df.columns:
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            return df.sort_values(time_col).set_index(time_col), True
        return df.copy(), False

    def forecast(self, history_df: pd.DataFrame, future_df: Optional[pd.DataFrame]=None,
                 horizon: int=30, time_col: Optional[str]="timestamp") -> pd.DataFrame:
        hist, has_time = self._maybe_time_index(history_df, time_col)
        self._ensure(hist, self.exog_cols, "history exog")

        # meta for history (seeding residuals)
        meta_hist = self.meta_model.predict(hist)  # needs base/raw columns present in hist
        if "Tot_Resistance_mOhm" in hist.columns:
            resid_hist = (hist["Tot_Resistance_mOhm"] - meta_hist).dropna()
            if len(resid_hist) >= self.p:
                y_hist = resid_hist.iloc[-self.p:][::-1].to_numpy(np.float64)
            else:
                y_hist = np.zeros(self.p)
        else:
            y_hist = np.zeros(self.p)

        # FUTURE sources
        if future_df is not None:
            fut, _ = self._maybe_time_index(future_df, time_col)
            self._ensure(fut, self.exog_cols, "future exog")
            meta_future = self.meta_model.predict(fut).to_numpy()[:horizon]
            exog_future = fut[self.exog_cols].to_numpy(np.float64)[:horizon]
            out_index = fut.index[:horizon] if has_time else None
        else:
            # hold-last values
            meta_future = np.full(horizon, float(meta_hist.dropna().iloc[-1]))
            last_exog = hist[self.exog_cols].iloc[-1].to_numpy(np.float64)
            exog_future = np.tile(last_exog, (horizon, 1))
            out_index = pd.date_range(hist.index[-1] + pd.Timedelta(seconds=1), periods=horizon, freq="s") if has_time else None

        # recursive residual forecast
        res_preds = []
        yh = y_hist.copy()
        for k in range(horizon):
            y_std = (yh - self._mu_ar) / self._sd_ar
            ex_std = (exog_future[k] - self._mu_ex) / self._sd_ex
            y_next = self._b0 + float(np.dot(y_std, self._b_ar)) + float(np.dot(ex_std, self._b_ex))
            res_preds.append(y_next)
            if self.p > 1: yh[1:] = yh[:-1]
            yh[0] = y_next

        res_preds = np.asarray(res_preds, np.float64)
        total = meta_future + res_preds
        out = pd.DataFrame({"Tot_Resistance_meta": meta_future,
                            "residual_pred": res_preds,
                            "Tot_Resistance_dyn": total})
        if out_index is not None: out.insert(0, "timestamp", out_index)
        return out

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt




def butter_lowpass(cutoff_Hz: float, fs_Hz: float, order: int = 4):
    nyq = 0.5 * fs_Hz
    wc = cutoff_Hz / nyq
    b, a = butter(order, wc, btype="low", analog=False)
    return b, a




def apply_filter(series: pd.Series, cutoff_Hz: float, fs_Hz: float, order: int) -> pd.Series:
    if series.isna().all():
        return series
    
    b, a = butter_lowpass(cutoff_Hz, fs_Hz, order)
    x = series.astype(float).to_numpy()
    y = filtfilt(b, a, x, method="gust")
    return pd.Series(y, index=series.index, name=f"{series.name}_filt")




def filter_dataframe(df: pd.DataFrame, cols: list[str], cutoff_Hz: float, fs_Hz: float, order: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_filt"] = apply_filter(out[c], cutoff_Hz, fs_Hz, order)
    return out
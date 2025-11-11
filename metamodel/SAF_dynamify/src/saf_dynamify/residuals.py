import pandas as pd




def compute_residual_targets(df: pd.DataFrame, y_names: list[str]) -> pd.DataFrame:
    """target = plant_filtered − meta_unfiltered
    Expects columns: y (filtered) and meta_y (from metamodel.predict_meta)
    """
    tgt = {}
    for y in y_names:
        yf = f"{y}_filt" if f"{y}_filt" in df.columns else y
        my = f"meta_{y}"
        tgt[f"res_{y}"] = df[yf] - df[my]
    return pd.DataFrame(tgt, index=df.index)




def build_lagged_matrix(df: pd.DataFrame, cols: list[str], lags: int) -> pd.DataFrame:
    out = {}
    for c in cols:
        for k in range(1, lags + 1):
            out[f"{c}_t-{k}"] = df[c].shift(k)
    return pd.DataFrame(out, index=df.index)
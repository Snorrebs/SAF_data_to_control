from pathlib import Path
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from constants_metamodel import CW_thk, rCW, rSiC, BASE_X_COLS

def build_baseX(df: pd.DataFrame) -> pd.DataFrame:

    baseX = pd.DataFrame(index=df.index, data={
        'RMS Voltage at Transformer (V), ge1': df['RMS_V_transformer'],
        'El1 pos (m), ge2': df['El1_pos_m'],
        'El2 pos (m), ge3': df['El2_pos_m'],
        'El3 pos (m), ge4': df['El3_pos_m'],
        'CW1 Thickness (m), ge62': CW_thk[0],
        'CW2 Thickness (m), ge63': CW_thk[1],
        'CW3 Thickness (m), ge64': CW_thk[2],
        'res. CW 1 (mΩ*m), ge6': rCW[0],
        'res. CW 2 (mΩ*m), ge7': rCW[1],
        'res. CW 3 (mΩ*m), ge8': rCW[2],
        'res. SiC12 (mΩ*m), ge10': rSiC[0],
        'res. SiC23 (mΩ*m), ge11': rSiC[1],
        'res. SiC31 (mΩ*m), ge12': rSiC[2],
    })
    return baseX[BASE_X_COLS]

def expand_XIS_104(baseX: pd.DataFrame, poly: PolynomialFeatures | None = None) -> tuple[pd.DataFrame, PolynomialFeatures]:

    if poly is None:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        # dummy fit to lock in deterministic feature order
        poly.fit(baseX.values[:2])
    X_poly = poly.transform(baseX.values)
    # expose feature names
    try:
        names = poly.get_feature_names_out(BASE_X_COLS)
    except Exception:
        names = [f"x{i}" for i in range(X_poly.shape[1])]
    XIS = pd.DataFrame(X_poly, index=baseX.index, columns=names)
    return XIS, poly

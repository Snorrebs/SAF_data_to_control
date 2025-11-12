from typing import Tuple

# Defaults for metamodel inputs
CW_thk: Tuple[float, float, float] = (0.25, 0.25, 0.25)
rCW:    Tuple[float, float, float] = (2.3,  2.3,  2.3 )
rSiC:   Tuple[float, float, float] = (30.0, 30.0, 30.0)

BASE_X_COLS = [
    'RMS Voltage at Transformer (V), ge1',
    'El1 pos (m), ge2', 'El2 pos (m), ge3', 'El3 pos (m), ge4',
    'CW1 Thickness (m), ge62', 'CW2 Thickness (m), ge63', 'CW3 Thickness (m), ge64',
    'res. CW 1 (mΩ*m), ge6', 'res. CW 2 (mΩ*m), ge7', 'res. CW 3 (mΩ*m), ge8',
    'res. SiC12 (mΩ*m), ge10', 'res. SiC23 (mΩ*m), ge11', 'res. SiC31 (mΩ*m), ge12',
]

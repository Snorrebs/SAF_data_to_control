# Training Data Folder

Put your CSV data files in this folder before running `train_arx.py` or `train_gp.py`.

## Required columns

| Column name | Description |
|---|---|
| `El1_Resistance_mOhm_filt`, `El2_...`, `El3_...` | Arc resistance per electrode (mOhm) |
| `El1_pos_m`, `El2_...`, `El3_...` | Electrode position (m) |
| `El1_kA_filt`, `El2_...`, `El3_...` | Arc current per electrode (kA) |
| `El1_CalcReac_filt`, `El2_...`, `El3_...` | Arc reactance per electrode (mOhm) |
| `RMS_V_transformer_filt` | Transformer RMS voltage (V) |
| `TCA`, `TCB`, `TCC` | Tap changer positions |

## If your columns are named differently

Edit the `COLUMN_MAP` dict at the top of `train_arx.py` (and `train_gp.py`):

```python
COLUMN_MAP = {
    "r":    "YourResistanceColumnName_{i}",  # {i} is replaced by 1, 2, 3 for each electrode
    "pos":  "YourPositionColumnName_{i}",
    "ka":   "YourCurrentColumnName_{i}",
    "reac": "YourReactanceColumnName_{i}",
    "v":    "YourVoltageColumnName",
    "tca":  "YourTapChangerA",
    "tcb":  "YourTapChangerB",
    "tcc":  "YourTapChangerC",
}
```

## Multiple files

You can put more than one CSV in this folder. `train_arx.py` and `train_gp.py`
load all `.csv` files found here and chain them together before training.

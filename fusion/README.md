# SAF Fusion

Joint ARX simulator with per-electrode GP correction for all three electrodes.
Used as the plant model for closed-loop simulation and VRFT controller design.

Six variants are included. Each is a matched ARX and GP pair trained on the same dataset. V7 is the recommended choice for PI furnace data.

- `v7` — V6 joint ARX with two-stage correction: linear model first, then GP on the remaining nonlinear residual. Best on PI data.
- `v6` — V6 joint ARX with a single debiased GP correction. Nearly as good as V7, slightly simpler.
- `pi_512` — earlier ARX trained on PI data, middle 80 percent
- `txt2026_512` — ARX trained on 2026 txt plant data
- `combined_512` — ARX trained on combined PI and txt data, Matern 3/2 kernel
- `combined_deep_512` — same combined ARX, deep spectral mixture kernel

---

## Step 1 — Model files

Place all model files in `fusion/models/`. Each variant needs its ARX plus one GP file per electrode (three electrodes total). V7 also needs the linear residual files.

**Variant `v7` (recommended for PI furnace)**
```
arx_joint_v6.joblib
gp_el1_v7.pt
gp_el2_v7.pt
gp_el3_v7.pt
linear_residual_el1.joblib
linear_residual_el2.joblib
linear_residual_el3.joblib
```

**Variant `v6`**
```
arx_joint_v6.joblib
gp_el1_v6.pt
gp_el2_v6.pt
gp_el3_v6.pt
```

**Variant `txt2026_512`**
```
arx_joint_txt2026.joblib
gp_el1_txt2026_512.pt
gp_el2_txt2026_512.pt
gp_el3_txt2026_512.pt
```

**Variant `pi_512`**
```
arx_joint_pi_v3.joblib
gp_el1_pi_512.pt
gp_el2_pi_512.pt
gp_el3_pi_512.pt
```

**Variant `combined_deep_512`**
```
arx_joint_combined_v3.joblib
gp_el1_combined_deep_512.pt
gp_el2_combined_deep_512.pt
gp_el3_combined_deep_512.pt
```

**Variant `combined_512`**
```
arx_joint_combined_v3.joblib
gp_el1_combined_512.pt
gp_el2_combined_512.pt
gp_el3_combined_512.pt
```

To switch variants, either change `_GP_VARIANT` at the top of `run_closed_loop.py`, or pass `gp_variant` directly to `run_closed_loop_from_config`:

```python
# Set the default variant for all calls in this session:
_GP_VARIANT = "v7"

# Or override per call:
run_closed_loop_from_config(..., gp_variant="v7")
run_closed_loop_from_config(..., gp_variant="txt2026_512")
```

---

## Step 2 — Python dependencies

```
pip install numpy pandas scikit-learn joblib torch gpytorch matplotlib
```

---

## Step 3 — Drop-in usage

Call `run_closed_loop_from_config` directly from the fusion package:

```python
from fusion.run_closed_loop import run_closed_loop_from_config

run_closed_loop_from_config(
    ref_csv="path/to/reference.csv",
    controller_name="pid",
    controller_config="path/to/PID_params.csv",
    out_csv="path/to/output.csv",
    dt=1.0,
)
```

**Output CSV columns (66 total):**

| Column | Description |
|---|---|
| `t_s` | time (s) |
| `y1`, `y2`, `y3` | predicted arc resistance per electrode (mOhm) |
| `r1`, `r2`, `r3` | reference per electrode (mOhm) |
| `u1`, `u2`, `u3` | electrode position commands (m) |
| `e1`, `e2`, `e3` | controller error (reference - y_pred) |
| `v_transformer` | transformer RMS voltage (V) |
| `gp_var1`, `gp_var2`, `gp_var3` | GP predictive variance per electrode (mOhm²) — see note below |
| `El{1,2,3}_y_filt_lag{1,2,3}` | ARX resistance lag registers per electrode |
| `El{1,2,3}_kA_filt_lag{1,2,3}` | ARX current lag registers per electrode |
| `El{1,2,3}_CalcReac_filt_lag{1,2,3}` | ARX reactance lag registers per electrode |
| `El{1,2,3}_pos_m_lag{1,2,3}` | ARX position lag registers per electrode |
| `El{1,2,3}_dpos_mps_filt_lag{1,2,3}` | ARX velocity lag registers per electrode |
| `RMS_V_transformer_filt_lag1` | ARX transformer voltage lag register |
| `TCA`, `TCB`, `TCC` | tap changer positions |

> **GP posterior variance** (`gp_var1/2/3`): the variance of the GP correction term at each step, in mOhm².
> Take `sqrt(gp_var)` to get the one-sigma uncertainty on the GP correction (e.g. var=0.004 → ±0.063 mOhm).
> A rising variance indicates the simulator is moving out of the GP's training distribution — the ARX prediction is being trusted more and the GP correction less.
> Columns are placed after `v_transformer`; scroll right in Excel to find them.

**Reference CSV format:** columns `r1`, `r2`, `r3` for per-electrode references, or a single `reference` column broadcast to all three.

---

## Using from VRFT v5.py

Change one import line in `VRFT v5.py`:

```python
# Old:
from run_simulation.scripts.run_closed_loop import run_closed_loop_from_config

# New:
from fusion.run_closed_loop import run_closed_loop_from_config
```

Everything else stays the same. The output CSV will now include `gp_var1/2/3` and all ARX state columns.

---

## PID example

`example_pid_simulation.py` runs a 300-step closed-loop step-response for all three electrodes using the V7 model. Run from `SAF_data_to_control/`:

```
python fusion/example_pid_simulation.py
```

Outputs to `fusion/results/example_pid/`: `closed_loop_result.csv`, `resistance.pdf`, `positions.pdf`.

To test a different variant without editing the script, you can also call from Python:

```python
from fusion.run_closed_loop import run_closed_loop_from_config
df = run_closed_loop_from_config(
    ref_csv           = "path/to/reference.csv",
    controller_name   = "pid",
    controller_config = "path/to/PID_params.csv",
    out_csv           = "path/to/output.csv",
    gp_variant        = "v7",   # or "v6", "pi_512", "txt2026_512", etc.
)
```

---

## Other controller examples

`example_rule_controller.py` simulates the real plant's R deadband step controller for all three electrodes. Run from `SAF_data_to_control/`:

```
python fusion/example_rule_controller.py
```

Results saved to `fusion/results/rule_controller.csv` and `.pdf`.

---

## Retraining models from your own data

1. Put CSV files in `fusion/data/` (see `fusion/data/README.md` for required columns)
2. Run `fusion/train_arx.py` to retrain the ARX
3. Run `fusion/train_gp.py` to retrain the GP corrections

If your column names differ from the defaults, edit `COLUMN_MAP` at the top of `train_arx.py` and `train_gp.py`.

---

## Package structure

```
fusion/
  run_closed_loop.py          3-electrode closed-loop simulation entry point
  example_rule_controller.py  rule-based controller demo
  train_arx.py                retrain ARX from CSV data
  train_gp.py                 retrain GP corrections
  models/                     trained model files (8 total, see Step 1)
  simulators/
    saf_simulator.py          3-electrode ARX simulator
    plant.py                  single-electrode Plant with GP correction
  training/
    gp_loader.py              load GP bundles, single-sample prediction
    arx_model.py              ReducedRankRidge class
  data/
    README.md                 required column format
```

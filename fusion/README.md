# SAF Fusion

Joint ARX simulator with per-electrode GP correction for all three electrodes.
Used as the plant model for closed-loop simulation and VRFT controller design.

Two variants are included — each is a matched ARX + GP pair trained on the same dataset:

- `txt2026_512` — trained on 2026 txt plant data
- `pi_512` — trained on PI data, middle 80%

---

## Step 1 — Model files

Eight files are included in `fusion/models/`:

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

To switch variants, change `_GP_VARIANT` at the top of `run_closed_loop.py` — the matching ARX is selected automatically:

```python
_GP_VARIANT = "txt2026_512"   # 2026 txt data
_GP_VARIANT = "pi_512"        # PI data middle-80%
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

**Output CSV columns:**

| Column | Description |
|---|---|
| `t_s` | time (s) |
| `y1`, `y2`, `y3` | predicted arc resistance per electrode (mOhm) |
| `r1`, `r2`, `r3` | reference per electrode (mOhm) |
| `u1`, `u2`, `u3` | electrode position commands (m) |
| `e1`, `e2`, `e3` | controller error (reference - y_pred) |
| `v_transformer` | transformer RMS voltage (V) |

**Reference CSV format:** columns `r1`, `r2`, `r3` for per-electrode references, or a single `reference` column broadcast to all three.

---

## Accessing GP mean and uncertainty directly

```python
import sys, os
sys.path.insert(0, r"C:\path\to\SAF_data_to_control")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
from fusion.simulators.saf_simulator import SaFSimulator, build_init_row_from_scalars
from fusion.simulators.plant import Plant
from fusion.training.gp_loader import load_gp_bundle

arx = joblib.load(r"fusion/models/arx_joint_txt2026.joblib")
gp  = load_gp_bundle(r"fusion/models/gp_el1_txt2026_512.pt")

row = build_init_row_from_scalars(
    pos=1.04, r=1.006, ka=65.0, rx=0.82, v=165.0,
    arx_bundle=arx, electrode=1,
)

sim   = SaFSimulator(arx, row, electrode=1)
plant = Plant(sim, gp, clip_delta=0.15)

for step in range(100):
    y_pred = plant.predict_next_y()
    correction  = plant.gp_mean   # additive GP correction (mOhm)
    uncertainty = plant.gp_std    # one-sigma uncertainty (mOhm)
    plant.advance(u_new=1.04, y_new=y_pred)
```

Pass `gp_bundle=None` to run on ARX only:

```python
plant = Plant(sim, gp_bundle=None)
```

---

## Rule-based controller example

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

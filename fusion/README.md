# SAF Fusion

Joint ARX simulator with per-electrode GP correction for all three electrodes.
Used as the plant model for closed-loop simulation and VRFT controller design.

Eight variants are available. Each is a matched ARX and GP pair trained on the same dataset.
The active variant is set with `_GP_VARIANT` at the top of `run_closed_loop.py`.

---

## Model files

All model files live in `fusion/models/`. The currently active default is `v9`.

**Variant v9** (step-episode filtered ARX, recommended)
```
arx_joint_v9.joblib
gp_el1_v9.pt   gp_el2_v9.pt   gp_el3_v9.pt
```

**Variant v8** (full-dataset retrain, 10/80/10 split)
```
arx_joint_v8.joblib
gp_el1_v8.pt   gp_el2_v8.pt   gp_el3_v8.pt
```

**Variant v7** (two-stage: linear correction then GP)
```
arx_joint_v6.joblib
gp_el1_v7.pt   gp_el2_v7.pt   gp_el3_v7.pt
linear_residual_el1.joblib   linear_residual_el2.joblib   linear_residual_el3.joblib
```

**Variant v6** (joint ARX with debiased GP)
```
arx_joint_v6.joblib
gp_el1_v6.pt   gp_el2_v6.pt   gp_el3_v6.pt
```

**Variant txt2026_512** (ARX trained on 2026 txt plant data)
```
arx_joint_txt2026.joblib
gp_el1_txt2026_512.pt   gp_el2_txt2026_512.pt   gp_el3_txt2026_512.pt
```

**Variant pi_512** (ARX trained on PI data, middle 80%)
```
arx_joint_pi_v3.joblib
gp_el1_pi_512.pt   gp_el2_pi_512.pt   gp_el3_pi_512.pt
```

**Variant combined_512** (ARX trained on PI and txt combined, Matern32 kernel)
```
arx_joint_combined_v3.joblib
gp_el1_combined_512.pt   gp_el2_combined_512.pt   gp_el3_combined_512.pt
```

**Variant combined_deep_512** (ARX trained on PI and txt combined, deep kernel)
```
arx_joint_combined_v3.joblib
gp_el1_combined_deep_512.pt   gp_el2_combined_deep_512.pt   gp_el3_combined_deep_512.pt
```

To switch variants, change `_GP_VARIANT` at the top of `run_closed_loop.py`. The matching
ARX file is picked up automatically:

```python
_GP_VARIANT = "v9"                 # step-episode filtered, recommended
_GP_VARIANT = "v8"                 # full-dataset retrain
_GP_VARIANT = "v7"                 # two-stage linear + GP
_GP_VARIANT = "v6"                 # joint ARX, debiased GP
_GP_VARIANT = "txt2026_512"        # 2026 txt data
_GP_VARIANT = "pi_512"             # PI data middle-80%
_GP_VARIANT = "combined_512"       # PI + txt combined, Matern32 kernel
_GP_VARIANT = "combined_deep_512"  # PI + txt combined, deep kernel
```

---

## Python dependencies

```
pip install numpy pandas scikit-learn joblib torch gpytorch matplotlib pysindy
```

---

## Drop-in usage

Call `run_closed_loop_from_config` from the fusion package:

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

You can override the variant for a single call without changing the module default:

```python
run_closed_loop_from_config(..., gp_variant="v8")
```

**Supported controller types:**

- `pid` - velocity-form PID with block-diagonal gain matrix (one set of Kp/Ki/Kd per electrode)
- `relay` - step-and-wait relay matching the real plant controller logic
- `open_loop` - pre-specified position trajectory
- `generalized_controller` - custom state-space controller
- `pid_fullspace` - full-state PID

---

## Output CSV columns

| Column | Description |
|---|---|
| `t_s` | time (s) |
| `y1`, `y2`, `y3` | predicted arc resistance per electrode (mOhm) |
| `r1`, `r2`, `r3` | reference per electrode (mOhm) |
| `u1`, `u2`, `u3` | electrode position commands (m) |
| `e1`, `e2`, `e3` | controller error, reference minus y_pred |
| `v_transformer` | transformer RMS voltage (V) |
| `gp_var1/2/3` | GP predictive variance per electrode (mOhm^2) |
| `norm_var1/2/3` | normalised epistemic uncertainty, 0 to 1 (see note) |
| `ind_dist1/2/3` | distance to nearest GP inducing point in standardised space |
| `El{1,2,3}_y_filt_lag{1,2,3}` | ARX resistance lag registers |
| `El{1,2,3}_kA_filt_lag{1,2,3}` | ARX current lag registers |
| `El{1,2,3}_CalcReac_filt_lag{1,2,3}` | ARX reactance lag registers |
| `El{1,2,3}_pos_m_lag{1,2,3}` | ARX position lag registers |
| `El{1,2,3}_dpos_mps_filt_lag{1,2,3}` | ARX velocity lag registers |
| `RMS_V_transformer_filt_lag1` | ARX transformer voltage lag register |
| `TCA`, `TCB`, `TCC` | tap changer positions |

**On norm_var:** `norm_var` is the GP epistemic uncertainty normalised by the prior output scale.
A value near 0 means the model is confident the current operating point is inside the training
distribution. A value near 1 means the point is outside it (out of distribution). When the mean
norm_var across all three electrodes exceeds 0.5, the OOD gate activates: the controller holds
the previous position and integrators are frozen until confidence recovers. The threshold is set
by `_OOD_GATE_THRESHOLD` in `run_closed_loop.py`.

**Reference CSV format:** use columns `r1`, `r2`, `r3` for per-electrode references,
or a single `reference` column to broadcast the same value to all three electrodes.

---

## Using from VRFT v5.py

Change one import line in `VRFT v5.py`:

```python
# Old:
from run_simulation.scripts.run_closed_loop import run_closed_loop_from_config

# New:
from fusion.run_closed_loop import run_closed_loop_from_config
```

Everything else stays the same. The output CSV will include all GP uncertainty columns
and the full ARX state in addition to the standard columns.

---

## PID example

`example_pid_simulation.py` runs a 300-step closed-loop step-response for all three electrodes.
Run from `SAF_data_to_control/`:

```
python fusion/example_pid_simulation.py
```

Outputs to `fusion/results/example_pid/`: `closed_loop_result.csv`, `resistance.pdf`, `positions.pdf`.

---

## Other controller examples

`example_rule_controller.py` simulates the real plant deadband step controller for all three electrodes.
Run from `SAF_data_to_control/`:

```
python fusion/example_rule_controller.py
```

Results saved to `fusion/results/rule_controller.csv` and `.pdf`.

---

## Retraining models from your own data

1. Put CSV files in `fusion/data/` (see `fusion/data/README.md` for required columns)
2. Run `fusion/train_arx.py` to retrain the ARX
3. Run `fusion/train_gp.py` to retrain the GP corrections

If your column names differ from the defaults, edit `COLUMN_MAP` at the top of `train_arx.py`
and `train_gp.py`.

---

## Package structure

```
fusion/
  run_closed_loop.py          3-electrode closed-loop simulation entry point
  example_pid_simulation.py   PID controller demo
  example_rule_controller.py  rule-based controller demo
  train_arx.py                retrain ARX from CSV data
  train_gp.py                 retrain GP corrections
  models/                     trained model files (all variants, see above)
  controllers/
    relay.py                  step-and-wait relay controller
  simulators/
    saf_simulator.py          3-electrode ARX simulator
    plant.py                  single-electrode plant with GP correction
  training/
    gp_loader.py              load GP bundles, single-sample prediction, OOD certainty
    arx_model.py              ReducedRankRidge class
  data/
    README.md                 required column format
```

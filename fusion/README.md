# SAF Fusion

Joint ARX simulator with per-electrode GP correction for a three-electrode
submerged arc furnace. Used as the plant model for closed-loop simulation and
VRFT controller design.

The active model variant is set with `_GP_VARIANT` at the top of
`run_closed_loop.py`. Each variant is a matched ARX and GP pair.

---

## Model variants

All model files live in `fusion/models/`.

| Variant | ARX file | Description |
|---|---|---|
| `v6` | `arx_joint_v6.joblib` | Joint ARX, debiased GP |
| `v7` | `arx_joint_v6.joblib` | Two-stage: linear correction then GP |
| `v8` | `arx_joint_v8.joblib` | Full-dataset retrain (10/80/10 split) |
| `v9` | `arx_joint_v9.joblib` | Step-episode filtered ARX. Relay baseline |
| `rollout` | `arx_joint_v9.joblib` | SVGP trained on H=1000 rollout windows |
| `v11` | `arx_joint_pi_v3.joblib` | One-step training, no step_in_window feature |
| `v12` | `arx_joint_v12.joblib` | SEM-weighted rollout, retrained ARX |
| `v13` | `arx_joint_v13.joblib` | V12 without own-R lags |
| `v14` | `arx_joint_v14.joblib` | Per-electrode Ridge, delta-R target |
| `v15` | `arx_joint_v15.joblib` | Delta-R, 5 lags. PID baseline |

GP files follow the pattern `gp_el{1,2,3}_{variant}.pt`.

`tap_lookup.json` maps R setpoints to transformer tap positions for
automatic tap selection at simulation start.

---

## Usage

### Standard closed-loop (three independent controllers)

```python
from fusion.run_closed_loop import run_closed_loop_from_config

run_closed_loop_from_config(
    ref_csv           = "reference.csv",
    controller_name   = "pid",
    controller_config = "PID_params.csv",
    out_csv           = "result.csv",
    gp_variant        = "v9",
)
```

### Locked single-mover (one electrode moves per step)

Only the electrode furthest from its setpoint actuates at each step.
This prevents cross-electrode coupling instability with PID control.

```python
from fusion.run_locked_closed_loop import run_locked_closed_loop_from_config

run_locked_closed_loop_from_config(
    ref_csv           = "reference.csv",
    controller_config = "PID_params.csv",
    out_csv           = "result.csv",
    controller_name   = "pid",
    gp_variant        = "v15",
    gp_scale          = 0.0,
    warmup_hold_steps = 150,
    deadband          = 0.0,
)
```

`gp_scale=0.0` disables the GP correction (ARX-only). This gives better
closed-loop performance for V15 because the GP was trained on relay-controlled
data and introduces bias under PID regulation.

**Supported controllers:** `pid`, `relay`, `open_loop`

---

## PID tuning

`tune_pid_v13.py` runs a full PID tuning cycle: open-loop step test,
gain derivation, locked-PID validation with a reference step for El1,
and a relay comparison across variants.

```
python fusion/tune_pid_v13.py
```

Outputs to `fusion/results/pid_tuning_v15/`.

---

## Model accuracy comparison

`compare_models_openloop.py` runs open-loop rollout evaluation on historical
relay data to compare prediction accuracy across variants.

```
python fusion/compare_models_openloop.py
```

Outputs to `fusion/results/model_comparison/`.

---

## Controller examples

```
python fusion/example_pid_simulation.py
python fusion/example_relay_rollout.py
python fusion/example_rule_controller.py
```

---

## Retraining

| Script | What it trains |
|---|---|
| `train_gp_v15.py` | V15 delta-R ARX + SVGP (rollout dataset) |
| `train_gp.py` | Legacy GP trainer |
| `archive/train_gp_v14.py` | V14 per-electrode Ridge delta-R ARX (predecessor to V15) |
| `archive/train_gp_rollout.py` | Rollout SVGP variant |

Training data path is set at the top of each script.

---

## Output CSV columns

| Column | Description |
|---|---|
| `t_s` | Time (s) |
| `y1`, `y2`, `y3` | Predicted arc resistance per electrode (mOhm) |
| `r1`, `r2`, `r3` | Reference per electrode (mOhm) |
| `u1`, `u2`, `u3` | Electrode position commands (m) |
| `e1`, `e2`, `e3` | Error: reference minus predicted R |
| `v_transformer` | Transformer RMS voltage (V) |

Reference CSV uses columns `r1, r2, r3` (per-electrode) or a single
`reference` column broadcast to all three.

---

## Drop-in replacement for VRFT v5.py

```python
# Old:
from run_simulation.scripts.run_closed_loop import run_closed_loop_from_config

# New:
from fusion.run_closed_loop import run_closed_loop_from_config
```

---

## Package structure

```
fusion/
  run_closed_loop.py            closed-loop simulation entry point
  run_locked_closed_loop.py     single-mover locked PID/relay simulation
  tune_pid_v13.py               PID tuning and relay comparison for V15
  compare_models_openloop.py    open-loop model accuracy benchmark
  example_pid_simulation.py     PID example
  example_relay_rollout.py      relay example
  example_rule_controller.py    rule-based controller example
  train_gp_v15.py               V15 ARX + SVGP training (rollout)
  models/                       trained model files (see table above)
  controllers/
    relay.py                    step-and-wait relay controller
  simulators/
    saf_simulator.py            three-electrode ARX simulator
    plant.py                    single-electrode plant with GP correction
  training/
    delta_arx.py                DeltaARXWrapper for per-electrode Ridge models
    gp_loader.py                GP bundle loader and OOD certainty
    tap_lookup.py               R setpoint to transformer tap mapping
  archive/
    run_closed_loop_rollout.py  legacy rollout entry point
    train_gp_v14.py             V14 training (predecessor to V15)
    train_gp_rollout.py         rollout SVGP training
  data/
    README.md                   required data column format
```

---

## Dependencies

```
pip install numpy pandas scikit-learn joblib torch gpytorch matplotlib
```

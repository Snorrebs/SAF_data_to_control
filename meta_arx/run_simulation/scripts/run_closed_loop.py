from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_simulation.closed_loop.arx_state import load_arx_bundle, load_initial_state
from run_simulation.closed_loop.closed_loop_sim import run_closed_loop, run_coupled_closed_loop
from run_simulation.closed_loop.controller_registry import make_controllers
from run_simulation.closed_loop.reference_converter import ReferenceConverter


RESISTANCE_MODEL_PATH = Path("run_simulation/models/step9_n10_2sls_varx_model.joblib")
RESISTANCE_HIST_CSV = Path("run_simulation/init_data/step9_n10_2sls_varx_init.csv")

CURRENT_MODEL_PATH = Path("run_simulation/models/step11_current_varx_model.joblib")
CURRENT_HIST_CSV = Path("run_simulation/init_data/step11_current_varx_init.csv")

# Fallback current trajectory parameters
KA_MEANS = np.array([129.7, 126.5, 123.6])
KA_AMP = 0.0001
PERLIN_SCALE = 0.005
PERLIN_OCT = 4


def make_ka_exog(n: int, disturbed: bool, warmup_len: int = 300) -> dict[str, np.ndarray]:
    """Build a fallback kA trajectory for all three electrodes."""
    exog: dict[str, np.ndarray] = {}
    if disturbed:
        try:
            from noise import pnoise1
        except ImportError as exc:
            raise ImportError(
                "The 'noise' package is required for Perlin kA disturbance. "
                "Install with: pip install noise"
            ) from exc
        for i in range(3):
            exog[f"kA{i+1}"] = np.array([
                KA_MEANS[i] + KA_AMP * pnoise1(
                    (warmup_len + k) * PERLIN_SCALE + i * 100,
                    octaves=PERLIN_OCT,
                )
                for k in range(n)
            ])
    else:
        for i in range(3):
            exog[f"kA{i+1}"] = np.full(n, KA_MEANS[i])
    return exog

def make_ka_noise(
    n: int,
    disturbed: bool,
    amp: float = 0.0001,
    warmup_len: int = 300,
    perlin_scale: float = 0.02,
    perlin_octaves: int = 4,
) -> np.ndarray:
    """Build zero-mean kA disturbance added to predicted current.

    Returns:
        Array of shape (n + 1, 3), aligned with saved kA output.
    """
    noise = np.zeros((n + 1, 3), dtype=float)

    if not disturbed:
        return noise

    try:
        from noise import pnoise1
    except ImportError as exc:
        raise ImportError(
            "The 'noise' package is required for Perlin kA disturbance. "
            "Install with: pip install noise"
        ) from exc

    for i in range(3):
        noise[1:, i] = np.array([
            amp * pnoise1(
                (warmup_len + k) * perlin_scale + i * 100,
                octaves=perlin_octaves,
            )
            for k in range(n)
        ])

    return noise


def load_reference_csv(path: str | Path) -> np.ndarray:
    """Load an absolute resistance reference signal from CSV."""
    df = pd.read_csv(path)

    if all(c in df.columns for c in ["r1", "r2", "r3"]):
        return df[["r1", "r2", "r3"]].to_numpy(dtype=float)

    resistance_cols = ["El1_Resistance_mOhm", "El2_Resistance_mOhm", "El3_Resistance_mOhm"]
    if all(c in df.columns for c in resistance_cols):
        return df[resistance_cols].to_numpy(dtype=float)

    for col in ("r", "reference"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)

    raise ValueError(
        f"Reference CSV '{path}' must contain ['r1','r2','r3'], "
        "['El1_Resistance_mOhm','El2_Resistance_mOhm','El3_Resistance_mOhm'], "
        "or a scalar column 'r'/'reference'."
    )


def _as_3col(reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = np.repeat(reference.reshape(-1, 1), 3, axis=1)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError(f"Expected reference with shape (n,) or (n, 3), got {reference.shape}")
    return reference


def run_closed_loop_from_config(
    ref_csv: str | Path,
    controller_name: str,
    controller_config: str | Path,
    out_csv: str | Path,
    dt: float = 1.0,
    op_point: list[float] | None = None,
    use_current_model: bool = True,
    current_model_path: str | Path | None = None,
    current_hist_csv: str | Path | None = None,
    ka_disturbance: bool = True,
    plotting: bool = False,
) -> pd.DataFrame:
    """Run closed-loop simulation against the identified resistance model.

    If ``use_current_model`` is true, electrode current is predicted internally
    from the current model. Otherwise, a fallback external current trajectory is
    used.
    """
    resistance_bundle = load_arx_bundle(str(RESISTANCE_MODEL_PATH))
    resistance_state = load_initial_state(str(RESISTANCE_HIST_CSV), resistance_bundle)

    if op_point is None:
        op_point = [0.95, 1.03, 1.07]

    converter = ReferenceConverter.from_operating_point(
        op_point=op_point,
        window_s=1800,
    )
    reference_abs = load_reference_csv(ref_csv)
    reference_abs = _as_3col(reference_abs)
    reference_tilde = converter.convert_trajectory(reference_abs, freeze_trend=True)

    trend = converter.current_trend()
    print(f"  Trend at t=0:       {trend}")
    print(f"  Reference (abs)[0]: {reference_abs[0]}")
    print(f"  Reference (R~)[0]:  {reference_tilde[0]}")

    controllers = make_controllers(
        name=controller_name,
        config_path=str(controller_config),
        dt=dt,
    )

    if use_current_model:
        model_path = Path(current_model_path) if current_model_path is not None else CURRENT_MODEL_PATH
        hist_csv = Path(current_hist_csv) if current_hist_csv is not None else CURRENT_HIST_CSV

        print(f"  Current model: {model_path}")
        current_bundle = load_arx_bundle(str(model_path))
        current_state = load_initial_state(str(hist_csv), current_bundle)
        ka_noise = make_ka_noise(
            n=len(reference_tilde),
            disturbed=ka_disturbance,
            amp=0.16,
        )

        y, dpos, e, kA, r_abs = run_coupled_closed_loop(
            resistance_model=resistance_bundle,
            resistance_state=resistance_state,
            current_model=current_bundle,
            current_state=current_state,
            reference=reference_tilde,
            controllers=controllers,
            trend=trend,
            ka_noise=ka_noise,
        )
    else:
        n = len(reference_tilde)
        print(f"  kA disturbance: {'Perlin noise' if ka_disturbance else 'frozen at means'}")
        exog_traj = make_ka_exog(n, disturbed=ka_disturbance)

        y, dpos, e = run_closed_loop(
            model=resistance_bundle,
            state=resistance_state,
            reference=reference_tilde,
            controllers=controllers,
            exog_traj=exog_traj,
        )
        r_abs = trend[np.newaxis, :] + y
        kA = np.column_stack([
            np.r_[exog_traj["kA1"][0], exog_traj["kA1"]],
            np.r_[exog_traj["kA2"][0], exog_traj["kA2"]],
            np.r_[exog_traj["kA3"][0], exog_traj["kA3"]],
        ])

    dpos_cumsum = np.cumsum(np.vstack([np.zeros((1, 3)), dpos]), axis=0)

    out = pd.DataFrame({
        "t_s": np.arange(len(y), dtype=float) * dt,
        # Detrended resistance outputs
        "y1": y[:, 0],
        "y2": y[:, 1],
        "y3": y[:, 2],
        # Reconstructed absolute resistance outputs
        "R_abs1": r_abs[:, 0],
        "R_abs2": r_abs[:, 1],
        "R_abs3": r_abs[:, 2],
        # Detrended references
        "r1": np.r_[np.nan, reference_tilde[:, 0]],
        "r2": np.r_[np.nan, reference_tilde[:, 1]],
        "r3": np.r_[np.nan, reference_tilde[:, 2]],
        # Absolute references
        "R_ref_abs1": np.r_[np.nan, reference_abs[:, 0]],
        "R_ref_abs2": np.r_[np.nan, reference_abs[:, 1]],
        "R_ref_abs3": np.r_[np.nan, reference_abs[:, 2]],
        # Control inputs
        "dpos1": np.r_[0.0, dpos[:, 0]],
        "dpos2": np.r_[0.0, dpos[:, 1]],
        "dpos3": np.r_[0.0, dpos[:, 2]],
        "dpos1_cumsum": dpos_cumsum[:, 0],
        "dpos2_cumsum": dpos_cumsum[:, 1],
        "dpos3_cumsum": dpos_cumsum[:, 2],
        # Tracking errors
        "e1": np.r_[e[:, 0], np.nan],
        "e2": np.r_[e[:, 1], np.nan],
        "e3": np.r_[e[:, 2], np.nan],
        # Electrode currents
        "kA1": kA[:, 0],
        "kA2": kA[:, 1],
        "kA3": kA[:, 2],
    })

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    if plotting:
        from run_simulation.scripts.plotting import main as plot_main
        plot_main(path=out_csv)

    return out


if __name__ == "__main__":
    run_closed_loop_from_config(
        ref_csv="run_simulation/init_data/reference_res_2.csv",
        controller_name="pid",
        controller_config="run_simulation/init_data/PID_params.csv",
        out_csv="run_simulation/history/closed_loop_sim_varx.csv",
        dt=1.0,
        use_current_model=True,
        plotting=True,
        ka_disturbance=False
    )

"""
example_rule_controller.py
--------------------------
Simulates the real SAF plant R controller for all three electrodes.
This script demonstrates how to use the Plant and SaFSimulator classes.

Each electrode runs an independent rule-based controller:
  Target R = 1.2 mOhm
  Deadband +/- 0.07 mOhm
  Within deadband: hold position, reset step counter
  Outside deadband:
      Move electrode 1 cm, to shift R toward the target
      Wait 4 s before the next check
      After 10 consecutive steps without entering the deadband,
      switch to 20 s between steps (slow mode)
      Reset to normal mode when the deadband is re-entered
      Stop at position limits [0, 2]m even if still outside deadband

PF (power factor) is logged for monitoring but is not controlled.

Positive R error (R too high) -> increase electrode position (electrode goes
deeper, arc shortens, resistance drops).
Outputs
-------
  <this_folder>/results/rule_controller.csv
  <this_folder>/results/rule_controller.pdf

Usage
-----
  python example_rule_controller.py
  python example_rule_controller.py --r-nom 1.2 --steps 300
  python example_rule_controller.py --no-gp
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE         = Path(__file__).resolve().parent        # fusion/
_PROJECT_ROOT = _HERE.parent                           # SAF_data_to_control/
for _p in [str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Controller parameters matching the real plant
R_NOM          = 1.2    # target resistance (mOhm)
R_DEAD         = 0.07   # half-deadband (mOhm)
STEP_SIZE      = 0.01   # electrode step size (m)
WAIT_NORMAL    = 4      # seconds between steps in normal mode
WAIT_SLOW      = 20     # seconds between steps in slow mode
ESCALATE_AFTER = 10     # consecutive steps before switching to slow mode
U_MIN, U_MAX   = 0.0, 2.0

# Initial state of the simulator -- seed at the target R so simulation starts
# inside the deadband. The free-running ARX is not designed for large setpoint
# changes, so initialising at the target gives a much cleaner demonstration.
TYPICAL_POS  = 1.04
TYPICAL_R    = 1.006   # typical plant operating R; warm-up will settle from here
TYPICAL_KA   = 65.0
TYPICAL_REAC = 0.82
TYPICAL_V    = 165.0


class RuleController:
    """
    Per-electrode R deadband step controller.

    Move 1 cm when R leaves the deadband,
    wait before checking again, and slow down after many consecutive steps
    without success.
    """

    def __init__(
        self,
        initial_pos:    float,
        r_nom:          float = R_NOM,
        r_dead:         float = R_DEAD,
        step_size:      float = STEP_SIZE,
        wait_normal:    int   = WAIT_NORMAL,
        wait_slow:      int   = WAIT_SLOW,
        escalate_after: int   = ESCALATE_AFTER,
        pos_min:        float = U_MIN,
        pos_max:        float = U_MAX,
    ):
        self.pos            = float(initial_pos)
        self.r_nom          = r_nom
        self.r_dead         = r_dead
        self.step_size      = step_size
        self.wait_normal    = wait_normal
        self.wait_slow      = wait_slow
        self.escalate_after = escalate_after
        self.pos_min        = pos_min
        self.pos_max        = pos_max

        self.wait_left = 0     # steps remaining in current wait period
        self.consec    = 0     # steps taken without reaching deadband
        self.slow      = False # is slow mode active?
        self.stepped   = False # move made this call?

    def update(self, r: float) -> float:
        """
        Given the current measured R, return the new electrode position.
        Call once per simulation step.
        """
        self.stepped = False   # reset the "did it move?" flag each step

        # If we are still in the waiting after the last move, do nothing
        if self.wait_left > 0:
            self.wait_left -= 1
            return self.pos   # hold position

        # How far is R from the target?
        error = r - self.r_nom

        # If R is inside the deadband, reset and hold
        if abs(error) <= self.r_dead:
            self.consec = 0    # reset consecutive-step counter
            self.slow   = False
            return self.pos

        # R is outside the deadband: move the electrode one step toward the target.
        direction = float(np.sign(error))
        new_pos   = float(np.clip(
            self.pos + direction * self.step_size,
            self.pos_min, self.pos_max,
        ))

        if new_pos != self.pos:
            self.pos     = new_pos
            self.consec += 1   # count another step outside deadband
            self.stepped = True

        # Switch to slow mode after too many consecutive steps without improvement
        self.slow      = self.consec >= self.escalate_after
        # Start the waiting period before the next move
        self.wait_left = self.wait_slow if self.slow else self.wait_normal
        return self.pos


def _pf(r: float, rx: float) -> float:
    """Power factor: cos(phi) = R / sqrt(R^2 + X^2)."""
    denom = np.sqrt(max(r ** 2 + rx ** 2, 1e-12))
    return float(np.clip(r / denom, 0.0, 1.0))


def run_rule_controller(
    r_nom:   float = R_NOM,
    r_dead:  float = R_DEAD,
    n:       int   = 300,
    use_gp:  bool  = True,
    out_csv: "str | Path" = _HERE / "results" / "rule_controller.csv",
    out_pdf: "str | Path" = _HERE / "results" / "rule_controller.pdf",
) -> pd.DataFrame:

    import joblib
    _pkg = _HERE.name
    _sim_mod = __import__(f"{_pkg}.simulators.saf_simulator",
                          fromlist=["SaFSimulator", "build_init_row_from_scalars"])
    SaFSimulator             = _sim_mod.SaFSimulator
    build_init_row_from_scalars = _sim_mod.build_init_row_from_scalars
    _gp_mod  = __import__(f"{_pkg}.training.gp_loader", fromlist=["load_gp_bundle"])
    load_gp_bundle = _gp_mod.load_gp_bundle

    models_dir = _HERE / "models"
    arx = joblib.load(models_dir / "arx_joint_txt2026.joblib")

    init_row = build_init_row_from_scalars(
        pos=TYPICAL_POS, r=TYPICAL_R, ka=TYPICAL_KA,
        rx=TYPICAL_REAC, v=TYPICAL_V, arx_bundle=arx,
    )

    # Load per-electrode GP bundles (use ARX-only if files are missing)
    # Change "txt2026_512" to "pi_512" here to use the PI-trained GP variant.
    gp_variant = "txt2026_512"
    gp_bundles: dict = {}
    if use_gp:
        for i in (1, 2, 3):
            gp_path = models_dir / f"gp_el{i}_{gp_variant}.pt"
            if gp_path.exists():
                gp_bundles[i] = load_gp_bundle(gp_path)
            else:
                print(f"[rule_ctrl] El{i}: GP model not found at {gp_path}, using ARX only")

    sim = SaFSimulator(arx, init_row, electrode=1)

    gp_tag = "ARX+GP" if gp_bundles else "ARX only"
    print(f"[rule_ctrl] {n} steps  R_nom={r_nom} +/-{r_dead} mOhm  {gp_tag}")

    # For multi-electrode simulation we need direct access to all electrode predictions.
    # Use a helper that holds the simulator and GP bundles together.
    _plant_cache = {}

    def _current_r(electrode: int) -> float:
        """Get current GP-corrected R for any electrode."""
        _pkg = _HERE.name
        predict_single = __import__(f"{_pkg}.training.gp_loader",
                                    fromlist=["predict_single"]).predict_single
        sim._electrode = electrode
        y_arx = sim._predict_r(electrode)
        bun = gp_bundles.get(electrode)
        if bun is None:
            return y_arx
        feats = sim.get_gp_features_electrode(electrode)
        feats["step_in_window"] = float(min(_plant_cache.get("step", 0), 19))
        feats["y_sim"]          = y_arx
        feats["y_sim_sq"]       = y_arx * y_arx
        feats["y_real_lag1"]    = _plant_cache.get(f"r{electrode}", y_arx)
        feats["y_real_lag2"]    = _plant_cache.get(f"r{electrode}_lag2", y_arx)
        x = np.array([feats.get(f, 0.0) for f in bun["feature_names"]], dtype=np.float32)
        mu, _ = predict_single(bun, x)
        mu = float(np.clip(mu, -0.15, 0.15))
        return float(y_arx + mu)

    # One independent controller per electrode
    controllers = {
        i: RuleController(initial_pos=TYPICAL_POS, r_nom=r_nom, r_dead=r_dead)
        for i in (1, 2, 3)
    }

    # Set primary electrode to 1
    sim._electrode = 1

    # Warm up the ARX for 50 steps with fixed positions so the simulation
    # starts from the model's natural equilibrium rather than from the seeded
    # lag registers. Without this the first prediction jumps away from the
    # seed value and the initial R can be 5-15% below the seeded target.
    _WARMUP = 50
    for _ in range(_WARMUP):
        _ya = {i: sim._predict_r(i) for i in (1, 2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec={i: TYPICAL_POS for i in (1, 2, 3)},
                              y_new_vec=_ya)
    sim._electrode = 1

    # Seed the GP lag cache from the settled state so the correction is
    # calibrated from the very first step
    for _i in (1, 2, 3):
        _r0 = sim._predict_r(_i)
        _plant_cache[f"r{_i}"]      = _r0
        _plant_cache[f"r{_i}_lag2"] = _r0

    records = []
    for k in range(n):
        _plant_cache["step"] = k

        # Get current R for all electrodes
        current_r = {}
        for i in (1, 2, 3):
            sim._electrode = i
            current_r[i] = _current_r(i)

        sim._electrode = 1

        if k % 60 == 0:
            r_str    = "  ".join(f"R{i}={current_r[i]:.4f}" for i in (1, 2, 3))
            u_str    = "  ".join(f"u{i}={controllers[i].pos:.3f}" for i in (1, 2, 3))
            mode_str = "  ".join(
                f"El{i}:{'SLOW' if controllers[i].slow else 'norm'}"
                f"(c={controllers[i].consec})"
                for i in (1, 2, 3)
            )
            print(f"  step {k:4d}  {r_str}  {u_str}  {mode_str}", flush=True)

        u_new = {i: controllers[i].update(current_r[i]) for i in (1, 2, 3)}

        rec = {"t": k}
        for i in (1, 2, 3):
            r_i  = current_r[i]
            rx_i = float(sim._row.get(f"El{i}_CalcReac_filt_lag1", TYPICAL_REAC))
            rec[f"R{i}"]       = r_i
            rec[f"u{i}"]       = u_new[i]
            rec[f"pf{i}"]      = _pf(r_i, rx_i)
            rec[f"in_db{i}"]   = int(abs(r_i - r_nom) <= r_dead)
            rec[f"stepped{i}"] = int(controllers[i].stepped)
            rec[f"consec{i}"]  = controllers[i].consec
            rec[f"slow{i}"]    = int(controllers[i].slow)
            _plant_cache[f"r{i}_lag2"] = _plant_cache.get(f"r{i}", r_i)
            _plant_cache[f"r{i}"]      = r_i
        records.append(rec)

        y_arx_vec = {i: sim._predict_r(i) for i in (1, 2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.advance_multi(u_new_vec=u_new, y_new_vec=y_arx_vec)

    df = pd.DataFrame(records)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[rule_ctrl] CSV saved -> {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        out_pdf = Path(out_pdf)
        colors  = ["C0", "C1", "C2"]
        t       = df["t"].values

        fig, (ax_r, ax_u) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Top: resistance for all three electrodes
        ax_r.fill_between(t, r_nom - r_dead, r_nom + r_dead,
                          color="green", alpha=0.15, label="deadband")
        ax_r.axhline(r_nom, color="k", ls="--", lw=0.8, label=f"target {r_nom}")
        for idx, i in enumerate((1, 2, 3)):
            ax_r.plot(t, df[f"R{i}"].values, color=colors[idx], lw=1.0,
                      label=f"El{i}")
        ax_r.set_ylabel("Resistance (mOhm)")
        ax_r.legend(loc="upper right", fontsize=8)
        ax_r.grid(True, alpha=0.2)
        ax_r.set_title(f"Rule-based R controller - {gp_tag}  "
                       f"(target={r_nom} +/-{r_dead} mOhm)")

        # Bottom: electrode positions
        for idx, i in enumerate((1, 2, 3)):
            ax_u.plot(t, df[f"u{i}"].values, color=colors[idx], lw=1.0,
                      label=f"El{i}")
        ax_u.set_ylabel("Position (m)")
        ax_u.set_xlabel("Time (s)")
        ax_u.legend(loc="upper right", fontsize=8)
        ax_u.grid(True, alpha=0.2)

        fig.tight_layout()
        with PdfPages(str(out_pdf)) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"[rule_ctrl] PDF saved -> {out_pdf}")

    except Exception as e:
        print(f"WARNING: plot failed ({e})")

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rule-based SAF R controller simulation")
    ap.add_argument("--r-nom",  type=float, default=R_NOM,
                    help=f"Target R (mOhm, default {R_NOM})")
    ap.add_argument("--r-dead", type=float, default=R_DEAD,
                    help=f"Deadband half-width (mOhm, default {R_DEAD})")
    ap.add_argument("--steps",  type=int,   default=300,
                    help="Number of simulation steps (default 300)")
    ap.add_argument("--no-gp",  action="store_true",
                    help="Run on ARX only, no GP correction")
    ap.add_argument("--out",    default=str(_HERE / "results" / "rule_controller.csv"))
    ap.add_argument("--pdf",    default=str(_HERE / "results" / "rule_controller.pdf"))
    args = ap.parse_args()

    run_rule_controller(
        r_nom   = args.r_nom,
        r_dead  = args.r_dead,
        n       = args.steps,
        use_gp  = not args.no_gp,
        out_csv = args.out,
        out_pdf = args.pdf,
    )

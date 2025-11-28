from joblib import load
import numpy as np
import pandas as pd

MODEL_PATH = "models/arx_linear_ridge_stable_yonly.joblib"  # adjust if needed

def main():
    b = load(MODEL_PATH)

    exog_model = b["exog_model"]
    exog_cols  = b["exog_cols"]
    scalers    = b["scalers"]

    y_scaler      = scalers["y_scaler"]
    X_scaler_exog = scalers["X_scaler_exog"]

    if exog_model is None:
        print("No exogenous model in bundle.")
        return

    beta = np.asarray(exog_model.coef_, float).ravel()  # β_j in z-space

    # std devs in physical space
    sigma_y = float(y_scaler.scale_[0])
    sigma_x = np.asarray(X_scaler_exog.scale_, float)   # per feature

    # convert to approximate physical-unit gains: dy/dx_j
    # γ_j ≈ β_j * σ_y / σ_x_j
    gamma = beta * sigma_y / sigma_x

    df = pd.DataFrame({
        "feature": exog_cols,
        "beta_z": beta,
        "sigma_x": sigma_x,
        "gamma_phys": gamma,   # approx: mΩ per (unit of feature)
    }).set_index("feature")

    # Focus on electrode positions
    el_features = {}
    for el in (1, 2, 3):
        prefix = f"El{el}_pos_m_filt_lag"
        mask = df.index.str.startswith(prefix)
        sub = df[mask].copy()
        el_features[el] = sub

    # Print per-electrode summary
    for el, sub in el_features.items():
        if sub.empty:
            print(f"\nEl{el}: no position features found in exog_cols.")
            continue

        print(f"\n=== El{el} position influence ===")
        print(sub[["beta_z", "gamma_phys"]].sort_values("gamma_phys", key=np.abs, ascending=False))

        # some aggregate measures
        l1 = sub["gamma_phys"].abs().sum()
        l2 = np.sqrt((sub["gamma_phys"]**2).sum())
        print(f"  L1 norm of gamma (sum |γ|): {l1:.4g}")
        print(f"  L2 norm of gamma (sqrt(sum γ^2)): {l2:.4g}")

    # Compare El1 vs El2 vs El3 by L2 norm
    norms = {}
    for el, sub in el_features.items():
        if not sub.empty:
            norms[el] = np.sqrt((sub["gamma_phys"]**2).sum())

    if norms:
        print("\n=== Relative dynamic 'strength' (per electrode) ===")
        for el, val in sorted(norms.items(), key=lambda kv: kv[1], reverse=True):
            print(f"El{el}: L2 norm of γ over lags = {val:.4g}")
    else:
        print("\nNo electrode position features found at all in exog_cols.")


if __name__ == "__main__":
    main()

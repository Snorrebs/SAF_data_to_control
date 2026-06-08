"""
fusion/mpc/arx_casadi.py
Converts ARX bundles into CasADi symbolic expressions for use inside
the do-mpc NLP so IPOPT can differentiate through the prediction horizon.
"""
from __future__ import annotations

import numpy as np


def arx_expr(col_vals: "dict[str, ca.MX]", bundle: dict) -> "ca.MX":
    """Build a CasADi MX expression for one ARX prediction.

    col_vals : maps X_col name to a CasADi MX scalar (state or TVP symbol).
    bundle   : joblib ARX bundle with X_cols, model, X_scaler, y_scaler.

    Returns predicted output in physical units.
    """
    import casadi as ca

    coef      = bundle["model"].coef_.astype(np.float64)
    intercept = float(bundle["model"].intercept_)
    x_mean    = bundle["X_scaler"].mean_.astype(np.float64)
    x_std     = bundle["X_scaler"].scale_.astype(np.float64)
    y_mean    = float(bundle["y_scaler"].mean_[0])
    y_std     = float(bundle["y_scaler"].scale_[0])

    x_raw = ca.vertcat(*[col_vals[c] for c in bundle["X_cols"]])
    x_sc  = (x_raw - ca.DM(x_mean)) / ca.DM(x_std)
    y_sc  = ca.DM(coef).T @ x_sc + intercept
    return ca.reshape(y_sc * y_std + y_mean, 1, 1)


def joint_output_expr(
    col_vals:   "dict[str, ca.MX]",
    bundle:     dict,
    output_idx: int,
) -> "ca.MX":
    """CasADi expression for one output of a joint multi-output ARX (V9)."""
    import casadi as ca

    model  = bundle["model"]
    x_sc   = bundle["X_scaler"]
    y_sc   = bundle["Y_scaler"]

    # ReducedRankRidge stores coef_ as (n_features, n_outputs) so that
    # prediction is X @ coef_.  We extract column output_idx, not a row.
    coef_mat   = np.atleast_2d(model.coef_)
    intercepts = np.atleast_1d(model.intercept_)
    coef_row   = coef_mat[:, output_idx].astype(np.float64)
    intercept  = float(intercepts[output_idx])

    x_mean = x_sc.mean_.astype(np.float64)
    x_std  = x_sc.scale_.astype(np.float64)
    y_mean = float(np.atleast_1d(y_sc.mean_)[output_idx])
    y_std  = float(np.atleast_1d(y_sc.scale_)[output_idx])

    x_raw = ca.vertcat(*[col_vals.get(c, ca.DM(0.0)) for c in bundle["X_cols"]])
    x_sc_sym = (x_raw - ca.DM(x_mean)) / ca.DM(x_std)
    y_norm   = ca.DM(coef_row).T @ x_sc_sym + intercept
    return ca.reshape(y_norm * y_std + y_mean, 1, 1)


def delta_arx_r_expr(
    col_vals: "dict[str, ca.MX]",
    bundle:   dict,
    el:       int,
) -> "ca.MX":
    """CasADi R prediction for one electrode of a DeltaARXWrapper (V15).

    Returns R(t+1) = R_lag1 + inner_Ridge.predict(X_el_scaled).
    """
    import casadi as ca

    wrapper   = bundle["model"]
    inner     = wrapper.models_[el]
    x_sc      = wrapper.x_scalers_[el]
    xcols     = wrapper.xcols_per_el_[el]
    r_lag1_col = wrapper.r_lag1_name_[el]

    coef      = inner.coef_.astype(np.float64)
    intercept = float(inner.intercept_)
    x_mean    = x_sc.mean_.astype(np.float64)
    x_std     = x_sc.scale_.astype(np.float64)

    x_raw    = ca.vertcat(*[col_vals.get(c, ca.DM(0.0)) for c in xcols])
    x_sc_sym = (x_raw - ca.DM(x_mean)) / ca.DM(x_std)
    dR       = ca.DM(coef).T @ x_sc_sym + intercept
    R_prev   = col_vals.get(r_lag1_col, ca.DM(1.076))
    return ca.reshape(R_prev + dR, 1, 1)


def clip_bounds(bundle: dict, lo_floor: "float | None" = None) -> "tuple[float, float]":
    """Return (lo, hi) clip bounds from y_scaler mean +/- 6 sigma."""
    ys    = bundle.get("y_scaler") or bundle.get("Y_scaler")
    mu    = float(np.atleast_1d(ys.mean_)[0])
    sigma = float(np.atleast_1d(ys.scale_)[0])
    lo    = mu - 6.0 * sigma
    if lo_floor is not None:
        lo = max(lo, lo_floor)
    return (lo, mu + 6.0 * sigma)


def ca_clip(expr: "ca.MX", lo: float, hi: float) -> "ca.MX":
    """Element-wise clip in CasADi."""
    import casadi as ca
    return ca.fmax(ca.fmin(expr, hi), lo)

#!/usr/bin/env python3
# stable_arx_runner_yonly.py
#
# Closed-loop simulator for "AR-on-y-only" bundle.

import numpy as np
from joblib import load

class StableARXRunnerYOnly:
    def __init__(self, bundle, y_init):
        """
        Args:
            bundle: dict from arx_fit_stable_yonly.py
            y_init: initial y values (physical units), length >= ar_order
        """
        self.bundle = bundle

        self.a = np.asarray(bundle["ar_coeffs"], float).ravel()
        self.p = int(bundle["ar_order"])

        self.exog_model = bundle["exog_model"]
        self.exog_cols  = bundle["exog_cols"]

        scalers = bundle["scalers"]
        self.y_scaler      = scalers["y_scaler"]
        self.X_scaler_exog = scalers["X_scaler_exog"]

        y_init = np.asarray(y_init, float).ravel()
        if self.p > 0:
            assert y_init.size >= self.p, (
                f"Need at least {self.p} warmup samples, got {y_init.size}"
            )

        # store y history in z-space
        y_init_z = self.y_scaler.transform(y_init.reshape(-1, 1)).ravel()
        self.y_buffer_z = list(y_init_z)

    def advance(self, x_exog_t, clip_z=10.0):
        """
        Advance one step using exogenous inputs x_exog_t (same order as exog_cols).

        Returns:
            y_pred (float) in physical units.
        """
        # AR part
        if self.p > 0:
            last = np.array(self.y_buffer_z[-self.p:])  # [ ..., y_{t-2}, y_{t-1} ] in z
            last_rev = last[::-1]                       # [ y_{t-1}, ..., y_{t-p} ]
            y_ar = float(self.a @ last_rev)
        else:
            y_ar = 0.0

        # Exogenous part
        if self.exog_model is not None and self.exog_cols:
            x_exog_t = np.asarray(x_exog_t, float).reshape(1, -1)
            x_exog_scaled = self.X_scaler_exog.transform(x_exog_t)
            r_hat = float(self.exog_model.predict(x_exog_scaled))
        else:
            r_hat = 0.0

        y_z_new = y_ar + r_hat

        # small safety clip (optional)
        if not np.isfinite(y_z_new):
            y_z_new = 0.0
        else:
            y_z_new = float(np.clip(y_z_new, -clip_z, clip_z))

        y_new = float(self.y_scaler.inverse_transform([[y_z_new]])[0, 0])

        self.y_buffer_z.append(y_z_new)
        return y_new

    @classmethod
    def from_model_path(cls, model_path, y_init):
        bundle = load(model_path)
        return cls(bundle, y_init=y_init)

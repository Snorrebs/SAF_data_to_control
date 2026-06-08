"""
fusion/training/delta_arx.py
Importable home for DeltaARXWrapper so joblib can unpickle it regardless
of how train_gp_v14.py was invoked.
"""
from __future__ import annotations

import numpy as np


class DeltaARXWrapper:
    """Wraps per-electrode Ridge dR models behind the joint ARX interface.

    predict(X) returns shape (n, 10) matching _Y_COLS_JOINT.
      R columns [0,1,2]: R_lag1 + dR_pred (physical units)
      kA, reac, V:       held at their lag-1 values so advance_multi does
                         not corrupt those registers with zeros.

    R(t+1) = R_lag1(t) + dR_pred where dR_pred = Ridge.predict(X).
    """

    def __init__(
        self,
        models,
        x_scalers,
        xcols_per_el,
        r_lag1_name,
        y_cols,
        y_index,
    ) -> None:
        self.models_       = models
        self.x_scalers_    = x_scalers
        self.xcols_per_el_ = xcols_per_el
        self.r_lag1_name_  = r_lag1_name
        self.y_cols        = y_cols
        self.y_index       = y_index
        self.coef_         = np.zeros((1, len(y_cols)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X shape: (n, n_features_joint). Returns (n, 10)."""
        n   = X.shape[0]
        out = np.zeros((n, len(self.y_cols)), dtype=np.float64)

        for el in (1, 2, 3):
            x_sc   = self.x_scalers_[el]
            model  = self.models_[el]
            X_el   = X[:, self._el_indices[el]]
            Xz     = x_sc.transform(X_el)
            dR     = model.predict(Xz).ravel()
            R_prev = X[:, self._r_lag1_joint_idx[el]].ravel()
            out[:, self.y_index["R"][el]] = R_prev + dR

            # Hold kA and reac at their lag-1 values. If these returned zero,
            # advance_multi would corrupt the kA/reac lag registers and put the
            # ARX into a completely out-of-distribution state.
            ka_idx   = self._ka_lag1_joint_idx.get(el)
            reac_idx = self._reac_lag1_joint_idx.get(el)
            if ka_idx is not None:
                out[:, self.y_index["kA"][el]]   = X[:, ka_idx].ravel()
            if reac_idx is not None:
                out[:, self.y_index["reac"][el]] = X[:, reac_idx].ravel()

        # Hold transformer voltage at its lag-1 value
        v_idx = self._v_lag1_joint_idx
        if v_idx is not None:
            out[:, self.y_index["v"]] = X[:, v_idx].ravel()

        return out

    @classmethod
    def build(cls, models, x_scalers, xcols_per_el,
              joint_xcols, y_cols, y_index) -> "DeltaARXWrapper":
        wrapper = cls(models, x_scalers, xcols_per_el,
                      {el: f"El{el}_y_filt_lag1" for el in (1, 2, 3)},
                      y_cols, y_index)

        joint_idx = {c: i for i, c in enumerate(joint_xcols)}
        wrapper._el_indices          = {}
        wrapper._r_lag1_joint_idx    = {}
        wrapper._ka_lag1_joint_idx   = {}
        wrapper._reac_lag1_joint_idx = {}

        for el in (1, 2, 3):
            wrapper._el_indices[el] = np.array(
                [joint_idx[c] for c in xcols_per_el[el]], dtype=int)
            wrapper._r_lag1_joint_idx[el]    = joint_idx[f"El{el}_y_filt_lag1"]
            wrapper._ka_lag1_joint_idx[el]   = joint_idx.get(f"El{el}_kA_filt_lag1")
            wrapper._reac_lag1_joint_idx[el] = joint_idx.get(f"El{el}_CalcReac_filt_lag1")

        wrapper._v_lag1_joint_idx = joint_idx.get("RMS_V_transformer_filt_lag1")
        return wrapper

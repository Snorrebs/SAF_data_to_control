# forecaster_meta_adapter.py
import joblib, numpy as np, pandas as pd

class MetaPLSWithTerms:
    def __init__(self, artifact_path, build_x_from_terms):
        art = joblib.load(artifact_path)
        self.pls = art["pls"]
        self.x_terms = list(art["x_terms"])
        self.build_x_from_terms = build_x_from_terms
        self._n_in = getattr(self.pls, "n_features_in_", None)

    def predict(self, raw_df: pd.DataFrame) -> pd.Series:
        X_df = self.build_x_from_terms(raw_df, self.x_terms)
        for t in self.x_terms:
            if t not in X_df.columns:
                X_df[t] = 0.0
        X_df = X_df[self.x_terms]
        if self._n_in is not None and X_df.shape[1] != self._n_in:
            raise ValueError(f"Metamodel expected {self._n_in}, got {X_df.shape[1]}")
        y_hat = self.pls.predict(X_df.values).reshape(-1)
        return pd.Series(y_hat, index=raw_df.index, name="Tot_Resistance_meta")

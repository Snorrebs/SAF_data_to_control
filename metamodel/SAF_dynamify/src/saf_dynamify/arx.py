import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import joblib

@dataclass
class ARXConfig:
    lags: int = 10
    ridge: float = 1e-4

class ARX:
    def __init__(self, cfg: ARXConfig):
        self.cfg = cfg
        self.coef_: np.ndarray | None = None  # [p, ny]
        self.intercept_: np.ndarray | None = None  # [ny]
        self.columns_: list[str] | None = None
        self.y_names_: list[str] | None = None

    def _design(self, X_lagged: pd.DataFrame) -> np.ndarray:
        Phi = X_lagged.to_numpy(dtype=float)
        Phi = np.nan_to_num(Phi)
        return Phi

    def fit(self, X_lagged: pd.DataFrame, Y: pd.DataFrame):
        self.columns_ = list(X_lagged.columns)
        self.y_names_ = list(Y.columns)
        Phi = self._design(X_lagged)
        Ymat = Y.to_numpy(dtype=float)
        # Ridge closed form: (Phi^T Phi + λI)^{-1} Phi^T Y
        XtX = Phi.T @ Phi
        lamI = self.cfg.ridge * np.eye(XtX.shape[0])
        XtY = Phi.T @ Ymat
        beta = np.linalg.solve(XtX + lamI, XtY)
        self.coef_ = beta
        # Intercept (mean removed if standardized; keep zero)
        self.intercept_ = np.zeros(Ymat.shape[1])
        return self

    def predict(self, X_lagged: pd.DataFrame) -> pd.DataFrame:
        Phi = self._design(X_lagged)
        Yhat = Phi @ self.coef_ + self.intercept_
        return pd.DataFrame(Yhat, index=X_lagged.index, columns=self.y_names_)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "cfg": self.cfg, "coef_": self.coef_, "intercept_": self.intercept_,
            "columns_": self.columns_, "y_names_": self.y_names_
        }, path)

    @staticmethod
    def load(path: Path) -> "ARX":
        blob = joblib.load(path)
        model = ARX(blob["cfg"])
        model.coef_ = blob["coef_"]
        model.intercept_ = blob["intercept_"]
        model.columns_ = blob["columns_"]
        model.y_names_ = blob["y_names_"]
        return model
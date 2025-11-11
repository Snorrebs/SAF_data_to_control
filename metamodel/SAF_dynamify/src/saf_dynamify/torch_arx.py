import pandas as pd
import torch
from dataclasses import dataclass

@dataclass
class TorchARXConfig:
    lags: int = 10
    ridge: float = 1e-4
    device: str = "cpu"

class TorchARX:
    def __init__(self, cfg: TorchARXConfig):
        self.cfg = cfg
        self.W = None  # [p, ny]
        self.y_names = None
        self.columns = None

    def fit(self, X_lagged: pd.DataFrame, Y: pd.DataFrame):
        self.columns = list(X_lagged.columns)
        self.y_names = list(Y.columns)
        X = torch.tensor(X_lagged.fillna(0.0).to_numpy(), dtype=torch.float32, device=self.cfg.device)
        Yt = torch.tensor(Y.to_numpy(), dtype=torch.float32, device=self.cfg.device)
        XtX = X.T @ X
        p = XtX.shape[0]
        lamI = self.cfg.ridge * torch.eye(p, device=self.cfg.device)
        XtY = X.T @ Yt
        # robust solve
        try:
            W = torch.linalg.solve(XtX + lamI, XtY)
        except RuntimeError:
            W = torch.linalg.lstsq(XtX + lamI, XtY).solution
        self.W = W
        return self

    def predict(self, X_lagged: pd.DataFrame) -> pd.DataFrame:
        X = torch.tensor(X_lagged.fillna(0.0).to_numpy(), dtype=torch.float32)
        Yhat = X @ self.W.cpu()
        return pd.DataFrame(Yhat.numpy(), index=X_lagged.index, columns=self.y_names)
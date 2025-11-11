from __future__ import annotations
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def metrics(y_true, y_pred):
    """Compute R^2 and RMSE between y_true and y_pred."""
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": r2, "rmse": rmse}
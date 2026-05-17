from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
from scipy.linalg import block_diag
import numpy.typing as npt
import pandas as pd
import joblib
import pysindy as ps
@dataclass(frozen=True)
class GeneralizedParams:
    model: ps.SINDy
    
def load_generalized_params_csv(path: str | Path) -> list[GeneralizedParams]:
    """ Loads the identified controller object        
    """
    model = joblib.load(path)
    
    return GeneralizedParams(model = model)


 
class GeneralizedController:

    def __init__(self, params: GeneralizedParams, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be > 0")
 
        self.params = params
        self.dt = float(dt)
 
        self._i_term: npt.ndarray = np.array([0.0, 0.0, 0.0])
        self._prev_y_pred: list | None = None
 
    def reset(self) -> None:
        self._i_term = np.array([0.0, 0.0, 0.0])
        self._prev_y_pred = None

    def step(self, reference: npt.ndarray, y_pred: dict, u_prev: npt.ndarray) -> tuple[list, list]:
        e1 = float(reference[0]) - float(y_pred[1])
        e2 = float(reference[1]) - float(y_pred[2])
        e3 = float(reference[2]) - float(y_pred[3])
        e = np.array([e1,e2,e3])
        # Derivative on output (not error) to avoid derivative kick

        if self._prev_y_pred is None:
            d1_term, d2_term, d3_term = 0.0, 0.0, 0.0
        else:
            d1_term, d2_term, d3_term = (float(y_pred[1] - self._prev_y_pred[1]) / self.dt,
                                         float(y_pred[2] - self._prev_y_pred[2]) / self.dt,
                                         float(y_pred[3] - self._prev_y_pred[3]) / self.dt)

        self._i_term += e * self.dt # integrators are unused for now.
        self._prev_y_pred = y_pred
        #K = self.params.coeffs
        states = np.array([[e1,d1_term,
                           e2,d2_term,
                           e3,d3_term]])
        
        model = self.params.model
        u_des = model.predict(states) + np.atleast_2d(u_prev).T

        return (u_des[0]).tolist(), e.tolist()


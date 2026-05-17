from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import numpy.typing as npt
from scipy.linalg import block_diag
@dataclass(frozen=True)
class PIDParams:
    coeffs: npt.ndarray
 
 
def load_pid_params_csv(path: str | Path) -> PIDParams:
    """Load PID parameters from CSV.
       The CSV file must contain 3 rows and 3 columns.
 
    Returns:
        1 PIDparams instance.
    """
    df = pd.read_csv(path)
    if df.shape[0] == 1:
        print("found 1 row in PID params csv. Broadcasting coefficients is no longer supported. Specify controller coefficients for each electrode explicitly.")
    if df.shape[0] != 3:
        raise ValueError("PID params csv must have 3 rows. Found: ",df.shape[0])

            
    df.columns = [c.strip().lower() for c in df.columns]

    arr = df.to_numpy()
    lin1 = arr[0,:]
    lin2 = arr[1,:]
    lin3 = arr[2,:]
    coeffs = block_diag(lin1,lin2,lin3)
    
    return PIDParams(coeffs = coeffs)

 
class PIDController:
    """Velocity-form PID controller.
 
    - Derivative acts on the measured output (not the error) to avoid
      derivative kick on reference steps.
    - Velocity form: ``u = u_prev + Kp*e + Ki*integral - Kd*d_output``
 
    Actuator rate and position limits are applied externally by
    ``closed_loop_sim.apply_actuator_limits``, not here.
    """
 
    def __init__(self, params: PIDParams, dt: float) -> None:
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

        self._i_term += e * self.dt
        self._prev_y_pred = y_pred
        K = self.params.coeffs
        states = np.array([e1,float(self._i_term[0]),d1_term,
                           e2,float(self._i_term[1]),d2_term,
                           e3,float(self._i_term[2]),d3_term])
        u_des = K@ np.atleast_2d(states).T + np.atleast_2d(u_prev).T
 
        return u_des.tolist(), e.tolist()

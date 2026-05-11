from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
import pandas as pd
 
 
@dataclass(frozen=True)
class GeneralizedParams:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
 
 
def load_generalized_params_csv(path: str | Path) -> list[GeneralizedParams]:
    # Placeholder
    return None
 
 
class GeneralizedController:
    # Placeholder
    def __init__(self, params: GeneralizedParams, dt: float) -> None:

        return None
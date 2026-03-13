from __future__ import annotations

from run_simulation.closed_loop.controllers.pid import (
    PIDController,
    load_pid_params_csv,
)
from run_simulation.closed_loop.controllers.open_loop import (
    OpenLoopController,
    load_open_loop_params_csv,
)


def make_controller(name: str, config_path: str, dt: float):
    key = name.strip().lower()

    if key == "pid":
        params = load_pid_params_csv(config_path)
        return PIDController(params=params, dt=dt)

    if key == "open_loop":
        params = load_open_loop_params_csv(config_path)
        return OpenLoopController(params=params, dt=dt)

    raise ValueError(f"Unknown controller: {name}")

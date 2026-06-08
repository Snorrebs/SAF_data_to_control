from __future__ import annotations

from run_simulation.closed_loop.controllers.mpc import LinearMPC, load_mpc_params_csv
from run_simulation.closed_loop.controllers.open_loop import (
    OpenLoopController,
    load_open_loop_params_csv,
)
from run_simulation.closed_loop.controllers.pid import (
    PIDController,
    load_pid_params_csv,
)
from run_simulation.closed_loop.controllers.generalized_controller import (
    GeneralizedController,
    load_generalized_params_csv,
)
from run_simulation.closed_loop.controllers.pid_fullspace import (
    PidFullspaceController,
    load_pidfullspace_params_csv,
)


def make_mpc_controller(config_path: str, bundle: dict) -> LinearMPC:
    """Instantiate a LinearMPC from a params CSV and a loaded model bundle."""
    params = load_mpc_params_csv(config_path)
    return LinearMPC(params=params, bundle=bundle)


def make_controllers(name: str, config_path: str, dt: float) -> list:
    """Instantiate a list of 3 controllers (one per electrode).
 
    Args:
        name:        controller type, e.g. ``"pid"`` or ``"open_loop"``
        config_path: path to the parameter CSV (1 row = broadcast, 3 rows = per-electrode)
        dt:          simulation timestep [s]
 
    Returns:
        A list of 3 controller instances satisfying the ``Controller`` protocol,
        OR 1 instance of a unified controller.
    """
    key = name.strip().lower()
 
    if key == "pid":
        params_list = load_pid_params_csv(config_path)
        return PIDController(params=params_list, dt=dt)
 
    if key == "open_loop":
        params_list = load_open_loop_params_csv(config_path)
        return OpenLoopController(params=params_list, dt=dt)
   
    if key == "generalized_controller":
        params_list = load_generalized_params_csv(config_path)
        return GeneralizedController(params=params_list, dt=dt)
    
    if key == "pid_fullspace":
        params_list = load_pidfullspace_params_csv(config_path)
        return PidFullspaceController(params=params_list, dt=dt)

    raise ValueError(f"Unknown controller type: '{name}'. Valid options: 'pid', 'open_loop'")

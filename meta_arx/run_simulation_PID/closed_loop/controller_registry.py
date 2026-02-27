# run_simulation_PID/closed_loop/controller_registry.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict

from run_simulation_PID.closed_loop.controller_api import Controller

# Import your controller implementations here
from run_simulation_PID.closed_loop.controllers.pid import PIDController, PIDParams


# Registry maps a short name -> constructor function
_REGISTRY: Dict[str, Callable[..., Controller]] = {
    "pid": lambda **kw: PIDController(**kw),
    # Add more controllers here, e.g.:
    # "open_loop": lambda **kw: OpenLoopController(**kw),
}


def available_controllers() -> list[str]:
    """Return list of registered controller names."""
    return sorted(_REGISTRY.keys())


def make_controller(name: str, **kwargs: Any) -> Controller:
    """
    Factory: create controller by name.

    Example:
        ctrl = make_controller(
            "pid",
            params=PIDParams(kp=..., ki=..., kd=..., u_min=..., u_max=...),
            dt=1.0,
        )
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown controller '{name}'. Available: {', '.join(available_controllers())}"
        )


    return _REGISTRY[key](**kwargs)

from typing import Protocol


class Controller(Protocol):

    def reset(self) -> None:
        ...

    def step(self, reference: float, y_pred: float, u_prev: float) -> tuple[float, float]:
        """
        Returns:
            u_des : desired actuator position
            e     : control error
        """
        ...

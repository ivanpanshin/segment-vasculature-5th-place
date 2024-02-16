from abc import ABC, abstractmethod
from typing import Any, Dict, Set

import numpy as np

DIRECTIONS: Set[str] = {"maximize", "minimize"}


class BaseMetric(ABC):
    def __init__(
        self,
        main_metric: str,
        main_metric_direction: str,
        track_val_loss: bool,
    ):
        if main_metric_direction not in DIRECTIONS:
            raise ValueError(f"Got unexpected direction `{main_metric_direction}`. Must be on of {DIRECTIONS}")

        self.main_metric = main_metric
        self.main_metric_direction = main_metric_direction
        self.track_val_loss = track_val_loss

        self._is_direction_minimize = self.main_metric_direction == "minimize"

    @property
    def first_objective_metric_best(self):
        return np.Inf if self._is_direction_minimize else -np.Inf

    def main_metric_improved(self, current_value: float, prev_value: float) -> bool:
        condition_met: bool = current_value > prev_value

        if self._is_direction_minimize:
            condition_met = not condition_met

        return condition_met

    @abstractmethod
    def calculate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        return self.calculate(*args, **kwargs)

    @abstractmethod
    def prepare_predictions_init(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def prepare_predictions_batch(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

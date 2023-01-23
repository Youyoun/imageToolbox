import abc
from typing import Any, List, Dict

import numpy as np


class BaseMetric(abc.ABC):
    @abc.abstractmethod
    def update(self, value: float) -> None:
        pass

    @abc.abstractmethod
    def compute(self) -> float:
        pass

    @abc.abstractmethod
    def init(self) -> None:
        pass


class MetricAverage(BaseMetric):
    def __init__(self):
        self.value_array = []

    def update(self, value: float) -> None:
        self.value_array.append(value)

    def compute(self) -> float:
        return np.mean(self.value_array).item()

    def init(self):
        self.value_array = []


class MetricArray(BaseMetric):
    def __init__(self):
        self.value_array = []

    def update(self, value: float) -> None:
        self.value_array.append(value)

    def compute(self) -> List[float]:
        return self.value_array

    def init(self) -> None:
        self.value_array = []


class MetricsDictionary:
    def __init__(self):
        self.metrics = {}

    def add(self, metrics_dics: Dict[Any, float]) -> None:
        for k, v in metrics_dics.items():
            if k not in self.metrics:
                self.metrics[k] = MetricArray()
            self.metrics[k].update(v)

    def get_all(self) -> Dict[Any, List[float]]:
        return {k: v.compute() for k, v in self.metrics.items()}

    def __getitem__(self, item: Any) -> List[float]:
        return self.metrics[item].compute()

    def reset(self) -> None:
        for k in self.metrics:
            self.metrics[k].init()


class AverageMetricsDictionary:
    def __init__(self):
        self.metrics = {}

    def add(self, metrics_dics: Dict[Any, float]):
        for k, v in metrics_dics.items():
            if k not in self.metrics:
                self.metrics[k] = MetricAverage()
            self.metrics[k].update(v)

    def get_all(self) -> Dict[Any, float]:
        return {k: v.compute() for k, v in self.metrics.items()}

    def __getitem__(self, item: Any) -> float:
        return self.metrics[item].compute()

    def reset(self) -> None:
        for k in self.metrics:
            self.metrics[k].init()

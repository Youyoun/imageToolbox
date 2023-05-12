import abc
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


class BaseMetric(abc.ABC):
    """
    Base class for metrics
    Exposes the following methods:
    - update: update the metric with a new value
    - compute: compute the metric
    - init: reset the metric
    """

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
    """
    Compute the average of a metric
    Logs all the values and compute the mean
    """

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


class MetricsAggregator(abc.ABC):
    """
    Base class for metrics aggregator
    Exposes the following methods:
    - add: add a new metric dictionary
    - get_all: get all the metrics
    - __getitem__: get a specific metric
    - reset: reset the metrics
    """

    @abc.abstractmethod
    def add(self, metrics_dics: Dict[Any, float]) -> None:
        pass

    @abc.abstractmethod
    def get_all(self) -> Dict[Any, float]:
        pass

    @abc.abstractmethod
    def __getitem__(self, item: Any) -> float:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class MetricsDictionary(MetricsAggregator):
    """
    Compute a dictionary of MetricArray objects from a dictionary of {metrics: values}
    Returns a dictionary of {metrics: [values]}
    """

    def __init__(self):
        self.metrics: Dict[Any, MetricArray] = {}

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

    def save(self, path: Union[str, Path]) -> None:
        import json

        with open(path, "w") as f:
            json.dump(self.get_all(), f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MetricsDictionary":
        import json

        with open(path, "r") as f:
            metrics = json.load(f)
        metrics_dic = cls()
        for k, v in metrics.items():
            metrics_dic.metrics[k] = MetricArray()
            metrics_dic.metrics[k].value_array = v
        return metrics_dic


class AverageMetricsDictionary(MetricsAggregator):
    """
    Compute a dictionary of MetricAverage objects from a dictionary of {metrics: values}
    Returns a dictionary of {metrics: average}
    """

    def __init__(self):
        self.metrics: Dict[Any, MetricAverage] = {}

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

import numpy as np

from tests.parameters import NDIM_X
from toolbox.metrics import MetricsDictionary, AverageMetricsDictionary, MetricArray, MetricAverage


class TestMetricsObjects:

    @staticmethod
    def test_metrics_average():
        metrics = MetricAverage()
        values = np.random.randn(NDIM_X)
        for i, value in enumerate(values):
            metrics.update(value)
            assert metrics.compute() == np.mean(values[:i + 1]).item()

    @staticmethod
    def test_metrics_array():
        metrics = MetricArray()
        values = np.random.randn(NDIM_X)
        for i, value in enumerate(values):
            metrics.update(value)
            assert metrics.compute() == values[:i + 1].tolist()

    @staticmethod
    def test_metrics_dictionary():
        metrics = MetricsDictionary()
        values1 = np.random.randn(NDIM_X)
        values2 = np.random.randn(NDIM_X)
        for i in range(len(values1)):
            metrics.add({"values1": values1[i], "values2": values2[i]})
            assert metrics.get_all() == {"values1": values1[:i + 1].tolist(), "values2": values2[:i + 1].tolist()}

    @staticmethod
    def test_average_metrics_dictionary():
        metrics = AverageMetricsDictionary()
        values1 = np.random.randn(NDIM_X)
        values2 = np.random.randn(NDIM_X)
        for i in range(len(values1)):
            metrics.add({"values1": values1[i], "values2": values2[i]})
            assert metrics.get_all() == {"values1": np.mean(values1[:i + 1]).item(),
                                         "values2": np.mean(values2[:i + 1]).item()}

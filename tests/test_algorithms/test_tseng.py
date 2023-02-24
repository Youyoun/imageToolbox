import numpy as np
import torch

from tests.parameters import are_equal
from toolbox.algorithms.tseng_descent import tseng_gradient_descent


class TestTsengInverse:
    @staticmethod
    def test_tseng_numpy_inverse():
        w_mon = np.array([[1, 2, 3],
                          [0, 5, 6],
                          [0, 0, 9]])
        id_ = np.eye(3)
        w_inv = np.linalg.inv(w_mon)
        w_t_inv, _ = tseng_gradient_descent(
            id_,
            lambda u, y: -y,
            lambda u: w_mon @ u,
            gamma=1.0,
            lambda_=1.0,
            use_armijo=True,
            max_iter=100,
            do_compute_metrics=False
        )
        assert are_equal(w_inv, w_t_inv)

    @staticmethod
    def test_tseng_numpy_inverse_full_random():
        NDIM_X = 32
        w_mon = np.random.rand(NDIM_X, NDIM_X)
        w_mon = 0.5 * w_mon @ w_mon.T
        u, S, v = np.linalg.svd(w_mon)
        S[S < 1] = 1  # Make the smallest ev 1, for speed of convergence
        S[S > 5] = 5  # Make the largest ev 5, for speed of convergence
        w_mon = u @ np.diag(np.abs(S)) @ v
        identity_matrix = np.eye(NDIM_X)
        w_inv = np.linalg.inv(w_mon)
        w_t_inv, _ = tseng_gradient_descent(
            identity_matrix,
            lambda u, y: -y,
            lambda u: w_mon @ u,
            gamma=1.0,
            lambda_=1.0,
            use_armijo=True,
            max_iter=30000,
            do_compute_metrics=False
        )
        assert are_equal(w_inv, w_t_inv), f"{abs(w_inv - w_t_inv).max()}"

    @staticmethod
    def test_tseng_torch_inverse():
        w_mon = torch.Tensor([[1, 2, 3],
                              [0, 5, 6],
                              [0, 0, 9]])
        id_ = torch.eye(3)
        w_inv = torch.linalg.inv(w_mon)
        w_t_inv, _ = tseng_gradient_descent(
            id_,
            lambda u, y: -y,
            lambda u: w_mon @ u,
            gamma=1.0,
            lambda_=1.0,
            use_armijo=True,
            max_iter=100,
            do_compute_metrics=False
        )
        assert are_equal(w_inv, w_t_inv)

    @staticmethod
    def test_tseng_torch_inverse_full_random():
        NDIM_X = 32
        w_mon = torch.randn(NDIM_X, NDIM_X)
        w_mon = 0.5 * w_mon @ w_mon.T
        u, S, v = torch.linalg.svd(w_mon)
        S[S < 1] = 1  # Make the smallest ev 1, for speed of convergence
        S[S > 5] = 5  # Make the largest ev 5, for speed of convergence
        w_mon = u @ np.diag(np.abs(S)) @ v
        identity_matrix = torch.eye(NDIM_X)
        w_inv = torch.linalg.inv(w_mon)
        w_t_inv, _ = tseng_gradient_descent(
            identity_matrix,
            lambda u, y: -y,
            lambda u: w_mon @ u,
            gamma=1.0,
            lambda_=1.0,
            use_armijo=True,
            max_iter=30000,
            do_compute_metrics=False
        )
        assert are_equal(w_inv, w_t_inv), f"{abs(w_inv - w_t_inv).max()}"


class TestTsengConvergence:
    @staticmethod
    def test_tseng_quadratic_to_zero():
        NDIM_X = 32

        def grad_xF(x, y):
            return x

        def grad_xR(x):
            return torch.zeros_like(x)

        x = torch.randn(NDIM_X)
        y_, _ = tseng_gradient_descent(
            x,
            grad_xF,
            grad_xR,
            gamma=1.0,
            lambda_=1.0,
            use_armijo=True,
            max_iter=2000,
            do_compute_metrics=False
        )
        assert are_equal(y_, torch.zeros_like(y_)), f"{abs(y_).max()}"

    @staticmethod
    def test_tseng_quadratic_to_ones():
        NDIM_X = 32

        def grad_xF(x, y):
            return x - torch.ones_like(x)

        def grad_xR(x):
            return torch.zeros_like(x)

        x = torch.randn(NDIM_X)
        y_, _ = tseng_gradient_descent(
            x,
            grad_xF,
            grad_xR,
            gamma=1.0,
            lambda_=1.0,
            use_armijo=True,
            max_iter=2000,
            do_compute_metrics=False
        )
        assert are_equal(y_, torch.ones_like(y_)), f"{abs(y_ - torch.ones_like(y_)).max()}"

    @staticmethod
    def test_tseng_quadratic_to_random():
        NDIM_X = 32
        Y = torch.randn(NDIM_X)

        def grad_xF(x, y):
            return x - Y

        def grad_xR(x):
            return torch.zeros_like(x)

        x = torch.randn(NDIM_X)
        y_, _ = tseng_gradient_descent(
            x,
            grad_xF,
            grad_xR,
            gamma=1.0,
            lambda_=1.0,
            use_armijo=True,
            max_iter=2000,
            do_compute_metrics=False
        )
        assert are_equal(y_, Y), f"{abs(y_ - Y).max()}"
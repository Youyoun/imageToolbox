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

import itertools

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian

from tests.parameters import BATCH_SIZE, FLOAT_TOL, are_equal
from toolbox.jacobian import Ju, JTu, alpha_operator, sum_J_JT, compute_jacobian, batch_jacobian, \
    PenalizationMethods, MonotonyRegularization, power_method, \
    get_min_max_ev_neuralnet_fulljacobian, get_lambda_min_or_max_poweriter, generate_new_prediction, \
    get_neuralnet_jacobian_ev, MonotonyRegularizationShift

PERCENT_CLOSE = 0.9
NDIM = 50


class TestJacobienVectorProduct:

    @staticmethod
    def test_ju_x2():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        f = lambda x: x * x
        u = torch.randn((BATCH_SIZE, NDIM, 1))
        ju = Ju(x, f(x), u, True)

        with torch.no_grad():
            assert torch.isclose(ju, 2 * x * u, rtol=FLOAT_TOL).all()

    @staticmethod
    def test_ju_linear():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.randn((NDIM, NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        u = torch.randn((BATCH_SIZE, NDIM, 1))
        ju = Ju(x, f(x), u, True)

        with torch.no_grad():
            assert torch.isclose(ju, torch.matmul(W, u), rtol=FLOAT_TOL).all()

    @staticmethod
    def test_jtu_x2():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        f = lambda x: x * x
        u = torch.randn((BATCH_SIZE, NDIM, 1))
        jTu = JTu(x, f(x), u, True)

        with torch.no_grad():
            assert torch.isclose(jTu, 2 * x * u, rtol=FLOAT_TOL).all()

    @staticmethod
    def test_jtu_linear():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.randn((NDIM, NDIM)) * 10
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        u = torch.randn((BATCH_SIZE, NDIM, 1))
        jTu = JTu(x, f(x), u, True)

        with torch.no_grad():
            assert torch.isclose(jTu, torch.matmul(W.transpose(1, 2), u), rtol=FLOAT_TOL).all()

    @staticmethod
    @pytest.mark.parametrize('execution_number', range(50))
    def test_adjoint_linear(execution_number):
        x = torch.randn(BATCH_SIZE, NDIM, dtype=torch.float32).requires_grad_()
        model = nn.Sequential(nn.Linear(NDIM, NDIM, bias=True),
                              nn.ReLU(),
                              nn.Linear(NDIM, NDIM, bias=True),
                              nn.ReLU(),
                              nn.Linear(NDIM, NDIM, bias=True))

        u1 = torch.randn_like(x, requires_grad=True)
        u2 = torch.randn_like(x, requires_grad=True)

        Au1 = Ju(x, model(x), u1, is_eval=False)
        Atu2 = JTu(x, model(x), u2, is_eval=False)

        z1 = torch.matmul(u2.reshape(u2.shape[0], 1, -1), Au1.reshape(u2.shape[0], -1, 1))  # <u_2, Atu_1>
        z2 = torch.matmul(Atu2.reshape(u2.shape[0], 1, -1), u1.reshape(u2.shape[0], -1, 1))  # <Atu_2, u_1>

        assert torch.isclose(z1, z2, atol=FLOAT_TOL, rtol=FLOAT_TOL).all()

    @staticmethod
    @pytest.mark.parametrize('execution_number', range(50))
    def test_adjoint_convolution(execution_number):
        x = torch.randn(BATCH_SIZE, 1, NDIM, NDIM, dtype=torch.float32).requires_grad_()
        model = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1, stride=1, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(64, 1, 3, padding=1, stride=1, bias=True))

        u1 = torch.randn_like(x, requires_grad=True)
        u2 = torch.randn_like(x, requires_grad=True)

        Au1 = Ju(x, model(x), u1, is_eval=False)
        Atu2 = JTu(x, model(x), u2, is_eval=False)

        z1 = torch.matmul(u2.reshape(u2.shape[0], 1, -1), Au1.reshape(u2.shape[0], -1, 1))  # <u_2, Atu_1>
        z2 = torch.matmul(Atu2.reshape(u2.shape[0], 1, -1), u1.reshape(u2.shape[0], -1, 1))  # <Atu_2, u_1>

        assert torch.isclose(z1, z2, atol=FLOAT_TOL, rtol=FLOAT_TOL).all()


class TestJacobianOperator:
    @staticmethod
    @pytest.mark.parametrize('alpha', [5, 10, 13, 20])
    def test_alpha_operator(alpha):
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        u = torch.randn((BATCH_SIZE, NDIM, 1))
        w = torch.randn((NDIM, NDIM)) * 10
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        op = alpha_operator(x, f(x), u, alpha, is_eval=True)
        with torch.no_grad():
            truth = alpha * u - 1 / 2 * torch.matmul(W + W.transpose(1, 2), u)

        is_close = torch.isclose(op, truth, rtol=FLOAT_TOL, atol=FLOAT_TOL)
        assert is_close.sum() >= NDIM * BATCH_SIZE * 0.95, (op[~is_close] - truth[~is_close])

    @staticmethod
    def test_sum_j_jt():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        u = torch.randn((BATCH_SIZE, NDIM, 1))
        w = torch.randn((NDIM, NDIM)) * 10
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        op = sum_J_JT(x, f(x), u, True)
        with torch.no_grad():
            truth = 1 / 2 * torch.matmul(W + W.transpose(1, 2), u)

        is_close = torch.isclose(op, truth, rtol=FLOAT_TOL)
        assert is_close.sum() >= NDIM * BATCH_SIZE * 0.95, (op[~is_close] - truth[~is_close])

    @staticmethod
    @pytest.mark.parametrize(['channels', 'kernel_size'], [[1, 3], [1, 5], [3, 3], [3, 5]])
    def test_compute_jacobian_size(channels, kernel_size):
        x = torch.randn((BATCH_SIZE, channels, NDIM, NDIM), requires_grad=True)
        kernel = torch.randn((channels, channels, kernel_size, kernel_size))
        f = lambda x: F.conv2d(x, kernel, stride=1, padding=kernel_size // 2)

        jacob = compute_jacobian(f, x)

        assert jacob.shape == (BATCH_SIZE, channels * NDIM * NDIM, channels * NDIM * NDIM)

    @staticmethod
    def test_batch_jacobian():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.randn((NDIM, NDIM))
        f = lambda x: torch.matmul(w.unsqueeze(0), x)

        batched_jacob = batch_jacobian(f, x, True)
        jacobs = []
        for i in range(BATCH_SIZE):
            j = jacobian(f, x[i].unsqueeze(0), create_graph=False, vectorize=True)
            jacobs.append(j)
        jacobs = torch.cat(jacobs)
        assert torch.isclose(jacobs.squeeze().transpose(0, 1), batched_jacob.squeeze(), rtol=FLOAT_TOL).all()


class TestEVComputation:
    @staticmethod
    def test_power_method():
        x = torch.randn((BATCH_SIZE, NDIM), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        operator = lambda x: torch.matmul(W, x.unsqueeze(-1)).squeeze()
        lambda_max = power_method(x, operator, max_iter=1000, tol=1e-6, is_eval=True)
        true_lambda_max = w.flatten()[torch.abs(w).argmax()]
        assert are_equal(lambda_max, true_lambda_max), (lambda_max - true_lambda_max)

    @staticmethod
    def test_get_jacobian_ev():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        all_ev = get_neuralnet_jacobian_ev(f, x)
        assert are_equal(all_ev, torch.sort(torch.diag(w))[0])

    @staticmethod
    def test_get_min_max_ev_fulljacobian():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        ev_min, ev_max = get_min_max_ev_neuralnet_fulljacobian(f, x)
        assert are_equal(ev_min, w.min()) and are_equal(ev_max, w.max())

    @staticmethod
    @pytest.mark.parametrize('alpha', [0.0, 1.0, 2.0, 5.0])
    def test_get_lambda_max_min(alpha: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        ev_min = get_lambda_min_or_max_poweriter(f, x, alpha, is_eval=True, biggest=False, n_iter=_niter)
        ev_largest_abs = get_lambda_min_or_max_poweriter(f, x, None, is_eval=True, biggest=True, n_iter=_niter)
        true_lambda_max = w.max()
        true_lambda_min = w.min()
        abs_lambda_max = w.flatten()[torch.abs(w).argmax()]
        if alpha > (true_lambda_max + true_lambda_min) / 2:
            assert are_equal(ev_min, true_lambda_min), f"{(true_lambda_min - ev_min).abs().max()}"
        else:
            assert are_equal(ev_min, true_lambda_max), f"{(true_lambda_max - ev_min).max()}"
        assert are_equal(ev_largest_abs, abs_lambda_max), f"{(abs_lambda_max - ev_largest_abs).abs().max()}"


class TestPenalization:
    EPS = [0.0, 0.05, 0.1, 0.2]
    ALPHAS = [3.0, 5.0]

    @staticmethod
    @pytest.mark.parametrize('eps', EPS)
    def test_penalization_fulljacobian(eps: float):
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(eps=eps,
                                             eval_mode=True,
                                             method=PenalizationMethods.EVDECOMP,
                                             use_relu_penalization=False)(f, x)
        assert are_equal(pen, -w.min())
        assert are_equal(ev_min, w.min())

    @staticmethod
    @pytest.mark.parametrize('eps', EPS)
    def test_penalization_fulljacobian_relu(eps: float):
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(eps=eps,
                                             eval_mode=True,
                                             method=PenalizationMethods.EVDECOMP,
                                             use_relu_penalization=True)(f, x)
        assert are_equal(pen, (eps - w.min()) ** 2)
        assert are_equal(ev_min, w.min())

    @staticmethod
    def test_generate_new_prediction():
        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        y = f(x)
        x_new, y_new = generate_new_prediction(f, x)
        assert are_equal(x.squeeze(), x_new) and are_equal(y.squeeze(), y_new)

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_onevector(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((1, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             power_iter_tol=1e-10,
                                             use_relu_penalization=False)(f, x)
        assert are_equal(pen, -w.min()), f"{pen=} {(eps - w.min()) ** 2=}"
        assert are_equal(ev_min, w.min()), f"{ev_min=} {w.min()=}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_onevector_relu(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((1, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             power_iter_tol=1e-10,
                                             use_relu_penalization=True)(f, x)
        assert are_equal(pen, (eps - w.min()) ** 2), f"{pen=} {(eps - w.min()) ** 2=}"
        assert are_equal(ev_min, w.min()), f"{ev_min=} {w.min()=}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_batchvector(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter)(f, x)
        assert are_equal(pen, - w.min()), f"{pen=} {- w.min()=}"
        assert are_equal(ev_min, w.min()), f"{ev_min=} {w.min()=}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_batchvector_relu(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        assert are_equal(pen, (eps - w.min()) ** 2), f"{pen=} {(eps - w.min()) ** 2=}"
        assert are_equal(ev_min, w.min()), f"{ev_min=} {w.min()=}"


class TestOptimizedPenalizationPowerMethodReLU:
    EPS = [0.0, 0.05, 0.1, 0.2]
    ALPHAS = [3.0, 5.0]

    @staticmethod
    def test_power_method():
        x = torch.randn((BATCH_SIZE, NDIM, 1))
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        operator = lambda x: torch.matmul(W, x)
        lambda_max = power_method(x, operator, max_iter=1000, tol=1e-6, is_eval=True)
        true_lambda_max = w.flatten()[torch.abs(w).argmax()]
        assert are_equal(lambda_max, true_lambda_max), (lambda_max - true_lambda_max).mean()

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, 1))
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter)(f, x)
        assert are_equal(pen, - w.min()), f"{pen=} {- w.min()=}"
        assert are_equal(ev_min, w.min()), f"{ev_min=} {w.min()=}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_relu(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, 1))
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        assert are_equal(pen, (eps - w.min()) ** 2), f"{pen=} {(eps - w.min()) ** 2}"
        assert are_equal(ev_min, w.min()), f"{ev_min=} {w.min()=}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_optpowermethod_random_diag(alpha: float, eps: float):
        _niter = 500

        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter)(f, x)
        assert are_equal(pen, - w.min()), f"{pen} != {- w.min()}"
        assert are_equal(ev_min, w.min()), f"{ev_min} != {w.min()}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_optpowermethod_random_diag_relu(alpha: float, eps: float):
        _niter = 500

        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        assert are_equal(pen, (eps - w.min()) ** 2), f"{pen} != {eps - 2 * w.min()}"
        assert are_equal(ev_min, w.min()), f"{ev_min} != {2 * w.min()}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_optpowermethod_random_symmetric(alpha: float, eps: float):
        NDIM = 10
        _niter = 500

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        all_ev = get_neuralnet_jacobian_ev(f, x)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter)(f, x)
        assert are_equal(pen, - all_ev.min())
        assert are_equal(ev_min, all_ev.min())

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_optpowermethod_random_symmetric_relu(alpha: float, eps: float):
        NDIM = 10
        _niter = 500

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        all_ev = get_neuralnet_jacobian_ev(f, x)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        assert are_equal(pen, (eps - all_ev.min()) ** 2)
        assert are_equal(ev_min, all_ev.min())

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod(alpha: float, eps: float):
        _niter = 500

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter)(f, x)
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=True,
                                                     max_iter=_niter)(f, x)
        assert are_equal(pen, pen_opt), f"{pen} != {pen_opt}"
        assert are_equal(ev_min, ev_min_opt), f"{ev_min} != {ev_min_opt}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod_relu(alpha: float, eps: float):
        _niter = 500

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=True,
                                                     max_iter=_niter,
                                                     use_relu_penalization=True)(f, x)
        assert are_equal(pen, pen_opt), f"{pen} != {pen_opt}"
        assert are_equal(ev_min, ev_min_opt), f"{ev_min} != {ev_min_opt}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod_grads(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        W.requires_grad_(True)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=False,
                                             max_iter=_niter)(f, x)
        pen.backward()
        grad_pm = W.grad.clone()
        W.grad = None
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=False,
                                                     max_iter=_niter)(f, x)
        pen_opt.backward()
        grad_pm_opt = W.grad.clone()
        assert torch.isclose(grad_pm, grad_pm_opt, rtol=0.0, atol=FLOAT_TOL).all(), \
            f"||gradP - gradRC|| // ||gradP|| = {torch.norm(grad_pm_opt - grad_pm) / torch.norm(grad_pm)}, " \
            f"{torch.isclose(grad_pm, grad_pm_opt, rtol=FLOAT_TOL).sum()} / {grad_pm.nelement()}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod_grads_percentclose(alpha: float, eps: float):
        _niter = 2000

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        W.requires_grad_()
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=False,
                                             max_iter=_niter)(f, x)
        pen.backward()
        grad_pm = W.grad.clone()
        W.grad = None
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=False,
                                                     max_iter=_niter)(f, x)
        pen_opt.backward()
        grad_pm_opt = W.grad.clone()
        assert torch.isclose(grad_pm, grad_pm_opt, rtol=0.0, atol=FLOAT_TOL).sum() > grad_pm.nelement() * PERCENT_CLOSE, \
            f"||gradP - gradRC|| // ||gradP|| = {torch.norm(grad_pm_opt - grad_pm) / torch.norm(grad_pm)}, " \
            f"{are_equal(grad_pm, grad_pm_opt).sum()} / {grad_pm.nelement()}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod_grads_relu(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        W.requires_grad_(True)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=False,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        pen.backward()
        grad_pm = W.grad.clone()
        W.grad = None
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=False,
                                                     max_iter=_niter,
                                                     use_relu_penalization=True)(f, x)
        pen_opt.backward()
        grad_pm_opt = W.grad.clone()
        assert torch.isclose(grad_pm, grad_pm_opt, rtol=0.0, atol=FLOAT_TOL).all(), \
            f"||gradP - gradRC|| // ||gradP|| = {torch.norm(grad_pm_opt - grad_pm) / torch.norm(grad_pm)}, " \
            f"{torch.isclose(grad_pm, grad_pm_opt, rtol=FLOAT_TOL).sum()} / {grad_pm.nelement()}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod_grads_percentclose_relu(alpha: float, eps: float):
        _niter = 2000

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        W.requires_grad_()
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=False,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        pen.backward()
        grad_pm = W.grad.clone()
        W.grad = None
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=False,
                                                     max_iter=_niter,
                                                     use_relu_penalization=True)(f, x)
        pen_opt.backward()
        grad_pm_opt = W.grad.clone()
        assert torch.isclose(grad_pm, grad_pm_opt, rtol=0.0, atol=FLOAT_TOL).sum() > grad_pm.nelement() * PERCENT_CLOSE, \
            f"||gradP - gradRC|| // ||gradP|| = {torch.norm(grad_pm_opt - grad_pm) / torch.norm(grad_pm)}, " \
            f"{are_equal(grad_pm, grad_pm_opt).sum()} / {grad_pm.nelement()}"


class TestOptimizedNoAlphaPenalizationPowerMethodReLU:
    EPS = [0.0, 0.05, 0.1, 0.2]
    ALPHAS = [3.0, 5.0]

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_random_diag(alpha: float, eps: float):
        _niter = 500

        x = torch.randn((BATCH_SIZE, NDIM, 1), requires_grad=True)
        w = torch.diag(torch.randn(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)

        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.OPTPOWERNOALPHA,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        assert are_equal(pen, (eps - w.min()) ** 2), f"{pen} != {eps - w.min()}"
        assert are_equal(ev_min, w.min()), f"{ev_min} != {w.min()}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_random_symmetric(alpha: float, eps: float):
        NDIM = 10
        _niter = 500

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        all_ev = get_neuralnet_jacobian_ev(f, x)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.OPTPOWERNOALPHA,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        assert are_equal(pen, (eps - all_ev.min()) ** 2)
        assert are_equal(ev_min, all_ev.min())

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod_noalpha(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=True,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWERNOALPHA,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=True,
                                                     max_iter=_niter,
                                                     use_relu_penalization=True)(f, x)
        assert are_equal(ev_min, ev_min_opt), f"{ev_min} != {ev_min_opt}"
        assert are_equal(pen, pen_opt), f"{pen} != {pen_opt}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod_noalpha_grads(alpha: float, eps: float):
        _niter = 2000

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        W.requires_grad_()
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=False,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        pen.backward()
        grad_pm = W.grad.clone()
        W.grad = None
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWERNOALPHA,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=False,
                                                     max_iter=_niter,
                                                     use_relu_penalization=True)(f, x)
        pen_opt.backward()
        grad_pm_opt = W.grad.clone()
        assert are_equal(grad_pm, grad_pm_opt), \
            f"{pen.item()=} {pen_opt.item()=}" \
            f"||gradP - gradRC|| // ||gradP|| = {torch.norm(grad_pm_opt - grad_pm) / torch.norm(grad_pm)}, " \
            f"{torch.isclose(grad_pm, grad_pm_opt, rtol=FLOAT_TOL).sum()} / {grad_pm.nelement()}"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHAS, EPS))
    def test_penalization_powermethod_optpowermethod_noalpha_grads_percentclose(alpha: float, eps: float):
        _niter = 1000

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = w.T @ w
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        W.requires_grad_()
        pen, ev_min = MonotonyRegularization(method=PenalizationMethods.POWER,
                                             alpha=alpha,
                                             eps=eps,
                                             eval_mode=False,
                                             max_iter=_niter,
                                             use_relu_penalization=True)(f, x)
        pen.backward()
        grad_pm = W.grad.clone()
        W.grad = None
        pen_opt, ev_min_opt = MonotonyRegularization(method=PenalizationMethods.OPTPOWERNOALPHA,
                                                     alpha=alpha,
                                                     eps=eps,
                                                     eval_mode=False,
                                                     max_iter=_niter,
                                                     use_relu_penalization=True)(f, x)
        pen_opt.backward()
        grad_pm_opt = W.grad.clone()
        assert torch.isclose(grad_pm, grad_pm_opt, rtol=0.0,
                             atol=FLOAT_TOL).sum() > grad_pm.nelement() * PERCENT_CLOSE, \
            f"||gradP - gradRC|| // ||gradP|| = {torch.norm(grad_pm_opt - grad_pm) / torch.norm(grad_pm)}, " \
            f"{torch.isclose(grad_pm, grad_pm_opt, rtol=FLOAT_TOL).sum()} / {grad_pm.nelement()}"


class TestMonotonyLearningReLUPen:
    DEVICE = "cuda:0"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product([10], [0.1, 0.05]))
    def test_learning_monotony_power(alpha: float, eps: float):
        _niter = 100
        _max_iter = 200

        model = nn.Linear(NDIM, NDIM, bias=True).to(TestMonotonyLearningReLUPen.DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        optim.zero_grad()

        for _ in range(_max_iter):
            x = torch.randn((BATCH_SIZE, NDIM)).to(TestMonotonyLearningReLUPen.DEVICE)
            pen, evs = MonotonyRegularization(method=PenalizationMethods.POWER,
                                              alpha=alpha,
                                              eps=eps,
                                              eval_mode=False,
                                              max_iter=_niter,
                                              use_relu_penalization=True)(model, x)
            optim.zero_grad()
            pen.backward()
            optim.step()
            if evs.min() >= eps:
                break
        assert torch.linalg.eigvalsh(1 / 2 * (model.weight + model.weight.T).detach().cpu())[0] >= 0.0

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product([10], [0.1, 0.05]))
    def test_learning_monotony_poweropt(alpha: float, eps: float):
        _niter = 200
        _max_iter = 500

        model = nn.Linear(NDIM, NDIM, bias=True).to(TestMonotonyLearningReLUPen.DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        optim.zero_grad()

        for _ in range(_max_iter):
            x = torch.randn((BATCH_SIZE, NDIM)).to(TestMonotonyLearningReLUPen.DEVICE)
            pen, evs = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                              alpha=alpha,
                                              eps=eps,
                                              eval_mode=False,
                                              max_iter=_niter,
                                              use_relu_penalization=True)(model, x)
            optim.zero_grad()
            pen.backward()
            optim.step()
            if evs.min() >= eps:
                break
        assert torch.linalg.eigvalsh(1 / 2 * (model.weight + model.weight.T).detach().cpu())[0] >= 0.0

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product([10], [0.1, 0.05]))
    def test_learning_monotony_jacobian(alpha: float, eps: float):
        _max_iter = 50

        model = nn.Linear(NDIM, NDIM, bias=True).to(TestMonotonyLearningReLUPen.DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        optim.zero_grad()

        for _ in range(_max_iter):
            x = torch.randn((BATCH_SIZE, NDIM)).to(TestMonotonyLearningReLUPen.DEVICE)
            pen, evs = MonotonyRegularization(method=PenalizationMethods.EVDECOMP,
                                              eps=eps,
                                              eval_mode=False,
                                              use_relu_penalization=True)(model, x)
            optim.zero_grad()
            pen.backward()
            optim.step()
            if evs.min() >= eps:
                break
        assert torch.linalg.eigvalsh(1 / 2 * (model.weight + model.weight.T).detach().cpu())[0] >= 0.0


class TestMonotonyLearningPen:
    ALPHA = [10]
    EPS = [0.1, 0.05]
    DEVICE = "cuda:0"

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHA, EPS))
    def test_learning_monotony_power(alpha: float, eps: float):
        _niter = 100
        _max_iter = 200

        model = nn.Linear(NDIM, NDIM, bias=True).to(TestMonotonyLearningPen.DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        optim.zero_grad()

        for _ in range(_max_iter):
            x = torch.randn((BATCH_SIZE, NDIM)).to(TestMonotonyLearningPen.DEVICE)
            pen, evs = MonotonyRegularization(method=PenalizationMethods.POWER,
                                              alpha=alpha,
                                              eps=eps,
                                              eval_mode=False,
                                              max_iter=_niter)(model, x)
            print(pen)
            optim.zero_grad()
            pen.backward()
            optim.step()
            if evs.min() >= eps:
                break
        assert torch.linalg.eigvalsh(1 / 2 * (model.weight + model.weight.T).detach().cpu())[0] >= 0.0

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHA, EPS))
    def test_learning_monotony_poweropt(alpha: float, eps: float):
        _niter = 200
        _max_iter = 500

        model = nn.Linear(NDIM, NDIM, bias=True).to(TestMonotonyLearningPen.DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        optim.zero_grad()

        for _ in range(_max_iter):
            x = torch.randn((BATCH_SIZE, NDIM)).to(TestMonotonyLearningPen.DEVICE)
            pen, evs = MonotonyRegularization(method=PenalizationMethods.OPTPOWER,
                                              alpha=alpha,
                                              eps=eps,
                                              eval_mode=False,
                                              max_iter=_niter)(model, x)
            optim.zero_grad()
            pen.backward()
            optim.step()
            if evs.min() >= eps:
                break
        assert torch.linalg.eigvalsh(1 / 2 * (model.weight + model.weight.T).detach().cpu())[0] >= 0.0

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps"], itertools.product(ALPHA, EPS))
    def test_learning_monotony_jacobian(alpha: float, eps: float):
        _max_iter = 50

        model = nn.Linear(NDIM, NDIM, bias=True).to(TestMonotonyLearningPen.DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        optim.zero_grad()

        for _ in range(_max_iter):
            x = torch.randn((BATCH_SIZE, NDIM)).to(TestMonotonyLearningPen.DEVICE)
            pen, evs = MonotonyRegularization(method=PenalizationMethods.EVDECOMP,
                                              eps=eps,
                                              eval_mode=False)(model, x)
            optim.zero_grad()
            pen.backward()
            optim.step()
            if evs.min() >= eps:
                break
        assert torch.linalg.eigvalsh(1 / 2 * (model.weight + model.weight.T).detach().cpu())[0] >= 0.0


class TestConstraintValuesWithTransformedConstraint:
    ALPHA = [10]
    EPS = [0.1, 0.05]

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps", "method"],
                             itertools.product(ALPHA, EPS,
                                               [PenalizationMethods.POWER,
                                                PenalizationMethods.OPTPOWER,
                                                PenalizationMethods.EVDECOMP]))
    def test_values_equality_relu(alpha: float, eps: float, method: PenalizationMethods):
        _niter = 200
        BATCH_SIZE = 1
        NDIM = 30

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = 1 / 2 * (w.T + w)
        real_evs1 = torch.linalg.eigvalsh(W)
        real_evs2 = torch.linalg.eigvalsh(2 * W.T - torch.eye(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        f2 = lambda x: 2 * torch.matmul(W, x) - x
        method_ = MonotonyRegularizationShift(method=method, alpha=alpha, eps=eps, eval_mode=True, max_iter=_niter,
                                              use_relu_penalization=True)
        method_2 = MonotonyRegularization(method=method, alpha=alpha, eps=-1 + eps, eval_mode=True, max_iter=_niter,
                                          use_relu_penalization=True)
        pen1, ev_min1 = method_2(f2, x)
        pen2, ev_min2 = method_(f, x)
        # The square complicates some stuff, so the precision is lowered for the square
        assert torch.isclose(pen1, pen2, atol=FLOAT_TOL * 10, rtol=0.0).all()
        assert torch.isclose(ev_min1, ev_min2, atol=FLOAT_TOL, rtol=0.0).all()
        assert torch.isclose(real_evs1[0], (ev_min2 + 1) / 2, atol=FLOAT_TOL, rtol=0.0).all()
        assert torch.isclose(max(0, (- 1 + eps - real_evs2[0])) ** 2, pen1, atol=FLOAT_TOL * 10, rtol=0.0).all()

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps", "method"],
                             itertools.product(ALPHA, EPS,
                                               [PenalizationMethods.POWER,
                                                PenalizationMethods.OPTPOWER,
                                                PenalizationMethods.EVDECOMP]))
    def test_values_equality(alpha: float, eps: float, method: PenalizationMethods):
        _niter = 200
        BATCH_SIZE = 1
        NDIM = 30

        x = torch.randn((BATCH_SIZE, NDIM, NDIM))
        w = torch.randn(NDIM, NDIM)
        W = 1 / 2 * (w.T + w)
        real_evs1 = torch.linalg.eigvalsh(W)
        real_evs2 = torch.linalg.eigvalsh(2 * W.T - torch.eye(NDIM))
        W = w.unsqueeze(0)
        f = lambda x: torch.matmul(W, x)
        f2 = lambda x: 2 * torch.matmul(W, x) - x
        method_ = MonotonyRegularizationShift(method=method, alpha=alpha, eps=eps, eval_mode=True, max_iter=_niter)
        method_2 = MonotonyRegularization(method=method, alpha=alpha, eps=-1 + eps, eval_mode=True, max_iter=_niter)
        pen1, ev_min1 = method_2(f2, x)
        pen2, ev_min2 = method_(f, x)
        assert torch.isclose(pen1, pen2, atol=FLOAT_TOL, rtol=0.0).all()
        assert torch.isclose(ev_min1, ev_min2, atol=FLOAT_TOL, rtol=0.0).all()
        assert torch.isclose(real_evs1[0], (ev_min2 + 1) / 2, atol=FLOAT_TOL, rtol=0.0).all()
        assert torch.isclose(torch.maximum(torch.ones(1) - eps, - real_evs2[0]), pen1, atol=FLOAT_TOL, rtol=0.0).all()


class TestMonotonyLearningWithTransformedConstraint:
    DEVICE = "cuda:0"
    ALPHA = [10]
    EPS = [0.1, 0.05]

    @staticmethod
    @pytest.mark.parametrize(["alpha", "eps", "method"],
                             itertools.product(ALPHA, EPS,
                                               [PenalizationMethods.POWER,
                                                PenalizationMethods.OPTPOWER,
                                                PenalizationMethods.EVDECOMP]))
    def test_learning_monotony_power(alpha: float, eps: float, method: PenalizationMethods):
        _niter = 100
        _max_iter = 200

        model = nn.Linear(NDIM, NDIM, bias=True).to(TestMonotonyLearningWithTransformedConstraint.DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        optim.zero_grad()
        pmethod = MonotonyRegularizationShift(method=method, alpha=alpha, eps=eps, max_iter=_niter)
        for _ in range(_max_iter):
            x = torch.randn((BATCH_SIZE, NDIM)).to(TestMonotonyLearningWithTransformedConstraint.DEVICE)
            pen, evs = pmethod(model, x)
            optim.zero_grad()
            pen.backward()
            optim.step()
            if evs.min() >= -1 + eps:
                break
        assert torch.linalg.eigvalsh(1 / 2 * (model.weight + model.weight.T).detach().cpu())[0] >= 0.0

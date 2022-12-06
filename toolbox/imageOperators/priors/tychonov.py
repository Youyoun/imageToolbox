from src.Function import Function


class Tychonov(Function):
    def __init__(self, op, mean):
        self.L = op
        self.mean = mean

    def f(self, x):
        if x.ndim == 4:
            return (self.L @ (x - self.mean)).pow(2).sum([1, 2, 3])  # Leave Batchsize
        else:
            return (self.L @ (x - self.mean)).pow(2).sum()

    def grad(self, x):
        return self.L.T @ (self.L @ (x - self.mean)) * 2

    def autograd(self, x):
        x_ = x.detach().clone()
        x_.requires_grad_()

        self.f(x_).backward()
        return x_.grad

from abc import ABC


class BasicSolver(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, *args, **kwargs):
        raise NotImplementedError()

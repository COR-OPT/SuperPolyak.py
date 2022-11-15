import scipy.sparse.linalg
import torch.autograd.functional
from torch.optim.optimizer import Optimizer
import numpy as np
from scipy.sparse.linalg import lsmr
from typing import Callable, Iterable



class NewtonCG(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
    ):

        super(NewtonCG, self).__init__(params, dict())

        self._params = self.param_groups[0]["params"]
        self._H = np.zeros((len(self._params), len(self._params)))

    def step(self, closure):

        def mv(v):
            grad = torch.autograd.grad(closure(self._params[0]), self._params[0], create_graph=True)[0]
            loss = grad.dot(torch.from_numpy(v))
            loss.backward()
            return self._params[0].grad
        H_np = scipy.sparse.linalg.LinearOperator((len(self._params[0]), len(self._params[0])), matvec=mv, rmatvec=mv)
        g = torch.autograd.grad(closure(self._params[0]), self._params[0], create_graph=True)[0]
        g_np = g.detach().clone().numpy()
        gap = g.norm().item()
        step = lsmr(
            H_np,
            g_np,
            atol=max(1e-15, 0.01 * gap),
            btol=max(1e-15, 0.01 * gap),
            conlim=0.0
        )[0]
        self._params[0].data = self._params[0].data - step
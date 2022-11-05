import torch
from torch.optim.optimizer import Optimizer
import numpy as np

from enum import Enum
from typing import Callable, Optional, Iterable

import numpy.linalg as linalg
from scipy.sparse.linalg import lsmr


class BundleLinearSystemSolver(Enum):
    """Contains the available linear system solvers for PolyakBundle."""

    NAIVE = 1
    LSMR = 2

class SuperPolyak(Optimizer):

    def __init__(self,
                 params: Iterable[torch.Tensor],
                 tau: float = np.inf,
                 min_f: float = 0,
                 eta_est: float = .1,
                 max_elts: Optional[int] = 1,
                 linsys_solver: BundleLinearSystemSolver = BundleLinearSystemSolver.NAIVE):
        defaults = dict(lr=1) # lr is not used, can it be removed?
        super(SuperPolyak, self).__init__(params, defaults)
        # """Build a Polyak bundle given a loss function and an initial point.
        #
        # Parameters
        # ----------
        # closure: Callable
        #     A closure that evaluates the objective function.
        # params : Iterable[torch.Tensor]
        #     The center point of the bundle.
        # tau : float
        #     The trust region parameter.
        # eta_est : float
        #     An estimate of the b-regularity exponent.
        # max_elts: int, optional
        #     The maximal number of elements in the bundle.
        # linsys_solver: BundleLinearSystemSolver (default: NAIVE)
        #     The linear system solver to use.
        # """
        # self.num_oracle = []
        self.num_oracle_iter = 0
        if len(self.param_groups) != 1:
            raise ValueError("SuperPolyak doesn't support per-parameter options "
                             "(parameter groups)")
        self.loss = np.inf
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.tau = tau
        self.min_f = min_f
        self.eta_est = eta_est
        self.max_elts = max_elts
        self.linsys_solver = linsys_solver

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    @torch.no_grad()
    def build_bundle_torch_param_groups(self,closure: Callable):

        # params = list(self.params)
        y0 = torch.nn.utils.parameters_to_vector(self._params).detach().clone().numpy()
        d = len(y0)
        with torch.enable_grad():
            fy0 = closure().item()
        gap = fy0 - self.min_f
        # If no value was given for `max_elts`, use ambient dimension.
        if self.max_elts is None:
            self.max_elts = len(y0)
        data_type = np.float32 if y0.dtype == np.float32 else np.float64
        data_type = np.float32
        fvals = np.zeros(self.max_elts, dtype=data_type)
        resid = np.zeros(self.max_elts, dtype=data_type)
        # Matrix of bundle elements, stored in row-major format.
        bmtrx = np.zeros((self.max_elts, d), order="C", dtype=data_type)
        fvals[0] = fy0 - self.min_f + bmtrx[0, :] @ (y0 - y0)
        bmtrx[0, :] = self._gather_flat_grad().numpy()
        # Below, we slice bmtrx from 0:1 to force NumPy to interpret it as row vector.
        torch.nn.utils.vector_to_parameters(
            torch.from_numpy(y0 - linalg.lstsq(bmtrx[0:1, :], [fvals[0]], rcond=None)[0]), self._params)
        # Compute function gap
        with torch.enable_grad():
            fy = closure().item()
        resid[0] = fy - self.min_f
        # Exit early if solution escaped ball.
        if np.linalg.norm(torch.nn.utils.parameters_to_vector(self._params).numpy() - y0) > self.tau * gap:
            torch.nn.utils.vector_to_parameters(torch.from_numpy(y0), self._params)
            return 1
        # Best solution and function value found so far.
        if fy < fy0:
            y_best = torch.nn.utils.parameters_to_vector(self._params).detach().clone().numpy()
            f_best = fy - self.min_f
        else:
            y_best = y0
            f_best = fy0 - self.min_f
        dy = torch.nn.utils.parameters_to_vector(self._params).numpy() - y0
        for bundle_idx in range(1, self.max_elts):
            bmtrx[bundle_idx, :] = self._gather_flat_grad().numpy()
            # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
            fvals[bundle_idx] = resid[bundle_idx - 1] + bmtrx[bundle_idx, :] @ (
                    y0 - torch.nn.utils.parameters_to_vector(self._params).numpy()
            )
            if self.linsys_solver is BundleLinearSystemSolver.NAIVE:
                dy = np.linalg.lstsq(
                    bmtrx[0: (bundle_idx + 1), :], fvals[0: (bundle_idx + 1)], rcond=None
                )[0]
            elif self.linsys_solver is BundleLinearSystemSolver.LSMR:
                dy, istop, itn = lsmr(
                    bmtrx[0: (bundle_idx + 1), :],
                    fvals[0: (bundle_idx + 1)],
                    atol=max(1e-15, 0.01 * gap),
                    btol=max(1e-15, 0.01 * gap),
                    conlim=0.0,
                    x0=dy,
                )[:3]
            else:
                raise ValueError(f"Unrecognized linear system solver {self.linsys_solver}!")
            # Update point and function gap.
            torch.nn.utils.vector_to_parameters(torch.from_numpy(y0 - dy), self._params)
            with torch.enable_grad():
                fy = closure().item()
            resid[bundle_idx] = fy - self.min_f
            # Terminate early if new point escaped ball around y₀.
            if np.linalg.norm(dy) > self.tau * gap:
                torch.nn.utils.vector_to_parameters(torch.from_numpy(y_best), self._params)
                return bundle_idx
            # Terminate early if function value decreased significantly.
            if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + self.eta_est)):
                # torch.nn.utils.vector_to_parameters(torch.from_numpy(y_best), self._params)
                # print("Early termination due to function value decrease: ", resid[bundle_idx], " gap ", gap)
                return bundle_idx
            # Otherwise, update best solution so far.
            if resid[bundle_idx] < f_best:
                y_best = torch.nn.utils.parameters_to_vector(self._params).detach().clone().numpy()
                f_best = resid[bundle_idx]
        torch.nn.utils.vector_to_parameters(torch.from_numpy(y_best), self._params)
        return self.max_elts

    @torch.no_grad()
    def step(self, closure):
        """Performs a single polyak bundle step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        closure = torch.enable_grad()(closure)
        self.loss = (closure().item())
        bundle_index = self.build_bundle_torch_param_groups(closure)
        self.loss = (closure().item())

        return self.loss, bundle_index
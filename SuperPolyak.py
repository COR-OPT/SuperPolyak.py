import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters
import numpy as np
import scipy.linalg as sp_linalg
from scipy.linalg import LinAlgError
from scipy.sparse.linalg import lsmr

from enum import Enum
from typing import Callable, Iterable


class BundleLinearSystemSolver(Enum):
    """Contains the available linear system solvers for PolyakBundle.

    NAIVE: The default numpy least-squares solver.
    LSMR: The SciPy implementation of the LSMR algorithm.
    QR: An incremental QR-based approach using SciPy's `qr_insert`.
    """

    NAIVE = 1
    LSMR = 2
    QR = 3


class SuperPolyak(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        tau: float = np.inf,
        min_f: float = 0.0,
        eta_est: float = 0.1,
        max_elts: int = 1,
        linsys_solver: BundleLinearSystemSolver = BundleLinearSystemSolver.NAIVE,
    ):
        """Initialize the SuperPolyak optimizer.

        Args:
            params (iterable): iterable of parameters to optimize.
            tau (float, optional): a trust region parameter controlling
                how far iterates can travel before stopping (default: np.inf).
            min_f (float, optional): A lower bound on the minimal function
                value (default: 0.0).
            eta_est (float, optional): An estimate of the modulus of
                (b)-regularity (default: 0.1).
            max_elts (int, optional): The number of bundle elements to use (default: 1).
                If the default value is used, every step reduces to one step
                of the Polyak subgradient method.
        """
        super(SuperPolyak, self).__init__(params, dict())
        if len(self.param_groups) != 1:
            raise ValueError(
                "SuperPolyak doesn't support per-parameter options "
                "(parameter groups)"
            )
        self.loss = np.inf
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self.tau = tau
        self.min_f = min_f
        self.eta_est = eta_est
        self.max_elts = max_elts
        self.linsys_solver = linsys_solver

    def _gather_flat_grad(self) -> np.ndarray:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0).numpy()

    @torch.no_grad()
    def build_bundle_qr(self, closure: Callable[..., torch.Tensor]) -> int:
        y0 = parameters_to_vector(self._params).detach().clone().numpy()
        with torch.enable_grad():
            fy0 = closure().item()
        gap = fy0 - self.min_f
        data_type = y0.dtype
        fvals = np.zeros(self.max_elts, dtype=data_type)
        resid = np.zeros(self.max_elts, dtype=data_type)
        fvals[0] = fy0 - self.min_f
        bvect = self._gather_flat_grad()
        # First bundle point is the standard Polyak step.
        vector_to_parameters(
            torch.from_numpy(
                y0 - fvals[0] * bvect / (np.linalg.norm(bvect) ** 2)
            ),
            self._params,
        )
        # Initialize Q, R from bvect.
        Q, R = sp_linalg.qr(bvect[:, np.newaxis], mode="economic", pivoting=False)
        # Compute function gap
        with torch.enable_grad():
            fy = closure().item()
        resid[0] = fy - self.min_f
        # Exit early if solution escaped ball.
        if (
            np.linalg.norm(parameters_to_vector(self._params).numpy() - y0)
            > self.tau * gap
        ):
            vector_to_parameters(torch.from_numpy(y0), self._params)
            return 1
        # Best solution and function value found so far.
        if fy < fy0:
            y_best = parameters_to_vector(self._params).detach().clone().numpy()
            f_best = fy - self.min_f
        else:
            y_best = y0
            f_best = fy0 - self.min_f
        dy = parameters_to_vector(self._params).numpy() - y0
        for bundle_idx in range(1, self.max_elts):
            bvect = self._gather_flat_grad()
            # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
            fvals[bundle_idx] = resid[bundle_idx - 1] + bvect @ (
                y0 - parameters_to_vector(self._params).numpy()
            )
            # Update the Q and R factors.
            try:
                Q, R = sp_linalg.qr_insert(
                    Q,
                    R,
                    bvect,
                    bundle_idx,
                    "col"
                )
                dy = Q @ (sp_linalg.solve_triangular(R, fvals[:(bundle_idx + 1)], trans=1))
            except LinAlgError:
                dy = np.linalg.lstsq(
                    np.hstack((Q @ R, bvect[:, np.newaxis])).T,
                    fvals[0 : (bundle_idx + 1)],
                    rcond=None
                )[0]
                print("Bundle matrix is degenerate - terminating.")
                if np.linalg.norm(dy) > self.tau * gap:
                    vector_to_parameters(torch.from_numpy(y_best), self._params)
                else:
                    vector_to_parameters(torch.from_numpy(y0 - dy), self._params)
                return bundle_idx
            # Update point and function gap.
            vector_to_parameters(torch.from_numpy(y0 - dy), self._params)
            with torch.enable_grad():
                fy = closure().item()
            resid[bundle_idx] = fy - self.min_f
            # Terminate early if new point escaped ball around y₀.
            if np.linalg.norm(dy) > self.tau * gap:
                vector_to_parameters(torch.from_numpy(y_best), self._params)
                return bundle_idx
            # Terminate early if function value decreased significantly.
            if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + self.eta_est)):
                return bundle_idx
            # Otherwise, update best solution so far.
            if resid[bundle_idx] < f_best:
                y_best = parameters_to_vector(self._params).detach().clone().numpy()
                f_best = resid[bundle_idx]
        vector_to_parameters(torch.from_numpy(y_best), self._params)
        return self.max_elts


    @torch.no_grad()
    def build_bundle(self, closure: Callable[..., torch.Tensor]) -> int:
        y0 = parameters_to_vector(self._params).detach().clone().numpy()
        d = len(y0)
        with torch.enable_grad():
            fy0 = closure().item()
        gap = fy0 - self.min_f
        data_type = np.float32 if y0.dtype == np.float32 else np.float64
        data_type = np.float32
        fvals = np.zeros(self.max_elts, dtype=data_type)
        resid = np.zeros(self.max_elts, dtype=data_type)
        # Matrix of bundle elements, stored in row-major format.
        bmtrx = np.zeros((self.max_elts, d), order="C", dtype=data_type)
        fvals[0] = fy0 - self.min_f + bmtrx[0, :] @ (y0 - y0)
        bmtrx[0, :] = self._gather_flat_grad()
        # Below, we slice bmtrx from 0:1 to force NumPy to interpret it as row vector.
        vector_to_parameters(
            torch.from_numpy(
                y0 - np.linalg.lstsq(bmtrx[0:1, :], [fvals[0]], rcond=None)[0]
            ),
            self._params,
        )
        # Compute function gap
        with torch.enable_grad():
            fy = closure().item()
        resid[0] = fy - self.min_f
        # Exit early if solution escaped ball.
        if (
            np.linalg.norm(parameters_to_vector(self._params).numpy() - y0)
            > self.tau * gap
        ):
            vector_to_parameters(torch.from_numpy(y0), self._params)
            return 1
        # Best solution and function value found so far.
        if fy < fy0:
            y_best = parameters_to_vector(self._params).detach().clone().numpy()
            f_best = fy - self.min_f
        else:
            y_best = y0
            f_best = fy0 - self.min_f
        dy = parameters_to_vector(self._params).numpy() - y0
        for bundle_idx in range(1, self.max_elts):
            bmtrx[bundle_idx, :] = self._gather_flat_grad()
            # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
            fvals[bundle_idx] = resid[bundle_idx - 1] + bmtrx[bundle_idx, :] @ (
                y0 - parameters_to_vector(self._params).numpy()
            )
            if self.linsys_solver is BundleLinearSystemSolver.NAIVE:
                dy = np.linalg.lstsq(
                    bmtrx[0 : (bundle_idx + 1), :],
                    fvals[0 : (bundle_idx + 1)],
                    rcond=None,
                )[0]
            elif self.linsys_solver is BundleLinearSystemSolver.LSMR:
                # TODO: Tune tolerances for ATOL, BTOL.
                dy = lsmr(
                    bmtrx[0 : (bundle_idx + 1), :],
                    fvals[0 : (bundle_idx + 1)],
                    atol=max(1e-15, 0.01 * gap),
                    btol=max(1e-15, 0.01 * gap),
                    conlim=0.0,
                    x0=dy,
                )[0]
            else:
                raise ValueError(
                    f"Unrecognized linear system solver {self.linsys_solver}!"
                )
            # Update point and function gap.
            vector_to_parameters(torch.from_numpy(y0 - dy), self._params)
            with torch.enable_grad():
                fy = closure().item()
            resid[bundle_idx] = fy - self.min_f
            # Terminate early if new point escaped ball around y₀.
            if np.linalg.norm(dy) > self.tau * gap:
                vector_to_parameters(torch.from_numpy(y_best), self._params)
                return bundle_idx
            # Terminate early if function value decreased significantly.
            if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + self.eta_est)):
                return bundle_idx
            # Otherwise, update best solution so far.
            if resid[bundle_idx] < f_best:
                y_best = parameters_to_vector(self._params).detach().clone().numpy()
                f_best = resid[bundle_idx]
        vector_to_parameters(torch.from_numpy(y_best), self._params)
        return self.max_elts

    @torch.no_grad()
    def step(self, closure):
        """Perform a single polyak bundle step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1
        if self.linsys_solver is BundleLinearSystemSolver.QR:
            bundle_index = self.build_bundle_qr(closure)
        else:
            bundle_index = self.build_bundle(closure)
        with torch.enable_grad():
            self.loss = closure().item()

        return self.loss, bundle_index

from enum import Enum
from typing import Callable, Optional, Iterable

import numpy as np
import numpy.linalg as linalg
from scipy.sparse.linalg import lsmr
import SuperPolyak

import torch
import torch.optim.optimizer as optimizer


def superpolyak_coupled_with_fallback(closure_superpolyak : Callable,
                                      closure_fallback : Callable,
                                      superpolyak_optimizer : SuperPolyak.SuperPolyak,
                                      fallback_optimizer : optimizer,
                                      mult_factor : float = .5,
                                      max_inner_iter : int = np.inf,
                                      tol : float = 1e-16,
                                      max_outer_iter : int = np.inf,
                                      verbose : bool = False):

    loss = closure_superpolyak().item()
    loss_list = [loss]
    oracle_calls = [0]
    for outer_iter in range(max_outer_iter):
        loss = closure_superpolyak().item()
        if loss < tol:
            if verbose:
                print("Tolerance reached!",
                      "Current oracle evaluation: ", oracle_calls[-1],
                      "Loss = ", loss_superpolyak_step)
            return oracle_calls, loss_list
        loss_superpolyak_step, bundle_idx = superpolyak_optimizer.step(closure_superpolyak)
        oracle_calls.append(oracle_calls[-1] + bundle_idx)
        loss_list.append(loss_superpolyak_step)
        if loss_superpolyak_step < mult_factor*loss:
            if verbose:
                print("SuperPolyak step accepted!",
                      "Current oracle evaluation: ", oracle_calls[-1],
                      ",Loss = ", loss_superpolyak_step,
                      ",Bundle index = ", bundle_idx)
            continue
        else:
            param_best = _clone_param(superpolyak_optimizer._params)
            loss_best = min([loss, loss_superpolyak_step])
            for k in range(max_inner_iter):
                fallback_optimizer.step(closure_fallback)
                fallback_loss = closure_superpolyak().item()
                if fallback_loss < loss_best:
                    _set_param(superpolyak_optimizer._params, param_best)
                    loss_best = closure_superpolyak().item()
                oracle_calls.append(oracle_calls[-1] + 1)
                loss_list.append(loss_best)
                if fallback_loss < mult_factor * loss:
                    break
                if fallback_loss < tol:
                    if verbose:
                        print("Tolerance reached",
                              "Current oracle evaluation: ", oracle_calls[-1],
                              "Loss = ", loss_best)
                    return oracle_calls, loss_list
            if verbose:
                print("Fallback step!",
                      "Current oracle evaluation: ", oracle_calls[-1],
                      ",Loss = ", loss_best)
            _set_param(superpolyak_optimizer._params, param_best)
    return oracle_calls, loss_list


def _set_param(param_original, params_new):
    for p, pdata in zip(param_original, params_new):
        p.data = pdata.data.clone()

def _clone_param(param):
    return [p.detach().clone(memory_format=torch.contiguous_format) for p in param]

def generate_sparse_vector(d: int, k: int) -> torch.Tensor:
    """Generate a random sparse vector with unit norm.

    Args:
        d (int): The dimension of the vector.
        k (int): The number of nonzero elements.

    Returns:
        torch.Tensor: A float64 tensor.
    """
    x = np.zeros(d)
    x[np.random.choice(range(d), size=k, replace=False)] = np.random.randn(k)
    return torch.tensor(x / np.linalg.norm(x), dtype=torch.double)

class BundleLinearSystemSolver(Enum):
    """Contains the available linear system solvers for PolyakBundle."""

    NAIVE = 1
    LSMR = 2

def build_bundle(
    f: Callable[[np.ndarray], np.number],
    gradf: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    tau: float,
    min_f: float,
    eta_est: float,
    max_elts: Optional[int],
    linsys_solver: BundleLinearSystemSolver = BundleLinearSystemSolver.NAIVE,
):
    """Build a Polyak bundle given a loss function and an initial point.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.number]
        A callable implementing the loss function.
    gradf : Callable[[np.ndarray], np.ndarray]
        A callable implementing the gradient of the loss function.
    y0 : np.ndarray
        The center of the bundle.
    tau : float
        The trust region parameter.
    eta_est : float
        An estimate of the b-regularity exponent.
    max_elts: int, optional
        The maximal number of elements in the bundle.
    linsys_solver: BundleLinearSystemSolver (default: NAIVE)
        The linear system solver to use.
    """
    d = len(y0)
    y = y0[:]
    # If no value was given for `max_elts`, use ambient dimension.
    if max_elts is None:
        max_elts = len(y0)
    fvals = np.zeros(max_elts)
    resid = np.zeros(max_elts)
    # Matrix of bundle elements, stored in row-major format.
    bmtrx = np.zeros((max_elts, d), order="C")
    bmtrx[0, :] = gradf(y)
    fvals[0] = f(y) - min_f + bmtrx[0, :] @ (y0 - y)
    # Below, we slice bmtrx from 0:1 to force NumPy to interpret it as row vector.
    y = y0 - linalg.lstsq(bmtrx[0:1, :], [fvals[0]])[0]
    resid[0] = f(y) - min_f
    gap = f(y0) - min_f
    # Exit early if solution escaped ball.
    if np.linalg.norm(y - y0) > tau * gap:
        return y0, 1
    # Best solution and function value found so far.
    y_best = y[:]
    f_best = resid[0]
    dy = y - y0
    for bundle_idx in range(1, max_elts):
        bmtrx[bundle_idx, :] = gradf(y)
        # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
        fvals[bundle_idx] = resid[bundle_idx - 1] + bmtrx[bundle_idx, :] @ (y0 - y)
        if linsys_solver is BundleLinearSystemSolver.NAIVE:
            dy = linalg.lstsq(
                bmtrx[0 : (bundle_idx + 1), :], fvals[0 : (bundle_idx + 1)], rcond=None
            )[0]
        elif linsys_solver is BundleLinearSystemSolver.LSMR:
            dy, istop, itn = lsmr(
                bmtrx[0 : (bundle_idx + 1), :],
                fvals[0 : (bundle_idx + 1)],
                atol=max(1e-15, gap),
                btol=max(1e-15, gap),
                conlim=0.0,
                x0=dy,
            )[:3]
            # TODO: Warn according to status in `istop`.
            # TODO: Keep track of the number of iterations `itn`.
        else:
            raise ValueError(f"Unrecognized linear system solver {linsys_solver}!")
        # Update point and function gap.
        y = y0 - dy
        resid[bundle_idx] = f(y) - min_f
        # Terminate early if new point escaped ball around y₀.
        if np.linalg.norm(dy) > tau * gap:
            return y_best, bundle_idx
        # Terminate early if function value decreased significantly.
        if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + eta_est)):
            return y, bundle_idx
        # Otherwise, update best solution so far.
        if resid[bundle_idx] < f_best:
            y_best = y
            f_best = resid[bundle_idx]
    return y_best, max_elts


@torch.no_grad()
def build_bundle_torch(
    closure: Callable,
    x: torch.Tensor,
    tau: float,
    min_f: float,
    eta_est: float,
    max_elts: Optional[int],
    linsys_solver: BundleLinearSystemSolver = BundleLinearSystemSolver.NAIVE,
):
    """Build a Polyak bundle given a loss function and an initial point.

    Parameters
    ----------
    closure: Callable
        A closure that evaluates the objective function.
    x : torch.Tensor
        The center point of the bundle.
    tau : float
        The trust region parameter.
    eta_est : float
        An estimate of the b-regularity exponent.
    max_elts: int, optional
        The maximal number of elements in the bundle.
    linsys_solver: BundleLinearSystemSolver (default: NAIVE)
        The linear system solver to use.
    """
    d = len(x)
    y0 = x.detach().clone().numpy()
    with torch.enable_grad():
        fy0 = closure().item()
    gap = fy0 - min_f
    # If no value was given for `max_elts`, use ambient dimension.
    if max_elts is None:
        max_elts = len(y0)
    fvals = np.zeros(max_elts)
    resid = np.zeros(max_elts)
    # Matrix of bundle elements, stored in row-major format.
    bmtrx = np.zeros((max_elts, d), order="C")
    fvals[0] = fy0 - min_f + bmtrx[0, :] @ (y0 - y0)
    bmtrx[0, :] = x.grad.numpy()
    # Below, we slice bmtrx from 0:1 to force NumPy to interpret it as row vector.
    x.add_(-torch.from_numpy(linalg.lstsq(bmtrx[0:1, :], [fvals[0]], rcond=None)[0]))
    # Compute function gap
    with torch.enable_grad():
        fy = closure().item()
    resid[0] = fy - min_f
    # Exit early if solution escaped ball.
    if np.linalg.norm(x.numpy() - y0) > tau * gap:
        return torch.from_numpy(y0), 1
    # Best solution and function value found so far.
    y_best = x.detach().clone().numpy()
    f_best = resid[0]
    for bundle_idx in range(1, max_elts):
        bmtrx[bundle_idx, :] = x.grad.numpy()
        # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
        fvals[bundle_idx] = resid[bundle_idx - 1] + bmtrx[bundle_idx, :] @ (
            y0 - x.numpy()
        )
        if linsys_solver is BundleLinearSystemSolver.NAIVE:
            dy = np.linalg.lstsq(
                bmtrx[0 : (bundle_idx + 1), :], fvals[0 : (bundle_idx + 1)], rcond=None
            )[0]
        elif linsys_solver is BundleLinearSystemSolver.LSMR:
            dy, istop, itn = lsmr(
                bmtrx[0 : (bundle_idx + 1), :],
                fvals[0 : (bundle_idx + 1)],
                atol=max(1e-15, gap),
                btol=max(1e-15, gap),
                conlim=0.0,
                x0=dy,
            )[:3]
        else:
            raise ValueError(f"Unrecognized linear system solver {linsys_solver}!")
        # Update point and function gap.
        x.copy_(torch.from_numpy(y0 - dy))
        with torch.enable_grad():
            fy = closure().item()
        resid[bundle_idx] = fy - min_f
        # Terminate early if new point escaped ball around y₀.
        if np.linalg.norm(dy) > tau * gap:
            return y_best, bundle_idx
        # Terminate early if function value decreased significantly.
        if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + eta_est)):
            return x, bundle_idx
        # Otherwise, update best solution so far.
        if resid[bundle_idx] < f_best:
            y_best = x.detach().clone().numpy()
            f_best = resid[bundle_idx]
    return y_best, max_elts


def build_bundle(
    f: Callable[[np.ndarray], np.number],
    gradf: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    tau: float,
    min_f: float,
    eta_est: float,
    max_elts: Optional[int],
    linsys_solver: BundleLinearSystemSolver = BundleLinearSystemSolver.NAIVE,
):
    """Build a Polyak bundle given a loss function and an initial point.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.number]
        A callable implementing the loss function.
    gradf : Callable[[np.ndarray], np.ndarray]
        A callable implementing the gradient of the loss function.
    y0 : np.ndarray
        The center of the bundle.
    tau : float
        The trust region parameter.
    eta_est : float
        An estimate of the b-regularity exponent.
    max_elts: int, optional
        The maximal number of elements in the bundle.
    linsys_solver: BundleLinearSystemSolver (default: NAIVE)
        The linear system solver to use.
    """
    d = len(y0)
    y = y0[:]
    # If no value was given for `max_elts`, use ambient dimension.
    if max_elts is None:
        max_elts = len(y0)
    fvals = np.zeros(max_elts)
    resid = np.zeros(max_elts)
    # Matrix of bundle elements, stored in row-major format.
    bmtrx = np.zeros((max_elts, d), order="C")
    bmtrx[0, :] = gradf(y)
    fvals[0] = f(y) - min_f + bmtrx[0, :] @ (y0 - y)
    # Below, we slice bmtrx from 0:1 to force NumPy to interpret it as row vector.
    y = y0 - linalg.lstsq(bmtrx[0:1, :], [fvals[0]])[0]
    resid[0] = f(y) - min_f
    gap = f(y0) - min_f
    # Exit early if solution escaped ball.
    if np.linalg.norm(y - y0) > tau * gap:
        return y0, 1
    # Best solution and function value found so far.
    y_best = y[:]
    f_best = resid[0]
    dy = y - y0
    for bundle_idx in range(1, max_elts):
        bmtrx[bundle_idx, :] = gradf(y)
        # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
        fvals[bundle_idx] = resid[bundle_idx - 1] + bmtrx[bundle_idx, :] @ (y0 - y)
        if linsys_solver is BundleLinearSystemSolver.NAIVE:
            dy = linalg.lstsq(
                bmtrx[0 : (bundle_idx + 1), :], fvals[0 : (bundle_idx + 1)], rcond=None
            )[0]
        elif linsys_solver is BundleLinearSystemSolver.LSMR:
            dy, istop, itn = lsmr(
                bmtrx[0 : (bundle_idx + 1), :],
                fvals[0 : (bundle_idx + 1)],
                atol=max(1e-15, gap),
                btol=max(1e-15, gap),
                conlim=0.0,
                x0=dy,
            )[:3]
            # TODO: Warn according to status in `istop`.
            # TODO: Keep track of the number of iterations `itn`.
        else:
            raise ValueError(f"Unrecognized linear system solver {linsys_solver}!")
        # Update point and function gap.
        y = y0 - dy
        resid[bundle_idx] = f(y) - min_f
        # Terminate early if new point escaped ball around y₀.
        if np.linalg.norm(dy) > tau * gap:
            return y_best, bundle_idx
        # Terminate early if function value decreased significantly.
        if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + eta_est)):
            return y, bundle_idx
        # Otherwise, update best solution so far.
        if resid[bundle_idx] < f_best:
            y_best = y
            f_best = resid[bundle_idx]
    return y_best, max_elts


def _gather_flat_grad(params : Iterable[torch.Tensor]):
    views = []
    for p in params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        elif p.grad.is_sparse:
            view = p.grad.to_dense().view(-1)
        else:
            view = p.grad.view(-1)
        views.append(view)
    return torch.cat(views, 0)

@torch.no_grad()
def build_bundle_torch_param_groups(
    closure: Callable,
    params: Iterable[torch.Tensor],
    tau: float,
    min_f: float,
    eta_est: float,
    max_elts: Optional[int],
    linsys_solver: BundleLinearSystemSolver = BundleLinearSystemSolver.NAIVE,
):
    """Build a Polyak bundle given a loss function and an initial point.

    Parameters
    ----------
    closure: Callable
        A closure that evaluates the objective function.
    params : Iterable[torch.Tensor]
        The center point of the bundle.
    tau : float
        The trust region parameter.
    eta_est : float
        An estimate of the b-regularity exponent.
    max_elts: int, optional
        The maximal number of elements in the bundle.
    linsys_solver: BundleLinearSystemSolver (default: NAIVE)
        The linear system solver to use.
    """
    params = list(params)
    y0 = torch.nn.utils.parameters_to_vector(params).detach().clone().numpy()
    d = len(y0)
    with torch.enable_grad():
        fy0 = closure().item()
    gap = fy0 - min_f
    # If no value was given for `max_elts`, use ambient dimension.
    if max_elts is None:
        max_elts = len(y0)
    data_type = np.float32 if y0.dtype == np.float32 else np.float64
    data_type = np.float32
    fvals = np.zeros(max_elts,dtype=data_type)
    resid = np.zeros(max_elts,dtype=data_type)
    # Matrix of bundle elements, stored in row-major format.
    bmtrx = np.zeros((max_elts, d), order="C",dtype=data_type)
    fvals[0] = fy0 - min_f + bmtrx[0, :] @ (y0 - y0)
    bmtrx[0, :] = _gather_flat_grad(params).numpy()
    # Below, we slice bmtrx from 0:1 to force NumPy to interpret it as row vector.
    torch.nn.utils.vector_to_parameters(torch.from_numpy(y0 - linalg.lstsq(bmtrx[0:1, :], [fvals[0]], rcond=None)[0]), params)
    # Compute function gap
    with torch.enable_grad():
        fy = closure().item()
    resid[0] = fy - min_f
    # Exit early if solution escaped ball.
    if np.linalg.norm(torch.nn.utils.parameters_to_vector(params).numpy() - y0) > tau * gap:
        torch.nn.utils.vector_to_parameters(torch.from_numpy(y0), params)
        return params, 1
    # Best solution and function value found so far.
    if fy < fy0:
        y_best = torch.nn.utils.parameters_to_vector(params).detach().clone().numpy()
        f_best = fy
    else:
        y_best = y0
        f_best = fy0
    dy = torch.nn.utils.parameters_to_vector(params).numpy() - y0
    for bundle_idx in range(1, max_elts):
        bmtrx[bundle_idx, :] = _gather_flat_grad(params).numpy()
        # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
        fvals[bundle_idx] = resid[bundle_idx - 1] + bmtrx[bundle_idx, :] @ (
            y0 - torch.nn.utils.parameters_to_vector(params).numpy()
        )
        if linsys_solver is BundleLinearSystemSolver.NAIVE:
            dy = np.linalg.lstsq(
                bmtrx[0 : (bundle_idx + 1), :], fvals[0 : (bundle_idx + 1)], rcond=None
            )[0]
        elif linsys_solver is BundleLinearSystemSolver.LSMR:
            dy, istop, itn = lsmr(
                bmtrx[0 : (bundle_idx + 1), :],
                fvals[0 : (bundle_idx + 1)],
                atol=max(1e-15, 0.01 * gap),
                btol=max(1e-15, 0.01 * gap),
                conlim=0.0,
                x0=dy,
            )[:3]
        else:
            raise ValueError(f"Unrecognized linear system solver {linsys_solver}!")
        # Update point and function gap.
        torch.nn.utils.vector_to_parameters(torch.from_numpy(y0 - dy),params)
        with torch.enable_grad():
            fy = closure().item()
        resid[bundle_idx] = fy - min_f
        # Terminate early if new point escaped ball around y₀.
        if np.linalg.norm(dy) > tau * gap:
            torch.nn.utils.vector_to_parameters(torch.from_numpy(y_best), params)
            return params, bundle_idx
        # Terminate early if function value decreased significantly.
        if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + eta_est)):
            torch.nn.utils.vector_to_parameters(torch.from_numpy(y_best), params)
            return params, bundle_idx
        # Otherwise, update best solution so far.
        if resid[bundle_idx] < f_best:
            y_best = torch.nn.utils.parameters_to_vector(params).detach().clone().numpy()
            f_best = resid[bundle_idx]
    torch.nn.utils.vector_to_parameters(torch.from_numpy(y_best), params)
    return params, max_elts



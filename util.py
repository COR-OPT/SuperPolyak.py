from enum import Enum
from typing import Callable, Optional

import numpy as np
import numpy.linalg as linalg

import scipy.sparse.linalg as sps_linalg

class BundleLinearSystemSolver(Enum):
    NAIVE = 1
    LSMR = 2

def build_bundle(f: Callable[[np.ndarray], np.number],
                 gradf: Callable[[np.ndarray], np.ndarray],
                 y0: np.ndarray,
                 tau: float,
                 min_f: float,
                 eta_est: float,
                 max_elts: Optional[int],
                 bundle_linear_system_solver: BundleLinearSystemSolver = BundleLinearSystemSolver.NAIVE):
    """Build a Polyak bundle for a given loss function and starting point.

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
    bundle_linear_system_solver: BundleLinearSystemSolver
        The method used to solve each linear system.

    Returns
    -------
    np.ndarray
        The best iterate from the Polyak bundle.
    int
        The number of bundle steps taken.
    """
    d = len(y0)
    y = y0[:]
    # If no value was given for `max_elts`, use ambient dimension.
    if max_elts is None:
        max_elts = len(y0)
    fvals = np.zeros(max_elts)
    resid = np.zeros(max_elts)
    # Matrix of bundle elements, stored in row-major format.
    bmtrx = np.zeros((max_elts, d), order='C')
    bmtrx[0, :] = gradf(y)
    fvals[0] = f(y) - min_f + bmtrx[0, :] @ (y0 - y)
    # Below, we slice bmtrx from 0:1 to force NumPy to interpret it as row vector.
    y = y0 - linalg.lstsq(bmtrx[0:1, :], [fvals[0]], rcond=None)[0]
    resid[0] = f(y) - min_f
    gap = f(y0) - min_f
    # Exit early if solution escaped ball.
    if np.linalg.norm(y - y0) > tau * gap:
        return y0, 1
    # Best solution and function value found so far.
    y_best = y[:]
    f_best = resid[0]
    # Difference for warm starting
    dy = y - y0
    for bundle_idx in range(1, max_elts):
        bmtrx[bundle_idx, :] = gradf(y)
        # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
        fvals[bundle_idx] = resid[bundle_idx-1] + bmtrx[bundle_idx, :] @ (y0 - y)
        if bundle_linear_system_solver is BundleLinearSystemSolver.LSMR:
            # pylint: disable=W0612
            dy, istop, itn = sps_linalg.lsmr(
                bmtrx[0:(bundle_idx + 1), :],
                fvals[0:(bundle_idx + 1)],
                atol=max(1e-15, gap),
                btol=max(1e-15, gap),
                conlim=0.0,
                x0=dy[:])[:3]
            # TODO: Warn according to status in `istop`.
            # TODO: Keep track of the number of iterations `itn`.
        elif bundle_linear_system_solver is BundleLinearSystemSolver.NAIVE:
            dy = linalg.lstsq(
                bmtrx[0:(bundle_idx + 1), :],
                fvals[0:(bundle_idx + 1)],
                rcond=None)[0]
        else:
            raise ValueError("Unknown value for `bundle_method`!")
        # Update point and function gap.
        y = y0 - dy
        resid[bundle_idx] = f(y) - min_f
        # Terminate early if new point escaped ball around y₀.
        if (np.linalg.norm(dy) > tau * gap):
            return y_best, bundle_idx
        # Terminate early if function value decreased significantly.
        if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + eta_est)):
            return y, bundle_idx
        # Otherwise, update best solution so far.
        if (resid[bundle_idx] < f_best):
            y_best = y
            f_best = resid[bundle_idx]
    return y_best, max_elts

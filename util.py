import numpy as np
import numpy.linalg as linalg

from typing import Callable, Optional

def build_bundle_naive(f: Callable[[np.ndarray], np.number], gradf: Callable[[np.ndarray], np.ndarray], y0: np.ndarray, tau: float, min_f: float, eta_est: float, max_elts: Optional[int]):
    """Build a Polyak bundle by solving every linear system naively.

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
    y = y0 - linalg.lstsq(bmtrx[0:1, :], [fvals[0]])
    resid[0] = f(y) - min_f
    gap = f(y0) - min_f
    # Exit early if solution escaped ball.
    if np.linalg.norm(y - y0) > tau * gap:
        return y0, 1
    # # Best solution and function value found so far.
    # y_best = y[:]
    y_best = y[:]
    # f_best = resid[1]
    f_best = resid[0]
    for bundle_idx in range(1, max_elts):
        bmtrx[bundle_idx, :] = gradf(y)
        # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
        fvals[bundle_idx] = resid[bundle_idx-1] + bmtrx[bundle_idx, :] @ (y0 - y)
        # lsqr!(
        #   dy,
        #   At',
        #   view(fvals, 1:bundle_idx),
        #   maxiter = bundle_idx,
        #   atol = max(Δ, 1e-15),
        #   btol = max(Δ, 1e-15),
        #   conlim = 0.0,
        # )
        dy, _, _, _ = linalg.lstsq(
            bmtrx[0:(bundle_idx + 1), :],
            fvals[0:(bundle_idx + 1)],
            rcond=None)
        # Update point and function gap.
        y = y0 - dy
        # resid[bundle_idx] = f(y) - min_f
        resid[bundle_idx] = f(y) - min_f
        # # Terminate early if new point escaped ball around y₀.
        if (np.linalg.norm(dy) > tau * gap):
            return y_best, bundle_idx
        # # Terminate early if function value decreased significantly.
        if (gap < 0.5) and (resid[bundle_idx] < gap ** (1 + eta_est)):
            return y, bundle_idx
        # # Otherwise, update best solution so far.
        if (resid[bundle_idx] < f_best):
            y_best = y
            f_best = resid[bundle_idx]
    return y_best, d

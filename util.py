from enum import Enum
from typing import Callable, Optional, Iterable

import numpy as np
import numpy.linalg as linalg
from scipy.sparse.linalg import lsmr
import SuperPolyak

import torch
from torch.optim import Optimizer

def _set_param(param_original, params_new):
    for p, pdata in zip(param_original, params_new):
        p.data = pdata.data.clone()


def _clone_param(param):
    return [p.detach().clone(memory_format=torch.contiguous_format) for p in param]


def _tol_reached_msg(oracle_calls: int, loss: float):
    return (
        "Tolerance reached! Current oracle evals: {0}, Loss = {1}"
    ).format(oracle_calls, loss)


def superpolyak_coupled_with_fallback(superpolyak_closure: Callable,
                                      fallback_closure: Callable,
                                      superpolyak_optimizer: SuperPolyak.SuperPolyak,
                                      fallback_optimizer: Optimizer,
                                      max_inner_iter: int,
                                      max_outer_iter: int,
                                      mult_factor: float = .5,
                                      tol: float = 1e-16,
                                      verbose: bool = False):

    loss = superpolyak_closure().item()
    loss_list = [loss]
    oracle_calls = [0]
    for k_outer in range(max_outer_iter):
        loss = loss_list[-1]
        if loss < tol:
            if verbose:
                print(_tol_reached_msg(oracle_calls[-1], loss))
            return oracle_calls, loss_list
        loss_superpolyak_step, bundle_idx = superpolyak_optimizer.step(superpolyak_closure)
        loss_list.append(loss_superpolyak_step)
        oracle_calls.append(oracle_calls[-1] + bundle_idx)
        if loss_superpolyak_step < mult_factor * loss:
            if verbose:
                print("SuperPolyak step accepted!",
                      "Current oracle evaluations: ", oracle_calls[-1],
                      ", Loss = ", loss_superpolyak_step,
                      ", Bundle index = ", bundle_idx)
        else:
            param_best = _clone_param(superpolyak_optimizer._params)
            loss_best = min([loss, loss_superpolyak_step])
            for k_inner in range(max_inner_iter):
                fallback_optimizer.step(fallback_closure)
                fallback_loss = superpolyak_closure().item()
                if fallback_loss < loss_best:
                    _set_param(superpolyak_optimizer._params, param_best)
                    loss_best = superpolyak_closure().item()
                oracle_calls.append(oracle_calls[-1] + 1)
                loss_list.append(loss_best)
                if fallback_loss < mult_factor * loss:
                    break
                if fallback_loss < tol:
                    if verbose:
                        print(_tol_reached_msg(oracle_calls[-1], loss_best))
                    return oracle_calls, loss_list
            if verbose:
                print("Fallback step!",
                      "Current oracle evaluations: ", oracle_calls[-1],
                      ", Loss = ", loss_best)
            _set_param(superpolyak_optimizer._params, param_best)
    return oracle_calls, loss_list


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

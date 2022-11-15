from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .superpolyak import SuperPolyak


def _set_param(param_original, params_new):
    for p, pdata in zip(param_original, params_new):
        p.data = pdata.data.clone()


def _clone_param(param):
    return [p.detach().clone(memory_format=torch.contiguous_format) for p in param]


def _tol_reached_msg(oracle_calls: int, loss: float):
    return ("Tolerance reached! Current oracle evals: {0}, Loss = {1}").format(
        oracle_calls, loss
    )


def superpolyak_coupled_with_fallback(
    superpolyak_closure: Callable,
    fallback_closure: Callable,
    superpolyak_optimizer: SuperPolyak,
    fallback_optimizer: Optimizer,
    max_inner_iter: int,
    max_outer_iter: int,
    mult_factor: float = 0.5,
    tol: float = 1e-16,
    verbose: bool = False,
    metric_to_print: Optional[Callable] = None,
):
    """Couple the superpolyak optimizer with an arbitrary fallback method.

    Args:
        superpolyak_closure (callable): The closure to use with superpolyak.
        fallback_closure (callable): The closure to use with the fallback method.
        superpolyak_optimizer (SuperPolyak): An instance of the superpolyak
            optimizer.
        fallback_optimizer (torch.optim.Optimizer): An instance of an optimizer
            implementing the fallback method.
        max_inner_iter (int): The number of fallback steps to run after a step
            of superpolyak that did not sufficiently reduce the loss.
        max_outer_iter (int): The number of attempted superpolyak steps.
        mult_factor (float): A factor in (0, 1) determining sufficient decrease
            in loss (default: 0.5).
        tol (float): Numerical tolerance; if the loss falls below this number,
            the method terminates (default: 1e-16).
        verbose (bool): If set, logs information about the method's progress
            across each iteration (default: False).
        metric_to_print (callable): A function returning a metric of interest
            that is logged if `verbose == True` (default: None).

    Returns:
        oracle_calls (Sequence[int]): A list containing the cumulative calls to
            autodifferentiation oracles at each step of the algorithm.
        loss_list (Sequence[float]): A list containing the loss function value
            at each step of the algorithm.
    """

    def _get_metric_msg():
        return f", Metric = {metric_to_print()}" if metric_to_print is not None else ""

    loss = superpolyak_closure().item()
    loss_list = [loss]
    oracle_calls = [0]
    cloned_param = _clone_param(superpolyak_optimizer._params)
    for _ in range(max_outer_iter):
        loss = loss_list[-1]
        if loss < tol:
            if verbose:
                print(_tol_reached_msg(oracle_calls[-1], loss))
            return oracle_calls, loss_list
        loss_superpolyak_step, bundle_idx = superpolyak_optimizer.step(
            superpolyak_closure
        )
        loss_list.append(loss_superpolyak_step)
        oracle_calls.append(oracle_calls[-1] + bundle_idx)
        if loss_superpolyak_step < mult_factor * loss:
            if verbose:
                print(
                    "SuperPolyak step accepted!",
                    "Current oracle evaluations: ",
                    oracle_calls[-1],
                    ", Loss = ",
                    loss_superpolyak_step,
                    ", Bundle index = ",
                    bundle_idx,
                    _get_metric_msg(),
                )
        else:
            if loss_superpolyak_step >= loss:
                _set_param(superpolyak_optimizer._params, cloned_param)
                fallback_loss = loss
            else:
                fallback_loss = loss_superpolyak_step
            for k_inner in range(max_inner_iter):
                fallback_optimizer.step(fallback_closure)
                fallback_loss = superpolyak_closure().item()
                oracle_calls.append(oracle_calls[-1] + 1)
                loss_list.append(fallback_loss)
                if fallback_loss < mult_factor * loss:
                    break
                if fallback_loss < tol:
                    if verbose:
                        print(_tol_reached_msg(oracle_calls[-1], fallback_loss))
                    return oracle_calls, loss_list
            if verbose:
                print(
                    "Fallback step accepted!",
                    "Current oracle evaluations: ",
                    oracle_calls[-1],
                    ", Loss = ",
                    fallback_loss,
                    _get_metric_msg(),
                )
            if loss == fallback_loss:
                print("Algorithm is stuck, returning early.")
                return oracle_calls, loss_list
        _set_param(cloned_param, superpolyak_optimizer._params)
    return oracle_calls, loss_list

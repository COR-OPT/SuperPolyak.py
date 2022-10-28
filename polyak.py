import torch
from functools import reduce
from torch.optim.optimizer import Optimizer
import numpy as np

# Modified from https://github.com/pytorch/pytorch/blob/master/torch/optim/lbfgs.py

class Polyak(Optimizer):

    def __init__(self,
                 params,
                 lr=1,
                 history_size=100,
                 line_search_fn=None,
                 ls_params = None):

        defaults = dict(
            lr=lr,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super(Polyak, self).__init__(params, defaults)
        # self.num_oracle = []
        self.num_oracle_iter = 0
        if len(self.param_groups) != 1:
            raise ValueError("Polyak doesn't support per-parameter options "
                             "(parameter groups)")
        self.loss = np.inf
        self._params = self.param_groups[0]['params']
        self._numel_cache = None


        # add line-search tolerance
        if (ls_params is not None) and 'c1' in ls_params.keys():
            self.c1 = ls_params['c1']
        else:
            self.c1 = 0

        if (ls_params is not None) and 'c2' in ls_params.keys():
            self.c2 = ls_params['c2']
        else:
            self.c2 = 0.5

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

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

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        # loss = float(closure())
        loss = closure().item()
        flat_grad = self._gather_flat_grad()
        self._set_param(x)  # Reset the change induced by _add_grad
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        closure = torch.enable_grad()(closure)

        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        self.loss = (closure().item())
        d = self._gather_flat_grad()
        # Add to oracle count
        self.num_oracle_iter += 1
        nrmd2 = (d.vdot(d)).abs()

        # Check whether gradient is too small before dividing by itl.
        if nrmd2 > 1e-2000:
            t = -self.loss/nrmd2
        else:
            t = 0
        self._add_grad(t, d)
        self.f_eval_cur = 1

        state['n_iter'] += 1


        return self.loss
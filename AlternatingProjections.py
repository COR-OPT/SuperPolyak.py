import torch
from functools import reduce
from torch.optim.optimizer import Optimizer
import numpy as np

# Modified from https://github.com/pytorch/pytorch/blob/master/torch/optim/lbfgs.py

class AlternatingProjecions(Optimizer):

    def __init__(self,
                 params,
                 projs
                 ):
        super(AlternatingProjecions, self).__init__(params, dict())

        self.projs = projs
        if len(self.param_groups) != 1:
            raise ValueError("Alternating projections doesn't support per-parameter options "
                             "(parameter groups)")
        self._params = self.param_groups[0]['params']

    @torch.no_grad()
    def step(self, closure = None):
        """Performs a single alternating projections step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        for proj in self.projs:
            for p in self._params:
                p.data = proj(p.data)

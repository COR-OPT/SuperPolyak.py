# Adapted from https://gist.github.com/pmelchior/f371dda15df8e93776e570ffcb0d1494
from torch.optim.sgd import SGD

class ProxGradient(SGD):
    def __init__(self, params, proxs, lr, momentum=0, dampening=0, nesterov=False):
        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=0, nesterov=nesterov)
        super().__init__(params, **kwargs)

        if len(proxs) != len(self.param_groups):
            raise ValueError("Invalid length of argument proxs: {} instead of {}".format(len(proxs), len(self.param_groups)))

        for group, prox in zip(self.param_groups, list(proxs)):
            group.setdefault('prox', prox)

    def step(self, closure=None):
        # this performs a gradient step
        # optionally with momentum or nesterov acceleration
        super().step(closure=closure)

        for group in self.param_groups:
            prox = group['prox']
            lr = group['lr']

            # here we apply the proximal operator to each parameter in a group
            for p in group['params']:
                p.data = prox(p.data, lr)

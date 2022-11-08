import torch
from torch.nn.functional import normalize
import numpy as np

from util import generate_sparse_vector
from typing import Optional


class ReluRegressionProblem:
    def __init__(self, m, d):
        self.A = torch.randn(m, d, dtype=torch.double)
        self.x = normalize(torch.randn(d, dtype=torch.double), dim=-1)
        self.y = torch.max(self.A @ (self.x), torch.zeros(m))

    def loss(self):
        def f(z):
            return (1.0 / self.A.size()[0]) * torch.norm(
                torch.max(self.A @ (z), torch.zeros(self.A.size()[0])) - self.y, 1
            )

        return f

    def initializer(self, δ):
        return (
            self.x
            + δ * normalize(torch.randn(self.x.size(), dtype=torch.double), dim=-1)
        ).requires_grad_(True)


class MaxAffineRegressionProblem:
    def __init__(self, m, d, k):
        self.A = torch.randn(m, d, dtype=torch.double)
        self.βs = normalize(torch.randn(d, k, dtype=torch.double), dim=1)
        self.y = torch.max(self.A.mm(self.βs), dim=1)[0]

    def loss(self):
        def f(z):
            return (1.0 / self.A.size()[0]) * torch.linalg.norm(
                torch.max(self.A.mm(z.view(self.βs.size())), dim=1)[0] - self.y, 1
            )

        return f

    def initializer(self, δ):
        return (
            self.βs
            + δ * normalize(torch.randn(self.βs.size(), dtype=torch.double), dim=1)
        ).requires_grad_(True)


class CompressedSensingProblem:
    def __init__(self, m, d, k):
        self.A = torch.randn(m, d, dtype=torch.double)
        self.F = torch.qr(self.A, True)
        self.x = generate_sparse_vector(d, k)
        self.y = self.A @ (self.x)
        self.k = k

    def proj_sparse(self, x):
        x[torch.topk(torch.abs(x), self.k)[1][-1] + 1 :] = 0
        return x

    def dist_sparse(self, x):
        return torch.linalg.norm(
            x[torch.topk(torch.abs(x), self.x.size()[0] - self.k, largest=False)[1]]
        )

    def proj_range(self, x):
        return (
            x
            + torch.linalg.solve(
                self.F[0],
                self.y - self.A @ (x),
            )[0]
        )

    def dist_range(self, x):
        return torch.norm(torch.linalg.solve(self.F[0], self.y - self.A @ (x))[0])

    def loss(self):
        def f(z):
            return self.dist_sparse(z) + self.dist_range(z)

        return f

    def initializer(self, δ):
        return (
            self.x
            + δ * normalize(torch.randn(self.x.size(), dtype=torch.double), dim=-1)
        ).requires_grad_(True)


# The phase retrieval problem
class PhaseRetrievalProblem:
    def __init__(self, m, d):
        self.A = torch.randn(m, d)
        self.x = normalize(torch.randn(d), dim=-1)
        self.y = torch.abs(self.A @ self.x)

    def loss(self):
        m = self.A.size(0)
        return lambda z: (1.0 / m) * torch.norm(torch.abs(self.A @ z) - self.y, 1)

    def initializer(self, δ):
        return torch.tensor(
            self.x + δ * normalize(torch.randn(self.x.size()), dim=-1),
            requires_grad=True,
        )

    def loss_altproj(self):
        A = self.A
        y = self.y
        m = y.size(0)
        # Cache factorization of A.
        F = torch.qr(A)

        def f(z):
            z_comp = z[0:m] + z[m : 2 * m] * 1j
            return torch.norm(
                z_comp - A.mm(torch.linalg.solve(z_comp, F[0])[0])
            ) + torch.norm(z_comp - y * phase(z_comp))

        return f

    def alternating_projections_step(self):
        A = self.A
        y = self.y
        m = y.size(0)
        F = torch.qr(A)

        def f(z):
            phased = phase(
                A.mm(torch.linalg.solve(z[0:m] + z[m : 2 * m] * 1j, F[0])[0])
            )
            return torch.cat([y * phased.real, y * phased.imag])

        return f

    def initializer_altproj(self, δ):
        x = self.x + δ * normalize(torch.randn(self.x.size()), dim=-1)
        Ax = self.A @ (x)
        return torch.cat([Ax.real, Ax.imag]).requires_grad_(True)


# now we translate the quadratic sensing problem to pytorch


class QuadraticSensingProblem:
    def __init__(self, m, d, r):
        self.X = torch.tensor(
            np.linalg.qr(np.random.randn(d, r), mode="reduced")[0], dtype=torch.double
        )
        self.L = np.sqrt(2.0) * torch.randn(m, d, dtype=torch.double)
        self.R = np.sqrt(2.0) * torch.randn(m, d, dtype=torch.double)
        self.y = torch.sum(self.L.mm(self.X) * self.R.mm(self.X), dim=1)

    def loss(self):
        m = self.y.size(0)
        return lambda Z: (1 / m) * torch.linalg.norm(
            self.y - torch.sum(self.L.mm(Z) * self.R.mm(Z), dim=1), 1
        )

    def subgradient(self):
        def f(z):
            return torch.autograd.grad(self.loss()(z), z)[0]

        return f

    def initializer(self, δ):
        Δ = torch.randn(self.X.size(), dtype=torch.double)
        Δ = Δ / torch.linalg.norm(Δ)
        return (self.X + δ * Δ).requires_grad_(True)


class BilinearSensingProblem:
    def __init__(self, m, d, r):
        self.L = np.sqrt(2.0) * torch.randn(m, d, dtype=torch.double)
        self.R = np.sqrt(2.0) * torch.randn(m, d, dtype=torch.double)
        self.W = torch.tensor(
            np.linalg.qr(np.random.randn(d, r), mode="reduced")[0],
            dtype=torch.double
        )
        self.X = torch.tensor(
            np.linalg.qr(np.random.randn(d, r), mode="reduced")[0],
            dtype=torch.double
        )
        self.y = torch.sum(self.L.mm(self.W) * self.R.mm(self.X), dim=1)

    def loss(self):
        def f(z):
            W = z[0 : self.W.numel()].view(self.W.size())
            X = z[self.W.numel() :].view(self.X.size())
            return (1.0 / self.y.size(0)) * torch.linalg.norm(
                self.y - torch.sum(self.L.mm(W) * self.R.mm(X), dim=1), 1
            )

        return f

    def initializer(self, δ):
        Δ = torch.randn(self.W.numel() + self.X.numel(), dtype=torch.double)
        return (
            torch.cat([self.W.reshape(-1), self.X.reshape(-1)]) + δ * Δ / torch.norm(Δ)
        ).requires_grad_(True)


# here is the translation


def phase(v):
    return v / torch.norm(v)


def fnorm(v, nrm):
    return torch.zeros(v.size()) if nrm <= 1e-15 else (v / nrm)

def soft_threshold(x, threshold):
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class LassoProblem:
    def __init__(
        self,
        m: int,
        d: int,
        k: int,
        noise_stddev: float = 0.1,
        l1_penalty: Optional[float] = None,
    ):
        self.A = torch.tensor(
            np.linalg.qr(np.random.randn(d, m))[0].T, dtype=torch.double
        )
        self.x = generate_sparse_vector(d, k)
        self.y = self.A @ self.x + noise_stddev * torch.randn(m, dtype=torch.double)
        self.noise_stddev = noise_stddev
        if l1_penalty is None:
            self.l1_penalty = 0.2 * np.linalg.norm(self.A.T @ self.y, np.inf)
        else:
            self.l1_penalty = l1_penalty
        self.prox_step = 0.95 / (np.linalg.norm(self.A, 2) ** 2)

    def proximal_gradient(self, x):
        return soft_threshold(
            x - self.prox_step * (self.A.T @ (self.A @ x - self.y)),
            self.l1_penalty * self.prox_step,
        )

    def loss(self):
        return lambda x: torch.linalg.norm(x - self.proximal_gradient(x))

    def initializer(self, δ):
        return (
            self.x
            + δ * normalize(torch.randn(self.x.size(), dtype=torch.double), dim=-1)
        ).requires_grad_(True)

    def support_recovery_lambda(self):
        """Compute an l1 penalty suggested by nonasymptotic theory."""
        m, d = self.A.shape
        nnz_ind = np.abs(self.x) > 1e-15
        S = self.A[:, nnz_ind].numpy()
        T = self.A[:, ~nnz_ind].numpy()
        gamma = 1.0 - np.linalg.norm(T.T @ (S @ np.linalg.inv(S.T @ S)), np.inf)
        return (2 / gamma) * self.noise_stddev * np.sqrt(np.log(d) / m)

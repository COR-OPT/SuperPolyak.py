import torch
from torch.nn.functional import normalize
import numpy as np


class ReluRegressionProblem:
    def __init__(self, m, d):
        self.A = torch.randn(m, d, dtype=torch.double)
        self.x = normalize(torch.randn(d, dtype=torch.double), dim=-1)
        self.y = torch.max(self.A @ (self.x), torch.zeros(m))

    def loss(self):
        def f(z):
            return (1.0 / self.A.size()[0]) * torch.norm(
                torch.max(self.A @ (z), torch.zeros(self.A.size()[0])) - self.y, 1)

        return f

    def subgradient(self):
        def f(z):
            return torch.autograd.grad(self.loss()(z), z)[0]

        return f

    def initializer(self, δ):
        return (self.x + δ * normalize(torch.randn(self.x.size(), dtype=torch.double), dim=-1)).requires_grad_(True)


class MaxAffineRegressionProblem:
    def __init__(self, m, d, k):
        self.A = torch.randn(m, d, dtype=torch.double)
        self.βs = normalize(torch.randn(d, k, dtype=torch.double), dim=1)
        self.y = torch.max(self.A.mm(self.βs), dim=1)[0]

    def loss(self):
        def f(z):
            return (1.0 / self.A.size()[0]) * torch.linalg.norm(
                torch.max(self.A.mm(z.view(self.βs.size())), dim=1)[0] - self.y, 1)

        return f

    def subgradient(self):
        def f(z):
            return torch.autograd.grad(self.loss()(z), z)[0]

        return f

    def initializer(self, δ):
        return (self.βs + δ * normalize(torch.randn(self.βs.size(), dtype=torch.double), dim=1)).requires_grad_(True)


class CompressedSensingProblem:
    def __init__(self, m, d, k):
        self.A = torch.randn(m, d, dtype=torch.double)
        self.F = torch.qr(self.A, True)
        self.x = self.generate_sparse_vector(d, k)
        self.y = self.A @ (self.x)
        self.k = k

    def proj_sparse(self, x):
        x[torch.topk(torch.abs(x), self.k)[1][-1] + 1:] = 0
        return x

    def dist_sparse(self, x):
        return torch.linalg.norm(x[torch.topk(torch.abs(x), self.x.size()[0] - self.k, largest=False)[1]])
        # return torch.norm(x - self.proj_sparse(x))

    def grad_sparse(self, x):
        x_plus = self.proj_sparse(x)
        ds = torch.norm(x_plus - x)
        return torch.zeros(x.size()) if ds < 1e-15 else (x - x_plus) / ds

    def proj_range(self, x):
        return x + torch.linalg.solve(self.F[0], self.y - self.A @ (x), )[0]

    def dist_range(self, x):
        return torch.norm(torch.linalg.solve(self.F[0], self.y - self.A @ (x))[0])

    def grad_range(self, x):
        dx = torch.linalg.solve(self.F[0], self.y - self.A @ (x))[0]
        ds = torch.norm(dx)
        return torch.zeros(x.size()) if ds < 1e-15 else -dx / ds

    def loss(self):
        def f(z):
            return self.dist_sparse(z) + self.dist_range(z)

        return f

    def subgradient(self):
        def f(z):
            return self.grad_sparse(z) + self.grad_range(z)

        return f

    def initializer(self, δ):
        return (self.x + δ * normalize(torch.randn(self.x.size(), dtype=torch.double), dim=-1)).requires_grad_(True)


def generate_sparse_vector(self, d, k):
    x = torch.randn(d, dtype=torch.double)
    x[torch.topk(torch.abs(x), k)[1]] = 0
    return x


# The phase retrieval problem
class PhaseRetrievalProblem:
    def __init__(self, m, d):
        self.A = torch.randn(m, d)
        self.x = normalize(torch.randn(d), dim=-1)
        self.y = torch.abs(self.A @ self.x)

    def loss(self):
        def f(z):
            return (1.0 / self.A.size()[0]) * torch.norm(torch.abs(self.A @ z) - self.y, 1)

        return f

    def subgradient(self):
        def f(z):
            return torch.autograd.grad(self.loss()(z), z)[0]

        return f

    def initializer(self, δ):
        return (self.x + δ * normalize(torch.randn(self.x.size()), dim=-1)).requires_grad_(True)

    # Need to discuss this poriton of the code. Why not just give the alt proj part access to self?
    def loss_altproj(problem):
        A = problem.A
        y = problem.y
        m = y.size()[0]
        # Cache factorization of A.
        F = torch.qr(A)

        def f(z):
            z_comp = z[0:m] + z[m:2 * m] * 1j
            return torch.norm(z_comp - A.mm(torch.linalg.solve(z_comp, F[0])[0])) + torch.norm(
                z_comp - y * phase(z_comp))

        return f

    def subgradient_altproj(problem):
        A = problem.A
        y = problem.y
        m = y.size()[0]
        # Cache factorization of A.
        F = torch.qr(A)
        # Cache subgradients.
        g = torch.zeros(2 * m)
        g_comp = torch.zeros(m)

        def f(z):
            z_comp = z[0:m] + z[m:2 * m] * 1j
            diff_range = z_comp - A.mm(torch.linalg.solve(z_comp, F[0])[0])
            diff_phase = z_comp - y * phase(z_comp)
            norm_range = torch.norm(diff_range)
            norm_phase = torch.norm(diff_phase)
            # Separate real and imaginary parts in the subgradient.
            g_comp.copy_(fnorm(diff_range, norm_range) + fnorm(diff_phase, norm_phase))
            g[0:m] = g_comp.real
            g[m:2 * m] = g_comp.imag
            return g

        return f

    #

    def alternating_projections_step(problem):
        A = problem.A
        y = problem.y
        m = y.size()[0]
        F = torch.qr(A)

        def f(z):
            phased = phase(A.mm(torch.linalg.solve(z[0:m] + z[m:2 * m] * 1j, F[0])[0]))
            return torch.cat([y * phased.real, y * phased.imag])

        return f

    def initializer_altproj(problem, δ):
        x = problem.x + δ * normalize(torch.randn(problem.x.size()), dim=-1)
        Ax = problem.A @ (x)
        return torch.cat([Ax.real, Ax.imag]).requires_grad_(True)


# now we translate the quadratic sensing problem to pytorch

class QuadraticSensingProblem:
    def __init__(self, m, d, r):
        # self.X = torch.linalg.qr(torch.randn(d, r, dtype=torch.double))[0]
        ## the above results in the error
        ###  RuntimeError: view size is not compatible with input tensor's size and stride
        ### (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        ## Not sure why, so will come back to it.
        self.X = torch.randn(d, r, dtype=torch.double)
        self.L = np.sqrt(2.0) * torch.randn(m, d, dtype=torch.double)
        self.R = np.sqrt(2.0) * torch.randn(m, d, dtype=torch.double)
        self.y = torch.sum(self.L.mm(self.X) * self.R.mm(self.X), dim=1)

    def loss(self):
        def f(z):
            Z = z.view(self.X.size())
            return (1.0 / self.y.size()[0]) * torch.linalg.norm(self.y - torch.sum(self.L.mm(Z) * self.R.mm(Z), dim=1),
                                                                1)

        return f

    def subgradient(self):
        def f(z):
            return torch.autograd.grad(self.loss()(z), z)[0]

        return f

    def initializer(self, δ):
        Δ = torch.randn(self.X.size(), dtype=torch.double)
        Δ = Δ / torch.linalg.norm(Δ)
        return (self.X + δ * Δ).requires_grad_(True)


# now we translate the bilinear sensing problem to pytorch
# here is what the code looks like

class BilinearSensingProblem:
    def __init__(self, m, d, r):
        self.L = np.sqrt(2.0) * torch.randn(m, d, dtype=torch.double)
        self.R = np.sqrt(2.0) * torch.randn(m, d, dtype=torch.double)
        self.W = torch.qr(torch.randn(d, r, dtype=torch.double))[0]
        self.X = torch.qr(torch.randn(d, r, dtype=torch.double))[0]
        self.y = torch.sum(self.L.mm(self.W) * self.R.mm(self.X), dim=1)

    def loss(self):
        def f(z):
            W = z[0:self.W.numel()].view(self.W.size())
            X = z[self.W.numel():].view(self.X.size())
            return (1.0 / self.y.size()[0]) * torch.linalg.norm(self.y - torch.sum(self.L.mm(W) * self.R.mm(X), dim=1), 1)

        return f

    def subgradient(self):
        def f(z):
            return torch.autograd.grad(self.loss()(z), z)[0]

        return f

    def initializer(self, δ):
        Δ = torch.randn(self.W.numel() + self.X.numel(), dtype=torch.double)
        return (torch.cat([self.W.reshape(-1), self.X.reshape(-1)]) + δ * Δ / torch.norm(Δ)).requires_grad_(True)


# here is the translation
def generate_sparse_vector(d, k):
    x = torch.zeros(d, dtype=torch.double)
    x[torch.randperm(d)[:k]] = normalize(torch.randn(k, dtype=torch.double), dim=-1)
    return x


# here is the translation

def phase(v):
    return v / torch.norm(v)


def fnorm(v, nrm):
    return torch.zeros(v.size()) if nrm <= 1e-15 else (v / nrm)

# here is the translation

def soft_threshold(x, τ):
    # return torch.max(x - τ, torch.zeros(x.size())) - torch.max(-x - τ, torch.zeros(x.size()))
    return torch.sign(x) * torch.max(torch.abs(x) - τ, torch.zeros(x.size()))

# This class appears to be broken. I will come back to it.
class LassoProblem:
    def __init__(self, m, d, k, σ=0.1, λ=None):
        self.A = torch.tensor(np.linalg.qr(torch.randn(d, m))[0].T, dtype=torch.double)
        self.x = generate_sparse_vector(d, k)
        self.y = self.A @ self.x + σ * torch.randn(m, dtype=torch.double)
        self.λ = λ
        self.σ=σ
        if self.λ is None:
            self.λ = 0.2 * torch.linalg.norm(self.A.T@self.y, torch.inf) # support_recovery_lambda(self.A, self.x, self.σ)
        self.τ = 0.95 / torch.linalg.norm(self.A) ** 2

    def proximal_gradient(self, x):
        return soft_threshold(x - self.τ * (self.A.T @ (self.A @ x - self.y)), self.λ * self.τ)

    def loss(self):
        def f(z):
            # grad_step = z - self.τ * self.A.T @ (self.A @ z - self.y)
            return torch.linalg.norm(z - self.proximal_gradient(z))
            # return torch.norm(z - torch.sign(grad_step) * torch.max(torch.abs(grad_step) - self.λ * self.τ, torch.zeros(grad_step.size())))
        return f

    def subgradient(self):
        d = self.A.size(1)
        return lambda z: torch.autograd.grad(self.loss(self.τ)(z), z, create_graph=True)[0]

    def initializer(self, δ):
        return (self.x + δ * normalize(torch.randn(self.x.size(), dtype=torch.double),dim=-1)).requires_grad_(True)


def support_recovery_lambda(A, x, σ):
    m, d = A.size()
    nnz_ind = torch.abs(x) > 1e-15
    S = A[:, nnz_ind]
    T = A[:, ~nnz_ind]
    γ = 1.0 - torch.norm(T.t() @ (S @ torch.inverse(S.T @ S)), float('inf'))
    return (2.0 / γ) * np.sqrt(σ ** 2 * np.log(d) / m)

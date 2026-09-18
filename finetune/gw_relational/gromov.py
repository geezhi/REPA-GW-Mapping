"""Entropic Gromov-Wasserstein optimal transport (pure PyTorch, no POT dependency).

Notation follows the paper appendix
-----------------------------------
Student features :math:`X \\in R^{N x D_1}` (diffusion transformer, e.g. ``D_1 = 1920``),
teacher features :math:`Y \\in R^{N x D_2}` (frozen video encoder, e.g. ``D_2 = 768``).

**Intra-space structure** (Eq. 1)::

    C_X[i, k] = 1 - cos(X[:, i], X[:, k])       # (D1, D1)
    C_Y[j, l] = 1 - cos(Y[:, j], Y[:, l])       # (D2, D2)

Each *dimension* is treated as a point; each *token* is an observation of that point.

**Entropic GW objective** (Eq. 3)::

    T* = argmin_{T in Pi(a, b)}  sum_{i,j,k,l} |C_X[i,k] - C_Y[j,l]|^2 T[i,j] T[k,l]  -  eps * H(T)

with uniform marginals ``a = 1/D1``, ``b = 1/D2``.

**Solver.** We linearize around the current plan and solve the resulting entropic OT
problem with log-domain Sinkhorn (Eqs. 4-6)::

    M^(t)   = (C_X^2 a) 1^T + 1 (C_Y^2 b)^T - 2 C_X T^(t) C_Y^T
    u       = log a - LSE_j(-M^(t)/eps + v)
    v       = log b - LSE_i(-M^(t)/eps + u)
    T[i, j] = exp(u_i - M^(t)[i, j]/eps + v_j)

Everything here is ``torch.no_grad``-friendly and runs in fp32 for numerical safety;
bf16 inputs are upcast internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "GWSolverConfig",
    "pairwise_cosine_distance",
    "sinkhorn_log",
    "entropic_gromov_wasserstein",
    "gromov_cost",
    "dimension_transport_plan",
    "row_normalize",
]


@dataclass
class GWSolverConfig:
    """Hyper-parameters of the entropic GW solver.

    Args:
        reg: Entropic regularization ``eps``. Larger values give a smoother (more
            uniform) plan; smaller values approach an exact assignment.
        num_outer_iters: Maximum number of linearization (outer) iterations ``K_outer``.
        num_sinkhorn_iters: Maximum number of inner Sinkhorn iterations ``K_sink``.
        outer_tol: Relative Frobenius-norm tolerance for early stopping of the outer
            loop. Set to ``0`` to always run ``num_outer_iters``.
        sinkhorn_tol: Convergence tolerance for the inner Sinkhorn loop (``0`` disables).
        token_sample_size: If set, subsample this many tokens when building
            ``C_X``/``C_Y``. ``None`` (or ``<= 0``) uses all tokens, which is what the
            paper uses. Subsampling trades accuracy for speed.
    """

    reg: float = 0.1
    num_outer_iters: int = 50
    num_sinkhorn_iters: int = 100
    outer_tol: float = 1e-4
    sinkhorn_tol: float = 1e-4
    token_sample_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.reg <= 0:
            raise ValueError(f"reg (eps) must be > 0, got {self.reg}")
        if self.num_outer_iters < 1 or self.num_sinkhorn_iters < 1:
            raise ValueError("num_outer_iters and num_sinkhorn_iters must be >= 1")
        # ``0`` is the sentinel used by the CLI for "use all tokens".
        if self.token_sample_size is not None and self.token_sample_size <= 0:
            self.token_sample_size = None

    @classmethod
    def from_args(cls, args: Any) -> "GWSolverConfig":
        """Build a config from a training args namespace (``--gw_*`` flags)."""
        return cls(
            reg=getattr(args, "gw_reg", 0.1),
            num_outer_iters=getattr(args, "gw_outer_iters", 50),
            num_sinkhorn_iters=getattr(args, "gw_sinkhorn_iters", 100),
            outer_tol=getattr(args, "gw_outer_tol", 1e-4),
            sinkhorn_tol=getattr(args, "gw_sinkhorn_tol", 1e-4),
            token_sample_size=getattr(args, "gw_sample_size", 0),
        )


def _uniform(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((n,), 1.0 / n, device=device, dtype=dtype)


def pairwise_cosine_distance(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine distance matrix between the **columns** (dimensions) of ``x``.

    Args:
        x: ``(N, D)`` feature matrix. ``N`` tokens, ``D`` dimensions.
        eps: Numerical floor for the column norm.

    Returns:
        ``(D, D)`` matrix with entries in ``[0, 2]``.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D (N, D) tensor, got shape {tuple(x.shape)}")

    x = F.normalize(x.float(), dim=0, eps=eps)
    dist = 1.0 - x.T @ x
    return dist.clamp_(min=0.0, max=2.0)


def sinkhorn_log(
    cost: torch.Tensor,
    reg: float,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    num_iters: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Entropic optimal transport via log-domain Sinkhorn (Eqs. 5-6).

    Solves ``min_T <T, cost> - reg * H(T)`` subject to ``T 1 = a`` and ``T^T 1 = b``.

    Args:
        cost: ``(D1, D2)`` cost matrix.
        reg: Entropic regularization ``eps``.
        a: ``(D1,)`` source marginal. Defaults to uniform.
        b: ``(D2,)`` target marginal. Defaults to uniform.
        num_iters: Maximum number of Sinkhorn iterations.
        tol: Early-stopping tolerance on the dual potential ``u`` (``0`` disables).

    Returns:
        ``(D1, D2)`` transport plan with row sums ``a`` and column sums ``b``.
    """
    d1, d2 = cost.shape
    device, dtype = cost.device, cost.dtype

    if a is None:
        a = _uniform(d1, device, dtype)
    if b is None:
        b = _uniform(d2, device, dtype)

    log_a = torch.log(a.clamp_min(1e-20))
    log_b = torch.log(b.clamp_min(1e-20))
    log_k = -cost / reg  # (D1, D2)

    u = torch.zeros(d1, device=device, dtype=dtype)
    v = torch.zeros(d2, device=device, dtype=dtype)

    for i in range(num_iters):
        u_prev = u
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)

        if tol > 0 and i % 5 == 0:
            if torch.max(torch.abs(u - u_prev)).item() < tol:
                break

    return torch.exp(u.unsqueeze(1) + log_k + v.unsqueeze(0))


def entropic_gromov_wasserstein(
    c_x: torch.Tensor,
    c_y: torch.Tensor,
    reg: float = 0.1,
    num_outer_iters: int = 50,
    num_sinkhorn_iters: int = 100,
    outer_tol: float = 1e-4,
    sinkhorn_tol: float = 1e-4,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve the entropic GW problem (Eq. 3) by iterative linearization + Sinkhorn.

    Args:
        c_x: ``(D1, D1)`` intra-space distance matrix of the source (student).
        c_y: ``(D2, D2)`` intra-space distance matrix of the target (teacher).
        reg: Entropic regularization ``eps``.
        num_outer_iters: Number of linearization steps ``K_outer``.
        num_sinkhorn_iters: Number of inner Sinkhorn steps ``K_sink``.
        outer_tol: Relative Frobenius tolerance for early stopping (``0`` disables).
        sinkhorn_tol: Tolerance of the inner Sinkhorn loop.
        a, b: Optional non-uniform marginals (uniform by default).

    Returns:
        ``(D1, D2)`` transport plan.
    """
    d1, d2 = c_x.shape[0], c_y.shape[0]
    device, dtype = c_x.device, c_x.dtype

    if a is None:
        a = _uniform(d1, device, dtype)
    if b is None:
        b = _uniform(d2, device, dtype)

    # Uniform plan is a neutral initialisation: it carries no correspondence prior.
    plan = a.unsqueeze(1) * b.unsqueeze(0)

    c_x_sq = c_x**2
    c_y_sq = c_y**2

    for _ in range(num_outer_iters):
        # Linearized cost: (C_X^2 a) 1^T + 1 (C_Y^2 b)^T - 2 C_X T C_Y^T
        term_src = (c_x_sq @ a).unsqueeze(1)  # (D1, 1)
        term_tgt = (c_y_sq @ b).unsqueeze(0)  # (1, D2)
        term_cross = c_x @ plan @ c_y.T  # (D1, D2)
        cost_lin = term_src + term_tgt - 2.0 * term_cross

        plan_new = sinkhorn_log(
            cost_lin, reg=reg, a=a, b=b, num_iters=num_sinkhorn_iters, tol=sinkhorn_tol
        )

        if outer_tol > 0:
            rel_change = (
                torch.norm(plan_new - plan, p="fro") / (torch.norm(plan, p="fro") + 1e-20)
            ).item()
            plan = plan_new
            if rel_change < outer_tol:
                break
        else:
            plan = plan_new

    return plan


def gromov_cost(
    c_x: torch.Tensor,
    c_y: torch.Tensor,
    plan: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate the (unregularized) GW objective (Eq. 3) at ``plan``.

    Useful as a *diagnostic*: it is the quantity the solver minimizes, so a small
    value means the two structures are genuinely isomorphic.

    Args:
        c_x: ``(D1, D1)`` source structure matrix.
        c_y: ``(D2, D2)`` target structure matrix.
        plan: ``(D1, D2)`` transport plan.
        a, b: Marginals (uniform by default).

    Returns:
        Scalar ``sum_{i,j,k,l} |C_X[i,k] - C_Y[j,l]|^2 T[i,j] T[k,l]``.
    """
    if a is None:
        a = _uniform(c_x.shape[0], c_x.device, c_x.dtype)
    if b is None:
        b = _uniform(c_y.shape[0], c_y.device, c_y.dtype)

    const = (c_x**2 @ a).unsqueeze(1) + (c_y**2 @ b).unsqueeze(0)
    linearized = const - 2.0 * (c_x @ plan @ c_y.T)
    return (plan * linearized).sum()


@torch.no_grad()
def dimension_transport_plan(
    x: torch.Tensor,
    y: torch.Tensor,
    config: Optional[GWSolverConfig] = None,
    **solver_kwargs: Any,
) -> torch.Tensor:
    """Compute the dimension-level GW transport plan ``T in R^{D1 x D2}``.

    The plan is always computed under ``no_grad`` in fp32: it is a *correspondence*,
    not a learnable mapping, so it must not carry gradient or add memory to backward.

    Args:
        x: ``(N, D1)`` student features.
        y: ``(N, D2)`` teacher features (same ``N``: tokens are spatially aligned).
        config: Solver hyper-parameters. A default config is used when omitted.
        **solver_kwargs: Overrides forwarded to :class:`GWSolverConfig`.

    Returns:
        ``(D1, D2)`` transport plan, detached.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"student and teacher must share the token axis, got {x.shape[0]} vs {y.shape[0]}"
        )

    config = config or GWSolverConfig()
    if solver_kwargs:
        config = GWSolverConfig(**{**config.__dict__, **solver_kwargs})

    x_src, y_src = x.float(), y.float()

    # Optional token subsampling: C_X / C_Y are D x D, so their *construction* is
    # the O(N D^2) part of the cost. Subsampling only affects the structure matrices.
    sample_size = config.token_sample_size
    if sample_size is not None and 0 < sample_size < x_src.shape[0]:
        idx = torch.randperm(x_src.shape[0], device=x_src.device)[:sample_size]
        x_src, y_src = x_src[idx], y_src[idx]

    c_x = pairwise_cosine_distance(x_src)
    c_y = pairwise_cosine_distance(y_src)

    return entropic_gromov_wasserstein(
        c_x,
        c_y,
        reg=config.reg,
        num_outer_iters=config.num_outer_iters,
        num_sinkhorn_iters=config.num_sinkhorn_iters,
        outer_tol=config.outer_tol,
        sinkhorn_tol=config.sinkhorn_tol,
    )


def row_normalize(plan: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Row-normalize a transport plan into a soft projection matrix ``T_hat``.

    ``T_hat[i, j]`` reads as "how much of teacher dimension ``j`` contributes to
    student dimension ``i``", and each row sums to one.
    """
    return plan.float() / (plan.float().sum(dim=1, keepdim=True) + eps)

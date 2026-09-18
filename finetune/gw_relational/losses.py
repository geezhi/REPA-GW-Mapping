"""Alignment losses driven by a Gromov-Wasserstein dimension transport plan.

Given the plan ``T in R^{D1 x D2}`` from :func:`~gw_relational.gromov.dimension_transport_plan`,
we row-normalize it into a soft projection (Eq. 7 of the paper)::

    T_hat[i, j] = T[i, j] / sum_j' T[i, j']
    Y_hat       = Y @ T_hat^T                      # (N, D2) -> (N, D1)

and score the student in its own space. Two instantiations are provided:

* :class:`GWRelationalLoss` -- per-token cosine loss (Eq. 8, the default ``gw_relational``).
* :class:`GWGramLoss`       -- Gram-matrix (token-token similarity) MSE (``gw_gram``).

Both are ``nn.Module``, take batched ``(B, N, D)`` inputs, and are **parameter-free**:
the only learnable signal flows back into the student features.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gromov import GWSolverConfig, dimension_transport_plan, row_normalize

__all__ = ["GWAlignmentLoss", "GWRelationalLoss", "GWGramLoss"]


class GWAlignmentLoss(nn.Module):
    """Base class for GW-guided alignment losses.

    Args:
        config: GW solver hyper-parameters.
        margin: Hinge margin on the per-token cosine distance. Distances below the
            margin contribute zero loss. ``0.0`` (default) disables the hinge.
        token_sample_size: Optional subsample of tokens used for the *loss* term
            (independent of the solver's own subsampling).

    Shapes:
        ``forward(student, teacher)`` with ``student`` of ``(B, N, D1)`` and
        ``teacher`` of ``(B, N, D2)``, returning a scalar.
    """

    def __init__(
        self,
        config: Optional[GWSolverConfig] = None,
        margin: float = 0.0,
        token_sample_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config or GWSolverConfig()
        self.margin = float(margin)
        self.token_sample_size = token_sample_size

    # -- internals ---------------------------------------------------------
    def _maybe_subsample(
        self, student: torch.Tensor, teacher: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = student.shape[1]
        size = self.token_sample_size
        if size is not None and 0 < size < n:
            idx = torch.randperm(n, device=student.device)[:size]
            return student[:, idx], teacher[:, idx]
        return student, teacher

    def _project_teacher(
        self, student: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
        """Transport-plan projection of the teacher into the student's space."""
        batch = student.shape[0]
        projected = []
        for b in range(batch):
            plan = dimension_transport_plan(student[b].detach(), teacher[b].detach(), self.config)
            projected.append(teacher[b].float() @ row_normalize(plan).T)
        return torch.stack(projected, dim=0)

    def alignment_loss(
        self, student: torch.Tensor, teacher_projected: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    # -- public API --------------------------------------------------------
    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Compute the alignment loss for a batch.

        Args:
            student: ``(B, N, D1)``, requires grad.
            teacher: ``(B, N, D2)``, detached (frozen encoder).

        Returns:
            Scalar loss averaged over the batch.
        """
        self._validate(student, teacher)
        student, teacher = self._maybe_subsample(student, teacher)

        teacher_projected = self._project_teacher(student, teacher)

        total = student.new_zeros((), dtype=torch.float32)
        for b in range(student.shape[0]):
            total = total + self.alignment_loss(student[b], teacher_projected[b])
        return total / student.shape[0]

    def forward_flat(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for un-batched ``(N, D1)`` / ``(N, D2)`` inputs."""
        return self.forward(student.unsqueeze(0), teacher.unsqueeze(0))

    @staticmethod
    def _validate(student: torch.Tensor, teacher: torch.Tensor) -> None:
        if student.ndim != 3 or teacher.ndim != 3:
            raise ValueError(
                "expected (B, N, D) inputs, got "
                f"{tuple(student.shape)} and {tuple(teacher.shape)}"
            )
        if student.shape[0] != teacher.shape[0] or student.shape[1] != teacher.shape[1]:
            raise ValueError(
                "student and teacher must share the batch and token axes, got "
                f"{tuple(student.shape)} vs {tuple(teacher.shape)}"
            )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"reg={self.config.reg}, K_outer={self.config.num_outer_iters}, "
            f"K_sink={self.config.num_sinkhorn_iters}, margin={self.margin}"
        )


class GWRelationalLoss(GWAlignmentLoss):
    """Per-token cosine alignment loss (Eq. 8)::

        L_GW = 1/N sum_n [ 1 - cos(X_n, Y_hat_n) ]      (+ optional hinge margin)

    This is the ``--loss gw_relational`` variant: it aligns *feature values* once the
    dimension correspondence has been established by GW.
    """

    def alignment_loss(
        self, student: torch.Tensor, teacher_projected: torch.Tensor
    ) -> torch.Tensor:
        student_n = F.normalize(student.float(), dim=-1)
        teacher_n = F.normalize(teacher_projected.float(), dim=-1)

        cos_dist = 1.0 - (student_n * teacher_n).sum(dim=-1)
        if self.margin > 0.0:
            cos_dist = F.relu(cos_dist - self.margin)
        return cos_dist.mean()


class GWGramLoss(GWAlignmentLoss):
    """Gram-matrix alignment on top of the GW projection (``--loss gw_gram``).

    After projecting the teacher with ``T_hat``, both sides are L2-normalized and
    compared through their token-token cosine Gram matrices::

        L_gram = MSE( G_X , G_Y_hat ),   G_X = X_norm @ X_norm^T

    Compared to :class:`GWRelationalLoss` this is invariant to any global rotation of
    the token set and emphasizes relational structure over absolute values.
    """

    def alignment_loss(
        self, student: torch.Tensor, teacher_projected: torch.Tensor
    ) -> torch.Tensor:
        student_n = F.normalize(student.float(), dim=-1)
        teacher_n = F.normalize(teacher_projected.float(), dim=-1)

        gram_student = student_n @ student_n.T
        gram_teacher = teacher_n @ teacher_n.T
        return F.mse_loss(gram_student, gram_teacher)


def build_alignment_loss(
    loss_name: str,
    args: Any = None,
    config: Optional[GWSolverConfig] = None,
) -> GWAlignmentLoss:
    """Factory used by the trainer to map ``--loss`` to a loss module.

    Args:
        loss_name: ``"gw_relational"`` or ``"gw_gram"``.
        args: Optional args namespace used to build the solver config.
        config: Explicit solver config (takes precedence over ``args``).
    """
    if config is None:
        config = GWSolverConfig.from_args(args) if args is not None else GWSolverConfig()

    margin = float(getattr(args, "gw_margin", 0.0)) if args is not None else 0.0

    registry = {
        "gw_relational": GWRelationalLoss,
        "gw_gram": GWGramLoss,
    }
    if loss_name not in registry:
        raise ValueError(f"unknown GW loss '{loss_name}'. Choose from {sorted(registry)}")
    return registry[loss_name](config=config, margin=margin)

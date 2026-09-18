#!/usr/bin/env python
"""Minimal, dependency-free demo of GW dimension alignment.

Two parts:

1. **Sanity check** with ``D1 == D2``: the teacher is a random permutation of the
   student's dimensions. GW should recover the permutation *exactly* -- it matches
   structure without ever seeing the correspondence.
2. **Realistic geometry** with ``D1 != D2`` (e.g. CogVideoX 1920 vs VideoMAEv2 768):
   we show the shape of the transport plan, its marginals, and the projection of
   the teacher into the student's space. Note that with ``D1 != D2`` and uniform
   marginals a *hard* assignment is infeasible, so the plan is necessarily soft.

Run::

    python -m gw_relational.example           # from the ``finetune/`` directory
    python finetune/gw_relational/example.py  # from the repository root
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):  # allow ``python path/to/example.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_relational.gromov import (  # noqa: E402
    GWSolverConfig,
    dimension_transport_plan,
    gromov_cost,
    pairwise_cosine_distance,
    row_normalize,
)


def demo_permuted_dimensions() -> None:
    """Recover a hidden dimension permutation (D1 == D2, exact recovery possible)."""
    print("=" * 68)
    print("Part 1 - recovering a hidden dimension permutation (D1 == D2)")
    print("=" * 68)

    num_tokens, dim = 256, 64
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(num_tokens, dim, generator=generator)
    perm = torch.randperm(dim, generator=generator)
    teacher = student[:, perm]  # same features, relabelled dimensions

    # `reg` (eps) must be small compared to the spread of the structure matrix.
    # For i.i.d. Gaussian features that spread shrinks as ~1/sqrt(num_tokens), so
    # eps has to be tightened as the problem grows. Real (trained) features have a
    # much stronger structure and tolerate larger eps -- see the hyper-parameter
    # note in gw_relational/README.md.
    config = GWSolverConfig(reg=0.002, num_outer_iters=60, num_sinkhorn_iters=300)
    plan = dimension_transport_plan(student, teacher, config)

    predicted = plan.argmax(dim=0)  # best student dim for each teacher dim
    accuracy = (predicted == perm).float().mean().item()

    c_x = pairwise_cosine_distance(student)
    c_y = pairwise_cosine_distance(teacher)
    uniform_plan = torch.full_like(plan, 1.0 / (dim * dim))

    print(f"student {tuple(student.shape)}   teacher {tuple(teacher.shape)}")
    print(f"C_X {tuple(c_x.shape)}  C_Y {tuple(c_y.shape)}  T {tuple(plan.shape)}")
    print(f"row sums of T      : {plan.sum(dim=1).mean():.4f} (expected {1.0 / dim:.4f})")
    print(f"GW cost at T*      : {gromov_cost(c_x, c_y, plan):.5f}")
    print(f"GW cost at uniform : {gromov_cost(c_x, c_y, uniform_plan):.5f}")
    print(f"permutation recovery: {accuracy:.3f}  (chance = {1.0 / dim:.3f})")
    print()


def demo_asymmetric_spaces() -> None:
    """The realistic setting: 1920-dim student vs 768-dim teacher."""
    print("=" * 68)
    print("Part 2 - asymmetric spaces (D1=1920 student vs D2=768 teacher)")
    print("=" * 68)

    # Use a reduced surrogate so the demo runs in seconds on CPU.
    num_tokens, dim_student, dim_teacher = 96, 192, 96
    generator = torch.Generator().manual_seed(1)
    student = torch.randn(num_tokens, dim_student, generator=generator)
    source_idx = torch.randperm(dim_student, generator=generator)[:dim_teacher]
    teacher = student[:, source_idx]  # teacher keeps a subset of the dimensions

    config = GWSolverConfig(reg=0.01, num_outer_iters=20, num_sinkhorn_iters=100)
    plan = dimension_transport_plan(student, teacher, config)
    projection = row_normalize(plan)

    teacher_projected = teacher @ projection.T  # (N, D2) -> (N, D1)

    print(f"student {tuple(student.shape)}   teacher {tuple(teacher.shape)}")
    print(f"transport plan T  : {tuple(plan.shape)}")
    print(f"row sums of T     : {plan.sum(dim=1).mean():.6f} (expected {1.0 / dim_student:.6f})")
    print(f"col sums of T     : {plan.sum(dim=0).mean():.6f} (expected {1.0 / dim_teacher:.6f})")
    print(f"projected teacher : {tuple(teacher_projected.shape)}  (now in the student's space)")
    print(f"row sums of T_hat : {projection.sum(dim=1).mean():.6f} (expected 1.0)")
    print()
    print("With D1 != D2 the uniform marginals forbid a hard assignment, so T is a")
    print("soft many-to-many correspondence -- which is exactly what we want for a")
    print("differentiable, projector-free alignment loss.")


def main() -> None:
    torch.manual_seed(0)
    demo_permuted_dimensions()
    demo_asymmetric_spaces()


if __name__ == "__main__":
    main()

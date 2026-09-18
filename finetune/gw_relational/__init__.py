"""GW-Relational: projector-free representation alignment via Gromov-Wasserstein.

This package contains the *canonical* implementation of the GW alignment method
used in VideoREPA. It is deliberately dependency-light (``torch`` only) so that
the core solver can be reused / unit-tested outside of the training loop.

Public API
----------
:class:`GWSolverConfig`        Solver hyper-parameters, constructible from training args.
:func:`pairwise_cosine_distance`   Intra-space structure matrix :math:`C_X`.
:func:`sinkhorn_log`           Entropic OT solver in the log domain.
:func:`entropic_gromov_wasserstein`  Solver for Eq. (3) of the paper.
:func:`gromov_cost`                 Value of the GW objective at a given plan (diagnostic).
:func:`dimension_transport_plan`    ``(N, D1) x (N, D2) -> (D1, D2)`` plan.
:class:`GWRelationalLoss`      :math:`L_GW` (per-token cosine, Eq. 7).
:class:`GWGramLoss`            Gram-matrix variant of :math:`L_GW`.
:func:`build_alignment_loss`   Maps ``--loss {gw_relational,gw_gram}`` to a module.
:class:`ZeroParamFeatureAdapter`  Parameter-free student -> teacher grid resampling.

Reference
---------
See ``gw_relational/README.md`` for the full derivation and hyper-parameter table.
"""

from .adapter import ZeroParamFeatureAdapter
from .gromov import (
    GWSolverConfig,
    dimension_transport_plan,
    entropic_gromov_wasserstein,
    gromov_cost,
    pairwise_cosine_distance,
    row_normalize,
    sinkhorn_log,
)
from .losses import GWAlignmentLoss, GWGramLoss, GWRelationalLoss, build_alignment_loss

__all__ = [
    "GWSolverConfig",
    "GWAlignmentLoss",
    "GWGramLoss",
    "GWRelationalLoss",
    "ZeroParamFeatureAdapter",
    "build_alignment_loss",
    "dimension_transport_plan",
    "entropic_gromov_wasserstein",
    "gromov_cost",
    "pairwise_cosine_distance",
    "row_normalize",
    "sinkhorn_log",
]

__version__ = "1.0.0"

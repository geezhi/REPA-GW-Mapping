"""Deprecated. Use :mod:`finetune.gw_relational` instead.

The canonical implementation of the GW solver and the alignment losses now lives in
``finetune/gw_relational/`` (pure PyTorch, unit-tested, reusable outside the training
loop). This module is kept as a thin re-export so that older entry points and
notebooks keep working; it will be removed in a future cleanup.
"""

from finetune.gw_relational import (  # noqa: F401
    GWAlignmentLoss,
    GWGramLoss,
    GWRelationalLoss,
    GWSolverConfig,
    ZeroParamFeatureAdapter,
    dimension_transport_plan,
    entropic_gromov_wasserstein,
    gromov_cost,
    pairwise_cosine_distance,
    row_normalize,
    sinkhorn_log,
)

__all__ = [
    "GWAlignmentLoss",
    "GWGramLoss",
    "GWRelationalLoss",
    "GWSolverConfig",
    "ZeroParamFeatureAdapter",
    "dimension_transport_plan",
    "entropic_gromov_wasserstein",
    "gromov_cost",
    "pairwise_cosine_distance",
    "row_normalize",
    "sinkhorn_log",
]

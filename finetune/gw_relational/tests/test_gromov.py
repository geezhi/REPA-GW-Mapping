"""Unit tests for the GW solver, the alignment losses and the feature adapter.

Run from the repository root::

    pytest finetune/gw_relational/tests -q
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from finetune.gw_relational import (
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

# Small problem sizes keep the whole suite under a few seconds on CPU.
N_TOKENS = 48
D_STUDENT = 24
D_TEACHER = 12


# ---------------------------------------------------------------------------
# GWSolverConfig
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = GWSolverConfig()
    assert cfg.reg > 0
    assert cfg.num_outer_iters >= 1
    assert cfg.token_sample_size is None


def test_config_rejects_nonpositive_reg():
    with pytest.raises(ValueError):
        GWSolverConfig(reg=0.0)


def test_config_zero_sample_size_means_all_tokens():
    # ``0`` is the CLI sentinel for "no subsampling".
    assert GWSolverConfig(token_sample_size=0).token_sample_size is None


def test_config_from_args():
    class Args:
        gw_reg = 0.2
        gw_outer_iters = 7
        gw_sinkhorn_iters = 11
        gw_outer_tol = 1e-3
        gw_sinkhorn_tol = 1e-3
        gw_sample_size = 0

    cfg = GWSolverConfig.from_args(Args())
    assert (cfg.reg, cfg.num_outer_iters, cfg.num_sinkhorn_iters) == (0.2, 7, 11)


# ---------------------------------------------------------------------------
# Structure matrices and Sinkhorn
# ---------------------------------------------------------------------------
def test_pairwise_cosine_distance_properties():
    x = torch.randn(N_TOKENS, D_STUDENT)
    dist = pairwise_cosine_distance(x)

    assert dist.shape == (D_STUDENT, D_STUDENT)
    assert torch.allclose(dist, dist.T, atol=1e-5)
    assert torch.allclose(torch.diag(dist), torch.zeros(D_STUDENT), atol=1e-5)
    assert dist.min() >= 0.0 and dist.max() <= 2.0


def test_pairwise_cosine_distance_rejects_non_matrix():
    with pytest.raises(ValueError):
        pairwise_cosine_distance(torch.randn(2, 3, 4))


def test_sinkhorn_satisfies_marginals():
    cost = torch.rand(D_STUDENT, D_TEACHER)
    plan = sinkhorn_log(cost, reg=0.1, num_iters=200, tol=0.0)

    assert plan.shape == (D_STUDENT, D_TEACHER)
    assert torch.all(plan >= 0)
    assert torch.allclose(plan.sum(dim=1), torch.full((D_STUDENT,), 1.0 / D_STUDENT), atol=1e-4)
    assert torch.allclose(plan.sum(dim=0), torch.full((D_TEACHER,), 1.0 / D_TEACHER), atol=1e-4)


# ---------------------------------------------------------------------------
# Gromov-Wasserstein
# ---------------------------------------------------------------------------
def test_gw_plan_shape_and_marginals():
    c_x = pairwise_cosine_distance(torch.randn(N_TOKENS, D_STUDENT))
    c_y = pairwise_cosine_distance(torch.randn(N_TOKENS, D_TEACHER))
    plan = entropic_gromov_wasserstein(
        c_x, c_y, reg=0.1, num_outer_iters=5, num_sinkhorn_iters=50
    )

    assert plan.shape == (D_STUDENT, D_TEACHER)
    assert torch.all(plan >= 0)
    assert torch.allclose(plan.sum(dim=1), torch.full((D_STUDENT,), 1.0 / D_STUDENT), atol=1e-4)


def test_gw_recovers_planted_permutation_when_dims_match():
    """With D1 == D2 a (scaled) permutation is feasible, so GW should find it.

    Note: ``reg`` must be small relative to the *strength of the structure signal*.
    With i.i.d. Gaussian features the off-diagonal spread of ``C_X`` shrinks as
    ``~1/sqrt(N)`` (0.08 at N=128), so ``reg=0.02`` already over-smooths the plan
    while ``reg=0.005`` recovers the permutation exactly.
    """
    dim = 24
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(128, dim, generator=generator)
    perm = torch.randperm(dim, generator=generator)
    teacher = student[:, perm]

    plan = dimension_transport_plan(
        student, teacher, GWSolverConfig(reg=0.005, num_outer_iters=40, num_sinkhorn_iters=200)
    )

    # ``plan[i, j]`` is large when student dim ``i`` corresponds to teacher dim ``j``.
    # Ground truth: teacher dim ``j`` comes from student dim ``perm[j]``.
    predicted = plan.argmax(dim=0)
    accuracy = (predicted == perm).float().mean().item()

    assert accuracy > 0.8, f"GW failed to recover the permutation (acc={accuracy:.3f})"
    assert accuracy > 5 / dim


def test_gw_cost_near_zero_for_isometric_inputs():
    """A dimension permutation is an isometry, so the GW objective should vanish."""
    dim = 24
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(128, dim, generator=generator)
    teacher = student[:, torch.randperm(dim, generator=generator)]

    cfg = GWSolverConfig(reg=0.005, num_outer_iters=40, num_sinkhorn_iters=200)
    c_x = pairwise_cosine_distance(student)
    c_y = pairwise_cosine_distance(teacher)
    plan = dimension_transport_plan(student, teacher, cfg)

    cost_matched = gromov_cost(c_x, c_y, plan)

    # Same structures, but the plan is replaced by an uninformative one.
    cost_uniform = gromov_cost(
        c_x, c_y, torch.full((dim, dim), 1.0 / (dim * dim), dtype=plan.dtype)
    )

    assert cost_matched < 0.1 * cost_uniform, (
        f"GW did not exploit the isomorphism ({cost_matched:.5f} vs {cost_uniform:.5f})"
    )


def test_gw_cost_higher_for_unrelated_spaces():
    """Structurally unrelated spaces must cost strictly more than matched ones."""
    dim = 24
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(128, dim, generator=generator)
    teacher = student[:, torch.randperm(dim, generator=generator)]
    unrelated = torch.randn(128, dim, generator=generator)

    cfg = GWSolverConfig(reg=0.005, num_outer_iters=40, num_sinkhorn_iters=200)
    c_x = pairwise_cosine_distance(student)

    cost_matched = gromov_cost(c_x, pairwise_cosine_distance(teacher),
                               dimension_transport_plan(student, teacher, cfg))
    cost_unrelated = gromov_cost(c_x, pairwise_cosine_distance(unrelated),
                                 dimension_transport_plan(student, unrelated, cfg))

    assert cost_matched < cost_unrelated


def test_gw_is_invariant_to_dimension_permutation():
    """Permuting teacher dimensions must only permute the plan's columns."""
    generator = torch.Generator().manual_seed(1)
    student = torch.randn(64, D_STUDENT, generator=generator)
    teacher = torch.randn(64, D_TEACHER, generator=generator)
    perm = torch.randperm(D_TEACHER, generator=generator)

    cfg = GWSolverConfig(reg=0.1, num_outer_iters=5, num_sinkhorn_iters=50)
    plan = dimension_transport_plan(student, teacher, cfg)
    plan_permuted = dimension_transport_plan(student, teacher[:, perm], cfg)

    assert torch.allclose(plan[:, perm], plan_permuted, atol=1e-3)


def test_transport_plan_has_no_grad():
    student = torch.randn(N_TOKENS, D_STUDENT, requires_grad=True)
    teacher = torch.randn(N_TOKENS, D_TEACHER)
    plan = dimension_transport_plan(student, teacher)

    assert not plan.requires_grad
    assert student.grad is None


def test_transport_plan_accepts_subsampled_tokens():
    student = torch.randn(64, D_STUDENT)
    teacher = torch.randn(64, D_TEACHER)
    cfg = GWSolverConfig(token_sample_size=16, num_outer_iters=2, num_sinkhorn_iters=10)

    assert dimension_transport_plan(student, teacher, cfg).shape == (D_STUDENT, D_TEACHER)


def test_transport_plan_rejects_token_mismatch():
    with pytest.raises(ValueError):
        dimension_transport_plan(torch.randn(10, D_STUDENT), torch.randn(9, D_TEACHER))


def test_row_normalize_rows_sum_to_one():
    plan = torch.rand(D_STUDENT, D_TEACHER).abs() + 1e-3
    assert torch.allclose(row_normalize(plan).sum(dim=1), torch.ones(D_STUDENT), atol=1e-5)


# ---------------------------------------------------------------------------
# Alignment losses
# ---------------------------------------------------------------------------
def _tiny_problem(batch: int = 2):
    generator = torch.Generator().manual_seed(2)
    student = torch.randn(batch, N_TOKENS, D_STUDENT, generator=generator)
    teacher = torch.randn(batch, N_TOKENS, D_TEACHER, generator=generator)
    return student, teacher


def test_relational_loss_gradient_reaches_student_only():
    student, teacher = _tiny_problem()
    student.requires_grad_(True)

    loss = GWRelationalLoss(GWSolverConfig(num_outer_iters=2, num_sinkhorn_iters=10))(
        student, teacher
    )
    loss.backward()

    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None
    assert not torch.allclose(student.grad, torch.zeros_like(student.grad))


def test_relational_loss_range_is_bounded():
    student, teacher = _tiny_problem()
    loss = GWRelationalLoss(GWSolverConfig(num_outer_iters=2, num_sinkhorn_iters=10))(
        student, teacher
    )
    assert 0.0 <= loss.item() <= 2.0


def test_gram_loss_is_finite_and_differentiable():
    student, teacher = _tiny_problem()
    student.requires_grad_(True)

    loss = GWGramLoss(GWSolverConfig(num_outer_iters=2, num_sinkhorn_iters=10))(student, teacher)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None


def test_losses_accept_flat_inputs():
    student = torch.randn(N_TOKENS, D_STUDENT)
    teacher = torch.randn(N_TOKENS, D_TEACHER)
    loss = GWRelationalLoss(GWSolverConfig(num_outer_iters=2, num_sinkhorn_iters=10))
    assert torch.isfinite(loss.forward_flat(student, teacher))


def test_batched_loss_averages_per_sample_losses():
    student, teacher = _tiny_problem(batch=2)
    loss = GWRelationalLoss(GWSolverConfig(num_outer_iters=2, num_sinkhorn_iters=10))

    batched = loss(student, teacher)
    expected = 0.5 * (
        loss.forward_flat(student[0], teacher[0]) + loss.forward_flat(student[1], teacher[1])
    )
    assert torch.allclose(batched, expected, atol=1e-5)


def test_losses_validate_shapes():
    loss = GWRelationalLoss(GWSolverConfig(num_outer_iters=1, num_sinkhorn_iters=1))
    with pytest.raises(ValueError):
        loss(torch.randn(1, N_TOKENS, D_STUDENT), torch.randn(1, N_TOKENS + 1, D_TEACHER))


def test_margin_only_penalises_large_distances():
    student, teacher = _tiny_problem()
    cfg = GWSolverConfig(num_outer_iters=2, num_sinkhorn_iters=10)

    plain = GWRelationalLoss(cfg, margin=0.0)(student, teacher)
    hinged = GWRelationalLoss(cfg, margin=1.0)(student, teacher)

    assert hinged <= plain


# ---------------------------------------------------------------------------
# Zero-parameter feature adapter
# ---------------------------------------------------------------------------
def test_adapter_output_grid_matches_teacher():
    # CogVideoX-2B at 49x480x720: latent (13, 60, 90), patch 2 -> tokens (13, 30, 45).
    adapter = ZeroParamFeatureAdapter.for_cogvideox((1, 16, 13, 60, 90), patch_size=2)
    assert adapter.grid == (13, 30, 45)
    assert adapter.output_grid == (24, 10, 15)
    assert adapter.num_output_tokens == 3600


def test_adapter_forward_shape():
    adapter = ZeroParamFeatureAdapter(grid=(13, 30, 45))
    hidden = torch.randn(2, 13 * 30 * 45, 1920)
    assert adapter(hidden).shape == (2, 3600, 1920)


def test_adapter_rejects_inconsistent_token_count():
    adapter = ZeroParamFeatureAdapter(grid=(13, 30, 45))
    with pytest.raises(ValueError):
        adapter(torch.randn(2, 17, 1920))


def test_adapter_is_parameter_free():
    adapter = ZeroParamFeatureAdapter(grid=(13, 30, 45))
    assert sum(p.numel() for p in adapter.parameters()) == 0


def test_adapter_forwards_gradient():
    adapter = ZeroParamFeatureAdapter(grid=(13, 30, 45))
    hidden = torch.randn(1, 13 * 30 * 45, 8, requires_grad=True)
    adapter(hidden).sum().backward()
    assert hidden.grad is not None


def test_adapter_matches_average_pooling_reference():
    """The adapter must equal an explicit interpolate + avg_pool2d implementation."""
    torch.manual_seed(0)
    hidden = torch.randn(1, 13 * 30 * 45, 4)
    out = ZeroParamFeatureAdapter(grid=(13, 30, 45))(hidden)

    x = hidden.view(1, 13, 30, 45, 4).permute(0, 4, 1, 2, 3)  # (1, 4, 13, 30, 45)
    x = x[:, :, 1:]  # drop first frame -> 12
    x = F.interpolate(x, scale_factor=(2.0, 1.0, 1.0), mode="trilinear")  # -> 24
    x = x.permute(0, 2, 1, 3, 4).reshape(-1, 4, 30, 45)  # (24, 4, 30, 45)
    x = F.avg_pool2d(x, kernel_size=3, stride=3)  # -> (24, 4, 10, 15)
    x = x.reshape(1, 24, 4, 10, 15).permute(0, 1, 3, 4, 2).reshape(1, -1, 4)

    assert torch.allclose(out, x, atol=1e-6)

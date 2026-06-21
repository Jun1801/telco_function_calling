import math

from src.reward.scheduler import (
    ProgressiveRewardConfig,
    ProgressiveRewardScheduler,
    progressive,
    soft_only,
    strict_only,
)


def test_strict_weight_rises_from_low_to_high() -> None:
    sched = progressive()
    early = sched.strict_weight(0.0)
    late = sched.strict_weight(1.0)
    assert early < 0.4  # plan §9.7: ~30% strict at the start
    assert late > 0.85  # ~90% strict near the end
    assert early < late


def test_strict_weight_is_monotonic_non_decreasing() -> None:
    sched = progressive()
    weights = [sched.strict_weight(p / 10) for p in range(11)]
    assert all(b >= a - 1e-9 for a, b in zip(weights, weights[1:]))


def test_progress_is_clamped_to_unit_interval() -> None:
    sched = progressive()
    assert sched.strict_weight(-5.0) == sched.strict_weight(0.0)
    assert sched.strict_weight(9.0) == sched.strict_weight(1.0)


def test_blend_weights_soft_early_and_strict_late() -> None:
    sched = progressive()
    soft, strict = 1.0, 0.0
    early = sched.blend(soft, strict, progress=0.0)
    late = sched.blend(soft, strict, progress=1.0)
    # soft dominates early (reward stays high), strict dominates late (reward drops)
    assert early > late
    assert early > 0.6
    assert late < 0.15


def test_blend_at_step_matches_fractional_progress() -> None:
    sched = progressive()
    assert math.isclose(
        sched.blend_at_step(0.8, 0.2, step=50, total_steps=100),
        sched.blend(0.8, 0.2, progress=0.5),
    )


def test_blend_at_step_handles_zero_total() -> None:
    sched = progressive()
    # Degenerate total -> treat as final progress (fully strict-weighted end).
    assert math.isclose(
        sched.blend_at_step(0.8, 0.2, step=0, total_steps=0),
        sched.blend(0.8, 0.2, progress=1.0),
    )


def test_strict_only_ablation_ignores_soft() -> None:
    sched = strict_only()
    assert sched.strict_weight(0.0) == 1.0
    assert sched.strict_weight(1.0) == 1.0
    assert sched.blend(0.9, 0.1, progress=0.0) == 0.1


def test_soft_only_ablation_ignores_strict() -> None:
    sched = soft_only()
    assert sched.strict_weight(0.0) == 0.0
    assert sched.strict_weight(1.0) == 0.0
    assert sched.blend(0.9, 0.1, progress=1.0) == 0.9


def test_custom_config_bounds_are_respected() -> None:
    cfg = ProgressiveRewardConfig(lo_strict=0.2, hi_strict=0.8, midpoint=0.5, steepness=12.0)
    sched = ProgressiveRewardScheduler(cfg)
    assert math.isclose(sched.strict_weight(0.5), 0.5, abs_tol=1e-6)  # midpoint -> halfway
    assert 0.2 <= sched.strict_weight(0.0) <= 0.8
    assert 0.2 <= sched.strict_weight(1.0) <= 0.8

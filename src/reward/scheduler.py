"""Progressive soft-to-strict reward schedule (plan §9.7).

During training the reward shifts from forgiving partial-credit (soft) toward
binary correctness (strict):

    R_t = (1 - lambda_t) * R_soft + lambda_t * R_strict

where lambda_t (the weight on the strict reward) follows a bounded sigmoid that
rises from ~0.3 early to ~0.9 late. The bounds make the same class cover the
reward ablations in plan §15:

    strict_only()  -> lambda_t == 1  (binary reward from the start)
    soft_only()    -> lambda_t == 0  (partial credit throughout)
    progressive()  -> lambda_t sweeps 0.3 -> 0.9 (the M6 schedule)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class ProgressiveRewardConfig:
    """Bounds and shape of the strict-weight sigmoid.

    lo_strict / hi_strict: strict weight at the start / end of the sweep.
    midpoint:              training progress (0..1) where the sigmoid centers.
    steepness:             transition sharpness (higher = more abrupt).
    """

    lo_strict: float = 0.3
    hi_strict: float = 0.9
    midpoint: float = 0.35
    steepness: float = 10.0


class ProgressiveRewardScheduler:
    def __init__(self, config: ProgressiveRewardConfig | None = None) -> None:
        self.config = config or ProgressiveRewardConfig()

    def strict_weight(self, progress: float) -> float:
        """lambda_t: weight on the strict reward at the given training progress."""
        cfg = self.config
        span = cfg.hi_strict - cfg.lo_strict
        if span == 0.0:
            return cfg.lo_strict  # strict-only / soft-only ablations
        p = _clamp_unit(progress)
        return cfg.lo_strict + span * _sigmoid(cfg.steepness * (p - cfg.midpoint))

    def blend(self, reward_soft: float, reward_strict: float, progress: float) -> float:
        """R_t = (1 - lambda_t) * soft + lambda_t * strict."""
        lam = self.strict_weight(progress)
        return (1.0 - lam) * reward_soft + lam * reward_strict

    def blend_at_step(
        self, reward_soft: float, reward_strict: float, step: int, total_steps: int
    ) -> float:
        """Convenience: derive progress from a training step / total steps."""
        progress = 1.0 if total_steps <= 0 else step / total_steps
        return self.blend(reward_soft, reward_strict, progress)


def progressive() -> ProgressiveRewardScheduler:
    return ProgressiveRewardScheduler(ProgressiveRewardConfig())


def strict_only() -> ProgressiveRewardScheduler:
    return ProgressiveRewardScheduler(
        ProgressiveRewardConfig(lo_strict=1.0, hi_strict=1.0)
    )


def soft_only() -> ProgressiveRewardScheduler:
    return ProgressiveRewardScheduler(
        ProgressiveRewardConfig(lo_strict=0.0, hi_strict=0.0)
    )

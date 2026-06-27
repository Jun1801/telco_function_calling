"""LinUCB contextual bandit router for model selection.

Routes queries between two models (e.g. M1b and M6) using a
linear upper-confidence-bound algorithm with 6 binary context features.

Offline evaluation: both arms are pre-evaluated on all samples,
so the bandit operates in the full-information setting.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np


N_FEATURES = 6


def _features(sample: dict[str, Any]) -> np.ndarray:
    """6-dim binary context vector from sample metadata."""
    split = sample.get("split", "")
    family = sample.get("family", "")
    action = sample.get("expected_action", "")

    # Derive family from split name when not explicit (e.g. from results JSONL)
    if not family and split:
        for tag in ("parallel", "multi_step", "abstain", "missing_slot", "masked", "unseen"):
            if tag in split:
                family = tag
                break

    return np.array([
        float(action == "call_functions" or "parallel" in family),
        float("multi_step" in family),
        float("abstain" in family),
        float("missing_slot" in family),
        float("mask" in family),
        float("unseen" in family),
    ], dtype=np.float64)


class LinUCBBandit:
    """LinUCB contextual bandit for routing between two models.

    Args:
        arms: list of arm labels (e.g. ["m1b", "m6"])
        alpha: exploration coefficient (higher = more exploration)
    """

    def __init__(self, arms: list[str], alpha: float = 1.0) -> None:
        self.arms = arms
        self.alpha = alpha
        n = N_FEATURES
        self.A = [np.eye(n) for _ in arms]    # shape (n, n) per arm
        self.b = [np.zeros(n) for _ in arms]  # shape (n,) per arm

    def select(self, sample: dict[str, Any]) -> int:
        """Return arm index with highest UCB score."""
        x = _features(sample)
        ucbs = []
        for i in range(len(self.arms)):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            ucb = float(theta @ x) + self.alpha * float(np.sqrt(x @ A_inv @ x))
            ucbs.append(ucb)
        return int(np.argmax(ucbs))

    def update(self, arm: int, sample: dict[str, Any], reward: float) -> None:
        """Update arm statistics after observing reward."""
        x = _features(sample)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    # ------------------------------------------------------------------
    # Offline evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def oracle_accuracy(
        rewards_a: dict[str, float],
        rewards_b: dict[str, float],
    ) -> float:
        """Upper bound: always pick the better arm per sample."""
        ids = set(rewards_a) & set(rewards_b)
        if not ids:
            return 0.0
        return sum(max(rewards_a[i], rewards_b[i]) for i in ids) / len(ids)

    def simulate_offline(
        self,
        samples: list[dict[str, Any]],
        rewards_per_arm: list[dict[str, float]],
        seed: int = 42,
    ) -> dict[str, float]:
        """Simulate online LinUCB over pre-computed offline rewards.

        Returns dict with accuracy, oracle_accuracy, per-split breakdown.
        """
        rng = random.Random(seed)
        order = list(range(len(samples)))
        rng.shuffle(order)

        hits = 0
        total = 0
        split_hits: dict[str, list[float]] = {}

        for idx in order:
            s = samples[idx]
            sid = s["id"]
            available = [i for i, r in enumerate(rewards_per_arm) if sid in r]
            if len(available) < len(self.arms):
                continue

            arm = self.select(s)
            reward = rewards_per_arm[arm].get(sid, 0.0)
            self.update(arm, s, reward)

            hits += int(reward >= 1.0)
            total += 1

            split = s.get("split", "unknown")
            split_hits.setdefault(split, []).append(float(reward >= 1.0))

        accuracy = hits / total if total else 0.0
        oracle = self.oracle_accuracy(rewards_per_arm[0], rewards_per_arm[1])
        per_split = {k: sum(v) / len(v) for k, v in split_hits.items()}

        return {
            "linucb_accuracy": accuracy,
            "oracle_accuracy": oracle,
            "total": total,
            "per_split": per_split,
        }

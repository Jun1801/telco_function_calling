"""Offline bandit evaluation (M7).

Loads pre-computed rewards from M1b and M6 result JSONLs,
then simulates online LinUCB routing and reports accuracy.

Usage:
    python scripts/run_bandit_eval.py \
        --m1b-results reports/m1b_results_v2.jsonl \
        --m6-results  reports/m6b_results_v2.jsonl \
        --output      reports/m7_results_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.bandit_router import LinUCBBandit


def _load_rewards(path: Path) -> dict[str, float]:
    rewards: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record.get("id")
            if sid is None:
                continue
            reward = float(record.get("reward", record.get("reward_strict", 0.0)))
            rewards[sid] = reward
    return rewards


def _load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Reconstruct lightweight sample dict from result record
            samples.append({
                "id": record["id"],
                "split": record.get("split", ""),
                "family": record.get("split", ""),       # split name used as family proxy
                "expected_action": (record.get("prediction") or {}).get("action", ""),
            })
    return samples


def _heuristic_accuracy(
    samples: list[dict],
    rewards_a: dict[str, float],
    rewards_b: dict[str, float],
) -> float:
    """Always route to arm A (M1b) — naive baseline."""
    ids = {s["id"] for s in samples}
    matched = ids & set(rewards_a)
    if not matched:
        return 0.0
    return sum(rewards_a[sid] >= 1.0 for sid in matched) / len(matched)


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 offline bandit evaluation.")
    parser.add_argument("--m1b-results", required=True, help="Path to m1b_results_v2.jsonl")
    parser.add_argument("--m6-results", required=True, help="Path to m6b_results_v2.jsonl")
    parser.add_argument("--output", required=True, help="Output path for m7_results_v2.jsonl")
    parser.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration coefficient")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    m1b_path = Path(args.m1b_results)
    m6_path = Path(args.m6_results)

    if not m1b_path.exists():
        sys.exit(f"M1b results not found: {m1b_path}")
    if not m6_path.exists():
        sys.exit(f"M6 results not found: {m6_path}")

    print("Loading rewards...")
    rewards_m1b = _load_rewards(m1b_path)
    rewards_m6 = _load_rewards(m6_path)
    samples = _load_samples(m1b_path)

    common_ids = set(rewards_m1b) & set(rewards_m6)
    samples = [s for s in samples if s["id"] in common_ids]
    print(f"  M1b: {len(rewards_m1b)} samples  |  M6: {len(rewards_m6)} samples  |  overlap: {len(common_ids)}")

    # Oracle upper bound
    oracle_acc = LinUCBBandit.oracle_accuracy(rewards_m1b, rewards_m6)

    # Naive: always use M1b
    m1b_acc = sum(rewards_m1b[s["id"]] >= 1.0 for s in samples) / max(len(samples), 1)
    m6_acc = sum(rewards_m6[s["id"]] >= 1.0 for s in samples) / max(len(samples), 1)

    # LinUCB simulation
    bandit = LinUCBBandit(arms=["m1b", "m6"], alpha=args.alpha)
    results = bandit.simulate_offline(
        samples,
        rewards_per_arm=[rewards_m1b, rewards_m6],
        seed=args.seed,
    )

    print("\n=== M7 Bandit Routing Results ===")
    print(f"  Oracle (upper bound):  {oracle_acc*100:.1f}%")
    print(f"  M1b only:              {m1b_acc*100:.1f}%")
    print(f"  M6 only:               {m6_acc*100:.1f}%")
    print(f"  LinUCB (α={args.alpha}):       {results['linucb_accuracy']*100:.1f}%  (n={results['total']})")

    print("\nPer-split (LinUCB):")
    for split, acc in sorted(results["per_split"].items()):
        print(f"  {split:<30} {acc*100:.1f}%")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "oracle_accuracy": oracle_acc,
        "m1b_accuracy": m1b_acc,
        "m6_accuracy": m6_acc,
        "linucb_accuracy": results["linucb_accuracy"],
        "linucb_total": results["total"],
        "alpha": args.alpha,
        "per_split": results["per_split"],
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

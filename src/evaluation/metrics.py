from __future__ import annotations

from typing import Any


def argument_key_f1(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    pred_keys = set(predicted)
    gold_keys = set(gold)
    if not pred_keys and not gold_keys:
        return 1.0
    if not pred_keys or not gold_keys:
        return 0.0
    overlap = len(pred_keys & gold_keys)
    precision = overlap / len(pred_keys)
    recall = overlap / len(gold_keys)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def argument_value_accuracy(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0
    correct = 0
    for key, expected in gold.items():
        if predicted.get(key) == expected:
            correct += 1
    return correct / len(gold)

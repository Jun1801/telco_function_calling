"""Convert plan-format multi_step samples into ReAct turn-level records.

Each multi_step (gold_steps=[step1, step2-with-placeholder]) becomes TWO single-turn
call_function samples:
  R1: query -> call regional_station_info (step1)
  R2: query + [step1 observation] -> call step2 with the REAL station code (placeholder
      resolved from the synthesized observation)
This trains ReAct (observe->act) while reusing the single-turn SFT/eval pipeline.
Deterministic, local — no LLM.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.executor.kpi_mock import build_observation

STEP1_REF = "<from_step_1>"


def _dep_key(step2_args: dict) -> str | None:
    for k, v in step2_args.items():
        if STEP1_REF in str(v):
            return k
    return None


def _decompose(sample: dict, catalogue: list, idx: int) -> list[dict]:
    steps = sample.get("gold_steps") or []
    if len(steps) < 2:
        return []
    step1, step2 = steps[0], steps[1]
    dep_key = _dep_key(step2.get("arguments", {}))
    if dep_key is None:
        return []
    obs = build_observation(step1, catalogue, seed=idx)
    if not obs["stations"]:
        return []
    code = obs["stations"][0]["station_code"]

    base = {"source": "real_tool_xlsx", "customer_verified": True, "generator": "react",
            "scenario_family": "multi_step", "expected_action": "call_function"}
    r1 = {**base, "id": f"{sample['id']}_r1", "split": sample["split"], "scenario": "react_step1",
          "instruction": sample["instruction"],
          "gold_call": {"tool_name": step1["tool_name"], "arguments": copy.deepcopy(step1["arguments"])}}

    s2_args = copy.deepcopy(step2["arguments"]); s2_args[dep_key] = code
    obs_block = ("\n\n[Kết quả bước 1 — danh sách trạm]:\n"
                 + json.dumps(obs["stations"], ensure_ascii=False)
                 + "\nHãy dùng kết quả trên để tra cứu tiếp.")
    r2 = {**base, "id": f"{sample['id']}_r2", "split": sample["split"], "scenario": "react_step2",
          "instruction": sample["instruction"] + obs_block,
          "gold_call": {"tool_name": step2["tool_name"], "arguments": s2_args}}
    return [r1, r2]


def _convert_file(path: Path, target_records: int, rng: random.Random) -> tuple[list, list]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    others = [r for r in rows if r.get("scenario_family") != "multi_step"]
    multi = [r for r in rows if r.get("scenario_family") == "multi_step"]
    rng.shuffle(multi)
    n_src = max(0, target_records // 2)
    react = []
    for i, s in enumerate(multi):
        if len(react) >= target_records:
            break
        react += _decompose(s, _CATALOGUE, i)
    return others, react


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-train", type=int, default=520, help="ReAct records in train (~2x source).")
    ap.add_argument("--target-eval", type=int, default=156)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = ROOT / "data"
    global _CATALOGUE
    _CATALOGUE = json.loads((data / "real_station_catalogue.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    # train: replace multi_step with react, keep other families
    others, react_tr = _convert_file(data / "sft_train_real.jsonl", args.target_train, rng)
    train = others + react_tr
    with (data / "sft_train_real.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # eval_real_multi_step: react only
    _, react_ev = _convert_file(data / "eval_real_multi_step.jsonl", args.target_eval, rng)
    for r in react_ev:
        r["split"] = "eval_real_multi_step"
    with (data / "eval_real_multi_step.jsonl").open("w", encoding="utf-8") as f:
        for r in react_ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import collections
    print(f"train: {len(train)} ({len(react_tr)} react multi_step + {len(others)} other)")
    print("  react split:", dict(collections.Counter(r["scenario"] for r in react_tr)))
    print(f"eval_real_multi_step: {len(react_ev)} ({dict(collections.Counter(r['scenario'] for r in react_ev))})")


if __name__ == "__main__":
    main()

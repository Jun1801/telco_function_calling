"""Regenerate ONLY the missing_slot family with the bug-9 date-pair fix and splice
it into the existing real dataset, leaving every other family untouched.

Why: gen_missing_slot now drops the from_date/to_date pair together. The committed
missing_slot data was produced by the old code (one date declared missing while the
query dropped the whole period) and must be refreshed. missing_slot is independent of
all other families (masking derives from single/eval_seen, not missing) and the
verifier skips the semantic layer for it, so this is a clean, isolated regen.

  python3.11 scripts/regen_missing_slot.py --count 90
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.real_tool_llm_generator import RealToolLLMGenerator
from src.generation.real_tool_verifier import DualLayerRealVerifier

DATA = ROOT / "data"
EVAL_MISSING = DATA / "eval_real_missing_slot.jsonl"
TRAIN = DATA / "sft_train_real.jsonl"
EVAL_HOLD = 240  # matches DEFAULTS["eval_missing"] in generate_real_data.py


def _write_atomic(path: Path, rows: list[dict]) -> None:
    """Shadow-file write: build the new file, then replace, preserving git history."""
    shadow = path.with_suffix(path.suffix + ".shadow")
    with shadow.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shadow.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=90, help="raw missing samples per seen tool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-tools", type=int, default=0, help="0 = all seen tools (debug only)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tools = json.loads((DATA / "real_tools.json").read_text(encoding="utf-8"))
    refs = json.loads((DATA / "real_reference_codes.json").read_text(encoding="utf-8"))
    stations = json.loads((DATA / "real_station_catalogue.json").read_text(encoding="utf-8"))
    tbn = {t["name"]: t for t in tools}
    seen = [t for t in tools if t["split"] == "seen"]
    if args.limit_tools:
        seen = seen[: args.limit_tools]

    try:
        import mlx.core as mx
        mx.random.seed(args.seed)
    except ImportError:
        pass

    print(f"Loading generator (Qwen3-4B) ... seen tools: {len(seen)}")
    gen = RealToolLLMGenerator(references=refs, stations=stations)
    # semantic layer is skipped for missing_slot anyway → off for speed/determinism.
    ver = DualLayerRealVerifier(tbn, references=refs, stations=stations,
                                generator=gen, run_semantic=False)

    print(f"Generating missing_slot ({args.count}/tool) ...")
    missing: list[dict] = []
    for t in seen:
        missing += gen.gen_missing_slot(t, args.count)
    missing = ver.run(missing)
    print(f"  verified missing_slot: {len(missing)}  | verifier stats: {ver.stats}")

    rng.shuffle(missing)
    eval_missing, train_missing = missing[:EVAL_HOLD], missing[EVAL_HOLD:]
    for s in eval_missing:
        s["split"] = "eval_real_missing_slot"
    for s in train_missing:
        s["split"] = "train"

    if len(eval_missing) < EVAL_HOLD:
        sys.exit(f"ABORT: only {len(missing)} verified < eval hold {EVAL_HOLD}. "
                 f"Re-run with a larger --count.")
    if len(train_missing) < 400:
        print(f"  WARNING: train_missing={len(train_missing)} < 400 (baseline 432). "
              f"Consider --count {args.count + 30}.")

    # ---- splice: keep every non-missing line, swap in fresh missing_slot ----
    kept = [json.loads(l) for l in TRAIN.open(encoding="utf-8") if l.strip()]
    non_missing = [s for s in kept if s.get("scenario_family") != "missing_slot"]
    old_missing = len(kept) - len(non_missing)
    new_train = non_missing + train_missing

    _write_atomic(EVAL_MISSING, eval_missing)
    _write_atomic(TRAIN, new_train)

    fam = Counter(s.get("scenario_family") for s in new_train)
    print("\n=== SPLICE DONE ===")
    print(f"  eval_real_missing_slot.jsonl: {len(eval_missing)} (was {EVAL_HOLD})")
    print(f"  sft_train_real.jsonl missing_slot: {old_missing} -> {len(train_missing)}")
    print(f"  train total: {len(kept)} -> {len(new_train)}")
    print(f"  train by family: {dict(sorted(fam.items()))}")


if __name__ == "__main__":
    main()

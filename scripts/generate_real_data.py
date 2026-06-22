"""Generate scaled, clean SFT + eval data from real KPI tools (ToolACE-adapted).

Construct-then-paraphrase (gold from catalogue) + Dual-Layer Verification.
Targets ~2500 train / ~1200 test across §7.2-adapted families.

Outputs:
  data/sft_train_real.jsonl
  data/eval_real_{seen,unseen,masked,missing_slot,multi_step,parallel,abstain}.jsonl
  reports/real_data_generation.json
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

PARALLEL_PAIRS = [
    ("kqi_province", "download_throughput_oss"),
    ("vung_lom_all", "pakh_all"),
    ("tram_nha_mang_khac_province", "vung_phu_province"),
    ("kqi_province", "sub_attached_all"),           # KQI + thuê bao, share location+time
    ("download_throughput_oss", "speedtest_province"),  # 2 nguồn throughput cùng tỉnh
    ("thong_ke_kpi", "nguong_kpi"),                 # thống kê + ngưỡng, share kpi_code
]
MULTI_STEP_CHAINS = [
    ("regional_station_info", "sub_attached_station", "station_code"),
    ("regional_station_info", "radio_traffic", "object_code"),
    ("regional_station_info", "alarm_count", "object_code"),
    ("regional_station_info", "top_port", "object_code"),
    ("regional_station_info", "alarm_unresolved", "object_code"),  # trạm → alarm chưa xử lý
]

# Per-unit generation counts at scale=1.0 (before verify/split). Over-generated to
# absorb verification drops; final ≈ train 2500 / test 1200.
DEFAULTS = dict(seen_single=90, unseen_single=26, missing=90, multi_per_chain=300,
                parallel_per_pair=110, abstain=980,
                eval_seen=250, eval_unseen=150, eval_missing=240, eval_multi=155,
                eval_parallel=80, eval_abstain=155, mask_train=350, mask_eval_seen=110, mask_eval_unseen=60)


def _write(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _hold(rows, k, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:k], rows[k:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiplier for raw generation counts (train families + mask_train)")
    ap.add_argument("--eval-scale", type=float, default=1.0,
                    help="Multiplier for eval hold-out targets (default 1.0 = fixed at DEFAULTS)")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--limit-tools", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _EVAL_KEYS = {"eval_seen", "eval_unseen", "eval_missing", "eval_multi",
                  "eval_parallel", "eval_abstain", "mask_eval_seen", "mask_eval_unseen"}
    C = {k: int(v * (args.eval_scale if k in _EVAL_KEYS else args.scale))
         for k, v in DEFAULTS.items()}
    rng = random.Random(args.seed)

    data = ROOT / "data"
    tools = json.loads((data / "real_tools.json").read_text(encoding="utf-8"))
    refs = json.loads((data / "real_reference_codes.json").read_text(encoding="utf-8"))
    stations = json.loads((data / "real_station_catalogue.json").read_text(encoding="utf-8"))
    tbn = {t["name"]: t for t in tools}
    seen = [t for t in tools if t["split"] == "seen"]
    unseen = [t for t in tools if t["split"] == "unseen"]
    if args.limit_tools:
        seen = seen[: args.limit_tools]
        unseen = unseen[: max(1, args.limit_tools // 2)]

    try:
        import mlx.core as mx
        mx.random.seed(args.seed)
    except ImportError:
        pass

    print(f"Loading generator (scale={args.scale}, eval-scale={args.eval_scale}) ...")
    gen = RealToolLLMGenerator(references=refs, stations=stations)
    ver = DualLayerRealVerifier(tbn, references=refs, stations=stations,
                                generator=gen, run_semantic=not args.no_semantic)

    # ---- generate + verify per family ----
    print("single (seen) ...")
    seen_single = []
    for t in seen:
        seen_single += gen.gen_single_step(t, C["seen_single"])
    seen_single = ver.run(seen_single)

    print("single (unseen) ...")
    unseen_single = []
    for t in unseen:
        unseen_single += gen.gen_single_step(t, C["unseen_single"])
    unseen_single += gen.from_seed_examples(unseen)
    unseen_single = ver.run(unseen_single)

    print("missing_slot (seen) ...")
    missing = []
    for t in seen:
        missing += gen.gen_missing_slot(t, C["missing"])
    missing = ver.run(missing)

    print("parallel ...")
    pairs = [(tbn[a], tbn[b]) for a, b in PARALLEL_PAIRS if a in tbn and b in tbn]
    parallel = ver.run(gen.gen_parallel(pairs, C["parallel_per_pair"]))

    print("multi_step ...")
    chains = [(tbn[s], tbn[d], k) for s, d, k in MULTI_STEP_CHAINS if s in tbn and d in tbn]
    multi = ver.run(gen.gen_multi_step(chains, C["multi_per_chain"]))

    print("abstain ...")
    abstain = ver.run(gen.gen_abstain(C["abstain"]))

    # ---- split test/train (eval held out first, leak-free) ----
    eval_seen, train_single = _hold(seen_single, C["eval_seen"], rng)
    eval_unseen = unseen_single[: C["eval_unseen"]]
    eval_missing, train_missing = _hold(missing, C["eval_missing"], rng)
    eval_multi, train_multi = _hold(multi, C["eval_multi"], rng)
    eval_parallel, train_parallel = _hold(parallel, C["eval_parallel"], rng)
    eval_abstain, train_abstain = _hold(abstain, C["abstain"] and C["eval_abstain"], rng)

    print("masking (derive) ...")
    mask_train = ver.run(gen.gen_masking(train_single, tbn, C["mask_train"]))
    mask_eval = ver.run(gen.gen_masking(eval_seen, tbn, C["mask_eval_seen"])
                        + gen.gen_masking(eval_unseen, tbn, C["mask_eval_unseen"]))

    # ---- assemble + tag splits ----
    train = train_single + train_missing + train_multi + train_parallel + train_abstain + mask_train
    for s in train:
        s["split"] = "train"
    eval_files = {
        "eval_real_seen": eval_seen, "eval_real_unseen": eval_unseen, "eval_real_masked": mask_eval,
        "eval_real_missing_slot": eval_missing, "eval_real_multi_step": eval_multi,
        "eval_real_parallel": eval_parallel, "eval_real_abstain": eval_abstain,
    }
    for name, rows in eval_files.items():
        for s in rows:
            s["split"] = name

    _write(data / "sft_train_real.jsonl", train)
    for name, rows in eval_files.items():
        _write(data / f"{name}.jsonl", rows)

    report = {
        "train_total": len(train),
        "train_by_family": dict(Counter(s["scenario_family"] for s in train)),
        "eval_total": sum(len(r) for r in eval_files.values()),
        "eval_counts": {k: len(v) for k, v in eval_files.items()},
        "verifier_stats": ver.stats,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "real_data_generation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

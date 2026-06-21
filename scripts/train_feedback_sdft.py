"""
Phase 4 — Feedback-SDFT training.

Trains on correction pairs (corrections.jsonl), optionally mixed with
domain augmented data, starting from the M1 (domain-aug) adapter.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    args = _parse_args()
    _require_mlx()

    data_dir = _prepare_mlx_data(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iters = args.iters or _count_iters(args, data_dir / "train.jsonl")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", str(data_dir),
        "--iters", str(iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.lora_layers),
        "--learning-rate", str(args.learning_rate),
        "--adapter-path", str(output_dir),
        "--val-batches", str(args.val_batches),
        "--steps-per-report", str(args.steps_per_report),
        "--steps-per-eval", str(args.steps_per_eval),
        "--save-every", str(args.save_every),
        "--seed", str(args.seed),
        "--grad-checkpoint",
    ]
    if args.mask_prompt:
        # Train only on the correct action JSON, not the prompt/feedback context.
        cmd.append("--mask-prompt")
    if args.resume_from:
        resume_file = Path(args.resume_from) / "adapters.safetensors"
        if resume_file.exists():
            cmd += ["--resume-adapter-file", str(resume_file)]
            print(f"  Resuming from: {resume_file}")
        else:
            print(f"  WARNING: resume file not found: {resume_file}")

    print("Command:", " ".join(cmd))
    print(f"Training: {iters} iters on {data_dir / 'train.jsonl'}")
    subprocess.run(cmd, check=True)
    print(f"\nAdapter saved to: {output_dir}")
    print(f"\nEval with:")
    print(f"  python3.11 scripts/run_baseline.py --backend mlx --model {args.model} --adapter {output_dir}")


def _prepare_mlx_data(args: argparse.Namespace) -> Path:
    corrections_path = Path(args.corrections)
    out_dir = Path(args.mlx_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corrections = [json.loads(l) for l in corrections_path.open() if l.strip()]
    print(f"  Corrections: {len(corrections)} samples")

    train_samples = corrections[:]

    # Optionally mix in domain augmented data to avoid catastrophic forgetting
    if args.mix_domain:
        domain_path = Path(args.mix_domain)
        if domain_path.exists():
            domain = [json.loads(l) for l in domain_path.open() if l.strip()]
            train_samples.extend(domain)
            print(f"  + Domain mix: {len(domain)} samples")

    # Write train.jsonl (messages only, mlx-lm format)
    train_out = out_dir / "train.jsonl"
    with train_out.open("w") as f:
        for s in train_samples:
            f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")
    print(f"  train.jsonl: {len(train_samples)} samples → {train_out}")

    # Use sft_eval.jsonl for validation
    eval_path = ROOT / "data" / "sft_eval.jsonl"
    valid_out = out_dir / "valid.jsonl"
    if eval_path.exists():
        eval_samples = [json.loads(l) for l in eval_path.open() if l.strip()]
        with valid_out.open("w") as f:
            for s in eval_samples:
                f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")
        print(f"  valid.jsonl: {len(eval_samples)} samples → {valid_out}")
    else:
        # Fallback: use last 10% of train as validation
        split = max(1, len(train_samples) // 10)
        valid_samples = train_samples[-split:]
        with valid_out.open("w") as f:
            for s in valid_samples:
                f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")
        print(f"  valid.jsonl: {len(valid_samples)} samples (fallback split)")

    return out_dir


def _count_iters(args: argparse.Namespace, train_file: Path) -> int:
    n = sum(1 for l in train_file.open() if l.strip())
    return max(1, round(n * args.epochs / args.batch_size))


def _require_mlx() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        raise SystemExit("mlx-lm not found. Run with python3.11.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4: Feedback-SDFT training on correction pairs.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corrections", default=str(ROOT / "data" / "corrections.jsonl"),
                   help="Correction pairs from build_corrections.py.")
    p.add_argument("--mix-domain", default=str(ROOT / "data" / "sft_train_augmented.jsonl"),
                   help="Optional domain data to mix in (prevents forgetting). Set to '' to disable.")
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "sft_mlx" / "qwen3-4b-feedback-sdft"))
    p.add_argument("--mlx-data-dir", default=str(ROOT / "data" / "mlx_sft_feedback"))
    p.add_argument("--resume-from", default=str(ROOT / "outputs" / "sft_mlx" / "qwen3-4b-domain-aug"),
                   help="M1 adapter to start from.")
    p.add_argument("--epochs", type=float, default=3.0,
                   help="Epochs over correction data (more epochs OK — small dataset).")
    p.add_argument("--iters", type=int, default=0,
                   help="Override epochs with exact iter count.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lora-layers", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="Lower LR than SFT (1e-5) to fine-tune on corrections without overwriting M1.")
    p.add_argument("--val-batches", type=int, default=5)
    p.add_argument("--steps-per-report", type=int, default=5)
    p.add_argument("--steps-per-eval", type=int, default=20)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-prompt", dest="mask_prompt", action="store_true", default=True,
                   help="Mask prompt tokens in the loss (train only on the action JSON). Default on.")
    p.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false",
                   help="Disable prompt masking (train on full sequence).")
    return p.parse_args()


if __name__ == "__main__":
    main()

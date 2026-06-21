from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    args = parse_args()
    _require_mlx()

    data_dir = _prepare_mlx_data(
        train_file=Path(args.train_file),
        eval_file=Path(args.eval_file),
        out_dir=Path(args.mlx_data_dir),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iters = args.iters or _epochs_to_iters(args.epochs, args.batch_size, data_dir / "train.jsonl")

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
        # Only backprop the assistant action JSON; the large schema blob in
        # system/user turns would otherwise dilute the gradient.
        cmd.append("--mask-prompt")
    if args.resume_from:
        resume_file = Path(args.resume_from) / "adapters.safetensors"
        if resume_file.exists():
            cmd += ["--resume-adapter-file", str(resume_file)]
            print(f"  Resuming from: {resume_file}")
    print("Command:", " ".join(cmd))
    print(f"Training: {iters} iters  (~{iters * args.batch_size} samples seen)")
    subprocess.run(cmd, check=True)
    print(f"\nAdapter saved to: {output_dir}")
    print(f"\nEval with:")
    print(f"  python scripts/run_baseline.py --backend mlx --model {args.model} --adapter {output_dir}")


def _require_mlx() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        raise SystemExit(
            "mlx-lm not found. Install with:\n"
            "  pip install mlx-lm\n"
            "Requires macOS with Apple Silicon (M-series)."
        )


def _prepare_mlx_data(train_file: Path, eval_file: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, dst_name in [(train_file, "train.jsonl"), (eval_file, "valid.jsonl")]:
        records = [json.loads(l) for l in src.open() if l.strip()]
        dst = out_dir / dst_name
        with dst.open("w") as f:
            for r in records:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        print(f"  {dst_name}: {len(records)} samples → {dst}")
    return out_dir


def _epochs_to_iters(epochs: float, batch_size: int, train_file: Path) -> int:
    n = sum(1 for l in train_file.open() if l.strip())
    return max(1, round(n * epochs / batch_size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX-LM LoRA SFT for Telco function-calling (Apple Silicon).")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="3-4B models fit M3 Pro 18GB for training. 7B requires Colab/CUDA.")
    parser.add_argument("--train-file", default=str(ROOT / "data" / "sft_train_with_warmup.jsonl"))
    parser.add_argument("--eval-file", default=str(ROOT / "data" / "sft_eval.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "sft_mlx" / "qwen2.5-3b"))
    parser.add_argument("--mlx-data-dir", default=str(ROOT / "data" / "mlx_sft"),
                        help="Directory where mlx-lm train.jsonl/valid.jsonl are written.")
    parser.add_argument("--epochs", type=float, default=3.0,
                        help="Converted to --iters automatically unless --iters is set.")
    parser.add_argument("--iters", type=int, default=0,
                        help="Override epochs with exact iteration count.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-layers", type=int, default=8,
                        help="Number of transformer layers to apply LoRA to (from the top).")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default=None,
                        help="Path to an existing adapter dir to resume/continue training from.")
    parser.add_argument("--mask-prompt", dest="mask_prompt", action="store_true", default=True,
                        help="Mask prompt tokens in the loss (train only on the action JSON). Default on.")
    parser.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false",
                        help="Disable prompt masking (train on full sequence).")
    return parser.parse_args()


if __name__ == "__main__":
    main()

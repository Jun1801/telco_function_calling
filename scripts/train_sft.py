from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    args = parse_args()
    _require_training_packages(args.load_in_4bit)

    import torch
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = _load_sft_dataset(args.train_file, tokenizer, args.max_train_samples)
    eval_dataset = _load_sft_dataset(args.eval_file, tokenizer, args.max_eval_samples) if args.eval_file else None

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.adapter:
        from peft import PeftModel
        print(f"Resuming from adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        peft_config = None  # LoRA already applied
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(","),
        )

    trainer = _build_trainer(
        SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        training_args=_build_sft_config(SFTConfig, args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved_adapter: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal QLoRA SFT for Telco function-calling.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train-file", default=str(ROOT / "data" / "sft_train.jsonl"))
    parser.add_argument("--eval-file", default=str(ROOT / "data" / "sft_eval.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "sft" / "qwen2.5-coder-7b"))
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules for Qwen-style decoder blocks.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--adapter", default=None, help="Path to existing LoRA adapter to resume from.")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _require_training_packages(load_in_4bit: bool) -> None:
    packages = ["torch", "datasets", "peft", "transformers", "trl", "accelerate"]
    if load_in_4bit:
        packages.append("bitsandbytes")
    missing = []
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            "Missing training packages: "
            + ", ".join(missing)
            + "\nInstall in Colab with: pip install -U transformers datasets peft trl bitsandbytes accelerate"
        )


def _load_sft_dataset(path: str | Path, tokenizer: Any, limit: int | None = None) -> Any:
    from datasets import Dataset

    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            text = _render_messages(tokenizer, row["messages"])
            records.append({"text": text, "id": row["id"], "expected_action": row["expected_action"]})
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No SFT records found in {path}")
    return Dataset.from_list(records)


def _render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _build_sft_config(config_cls: Any, args: argparse.Namespace) -> Any:
    raw_kwargs = {
        "output_dir": args.output_dir,
        "dataset_text_field": "text",
        "max_length": args.max_seq_length,
        "packing": False,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_total_limit": 2,
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "seed": args.seed,
    }
    signature = inspect.signature(config_cls)
    kwargs = {key: value for key, value in raw_kwargs.items() if key in signature.parameters}
    return config_cls(**kwargs)


def _build_trainer(
    trainer_cls: Any,
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    peft_config: Any,
    training_args: Any,
) -> Any:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        kwargs["peft_config"] = peft_config
    try:
        return trainer_cls(**kwargs)
    except TypeError:
        kwargs.pop("processing_class", None)
        kwargs["tokenizer"] = tokenizer
        return trainer_cls(**kwargs)


if __name__ == "__main__":
    main()

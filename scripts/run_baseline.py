from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_prediction
from src.executor.mock_telco_api import MockTelcoApi
from src.model.output_parser import parse_model_output
from src.model.prompt_builder import build_prompt_messages
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-only baseline.")
    parser.add_argument("--backend", default="mock_oracle", choices=["mock_oracle", "mock_malformed", "transformers"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", help="Optional PEFT/LoRA adapter path to evaluate with the transformers backend.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, help="Optional max number of eval samples.")
    parser.add_argument("--enable-thinking", action="store_true", help="Pass enable_thinking=True when the tokenizer supports it.")
    parser.add_argument("--output", default=str(ROOT / "reports" / "prompt_only_results.jsonl"))
    parser.add_argument("--error-report", default=str(ROOT / "reports" / "error_analysis.md"))
    args = parser.parse_args()

    data_dir = ROOT / "data"
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tool_registry = ToolRegistry.from_file(data_dir / "tools.json")
    contract_registry = ContractRegistry.from_file(data_dir / "tool_contracts.json")
    generator = _build_generator(args)

    records = []
    for sample in _iter_eval_samples(data_dir)[: args.limit]:
        prompt = build_prompt_messages(sample, tool_registry, contract_registry)
        raw_output = generator.generate(sample, prompt)
        prediction = parse_model_output(raw_output)
        result = evaluate_prediction(
            sample,
            prediction,
            tool_registry,
            MockTelcoApi.from_file(data_dir / "mock_telco_db.json"),
            contract_registry,
        )
        records.append(
            {
                "id": sample["id"],
                "split": sample["split"],
                "scenario": sample["scenario"],
                "backend": args.backend,
                "prompt": prompt,
                "raw_output": raw_output,
                "prediction": prediction,
                "reward": result["reward_total"],
                "feedback": result["feedback"],
                "metrics": result["metrics"],
            }
        )

    with report_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_error_report(Path(args.error_report), records)
    print(f"records: {len(records)}")
    print(f"output: {report_path}")
    print(f"error_report: {args.error_report}")


def _build_generator(args: argparse.Namespace) -> Any:
    if args.backend in {"mock_oracle", "mock_malformed"}:
        return MockGenerator(args.backend)
    return TransformersGenerator(
        model_name=args.model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
        enable_thinking=args.enable_thinking,
    )


class MockGenerator:
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def generate(self, sample: dict[str, Any], prompt: list[dict[str, str]]) -> str:
        return _mock_model_output(sample, self.backend)


class TransformersGenerator:
    def __init__(
        self,
        model_name: str,
        adapter_path: str | None,
        max_new_tokens: int,
        temperature: float,
        load_in_4bit: bool,
        enable_thinking: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise SystemExit(
                "The transformers backend requires torch, transformers, and bitsandbytes for 4-bit loading. "
                "Install them in Colab before running this backend."
            ) from error

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise SystemExit("Evaluating a LoRA adapter requires peft. Install peft in the runtime.") from error
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def generate(self, sample: dict[str, Any], prompt: list[dict[str, str]]) -> str:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }
        try:
            text = self.tokenizer.apply_chat_template(prompt, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            text = self.tokenizer.apply_chat_template(prompt, **template_kwargs)

        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            generation_kwargs.update({"do_sample": False})

        with self.torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated = output_ids[0][prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def _iter_eval_samples(data_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("eval_*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            samples.extend(json.loads(line) for line in file if line.strip())
    return samples


def _mock_model_output(sample: dict[str, Any], backend: str) -> str:
    if backend == "mock_malformed":
        return "I think you should call the right tool, but this is not JSON."
    if sample["expected_action"] == "call_functions":
        payload = {"action": "call_functions", "calls": sample.get("gold_steps") or sample.get("gold_calls") or []}
    elif sample["expected_action"] == "call_function":
        payload = {"action": "call_function", "call": sample["gold_call"]}
    elif sample["expected_action"] == "ask_clarification":
        payload = {"action": "ask_clarification", "asked_slots": sample.get("missing_slots", [])}
    else:
        payload = {"action": "abstain", "reason": sample.get("prediction", {}).get("reason", "unsafe request")}
    return json.dumps(payload, ensure_ascii=False)


def _write_error_report(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    strict_success = sum(1 for item in records if item["reward"] == 1.0)
    parse_errors = sum(1 for item in records if item["prediction"].get("parse_error"))
    by_split: dict[str, list[float]] = {}
    for item in records:
        by_split.setdefault(item["split"], []).append(item["reward"])
    lines = [
        "# Prompt-Only Baseline Error Analysis",
        "",
        f"- total_records: {total}",
        f"- strict_success: {strict_success}/{total}",
        f"- parse_errors: {parse_errors}",
        "",
        "## Reward By Split",
        "",
    ]
    for split, rewards in sorted(by_split.items()):
        avg_reward = sum(rewards) / len(rewards)
        lines.append(f"- {split}: {avg_reward:.3f} over {len(rewards)} records")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

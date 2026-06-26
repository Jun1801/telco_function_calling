"""SDPO offline distillation training — HF Transformers port.

Algorithm:
  Teacher context = prompt + feedback + best sibling demo
  Student context = prompt only
  Loss = JSD(student || teacher) on response tokens, top-k=20
  IS correction = clip(student_lp / rollout_lp, max=is_clip)

Phase A: precompute teacher top-k logits → cache to data/sdpo_cache.jsonl
Phase B: train student with JSD loss using cached logits

python src/training/train_sdpo_hf.py \
  --rollouts data/sdpo_rollouts_m4.jsonl \
  --model /workspace/models/Qwen3-4B \
  --teacher-adapter outputs/sft/m1b_qwen3-4b \
  --student-resume outputs/sft/m1b_qwen3-4b \
  --output-dir outputs/sft/m4b_qwen3-4b \
  --top-k 20 --alpha 0.5 --is-clip 2.0 \
  --learning-rate 5e-6 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing import REAL_SOURCE
from src.reward.feedback_renderer import render_teacher_feedback

SDPO_TEACHER_SUFFIX = (
    "\n\n[SDPO: Previous Attempt]\n{best_response}"
    "\n\n[SDPO: Environment Feedback]\n{feedback_text}"
)


def _format_feedback(feedback: dict, lang: str = "vi") -> str:
    if feedback.get("errors"):
        return render_teacher_feedback(feedback, lang)
    texts = feedback.get("feedback_text", [])
    return " ".join(texts) if texts else "The response was incorrect."


def _build_teacher_messages(record: dict) -> list[dict] | None:
    """Build feedback-conditioned teacher context (prompt + best demo + feedback)."""
    rollouts = record["rollouts"]
    best_idx = record.get("best_idx", -1)
    lang = "vi" if record.get("source") == REAL_SOURCE else "en"

    if best_idx < 0 or best_idx >= len(rollouts):
        best_idx = max(range(len(rollouts)), key=lambda j: rollouts[j]["reward"])

    best_response = json.dumps(rollouts[best_idx]["prediction"], ensure_ascii=False)
    raw_out = rollouts[best_idx].get("raw_output", "").strip()
    if raw_out:
        best_response = raw_out
    feedback_text = _format_feedback(rollouts[best_idx]["feedback"], lang)

    original_msgs = record["prompt_messages"]
    if not original_msgs or original_msgs[0]["role"] != "system":
        return None

    teacher_system = original_msgs[0]["content"] + SDPO_TEACHER_SUFFIX.format(
        best_response=best_response, feedback_text=feedback_text
    )
    return [{"role": "system", "content": teacher_system}] + original_msgs[1:]


# ---------------------------------------------------------------------------
# Phase A — precompute teacher top-k logits
# ---------------------------------------------------------------------------

def compute_teacher_logits(
    records: list[dict],
    model: Any,
    tokenizer: Any,
    top_k: int,
    success_threshold: float,
) -> list[dict]:
    import torch
    import torch.nn.functional as F

    dataset = []
    for idx, record in enumerate(records):
        best_idx = record.get("best_idx", -1)
        rollouts = record["rollouts"]
        if best_idx < 0 or best_idx >= len(rollouts):
            best_idx = max(range(len(rollouts)), key=lambda j: rollouts[j]["reward"])

        if rollouts[best_idx]["reward"] < success_threshold:
            continue

        raw_out = rollouts[best_idx].get("raw_output", "").strip()
        if not raw_out:
            raw_out = json.dumps(rollouts[best_idx]["prediction"], ensure_ascii=False)
        target_response = raw_out

        student_msgs = record["prompt_messages"]
        teacher_msgs = _build_teacher_messages(record)
        if teacher_msgs is None:
            teacher_msgs = student_msgs

        eos = tokenizer.eos_token or ""

        student_text = tokenizer.apply_chat_template(
            student_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        teacher_text = tokenizer.apply_chat_template(
            teacher_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        response_text = target_response + eos

        student_ids = tokenizer(student_text, return_tensors="pt")["input_ids"][0]
        teacher_ids = tokenizer(teacher_text, return_tensors="pt")["input_ids"][0]
        response_ids = tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

        n_resp = len(response_ids)
        if n_resp == 0:
            continue

        dev = next(model.parameters()).device

        # Teacher forward (feedback-conditioned context)
        teacher_full = torch.cat([teacher_ids, response_ids]).unsqueeze(0).to(dev)
        with torch.no_grad():
            teacher_out = model(teacher_full)
        teacher_logits = teacher_out.logits[0]  # (T, V)
        resp_start_t = len(teacher_ids) - 1
        teacher_resp_lp = F.log_softmax(
            teacher_logits[resp_start_t: resp_start_t + n_resp], dim=-1
        )  # (n_resp, V)
        topk_lp, topk_idx = torch.topk(teacher_resp_lp, top_k, dim=-1)

        # Rollout log-probs for IS correction (teacher on student context)
        student_full = torch.cat([student_ids, response_ids]).unsqueeze(0).to(dev)
        with torch.no_grad():
            student_out = model(student_full)
        student_logits = student_out.logits[0]
        resp_start_s = len(student_ids) - 1
        student_resp_lp = F.log_softmax(
            student_logits[resp_start_s: resp_start_s + n_resp], dim=-1
        )  # (n_resp, V)
        rollout_lp = student_resp_lp.gather(
            1, response_ids.to(dev).unsqueeze(1)
        ).squeeze(1)  # (n_resp,)

        dataset.append({
            "id": record["id"],
            "student_ids": student_ids.tolist(),
            "response_ids": response_ids.tolist(),
            "n_response": n_resp,
            "teacher_topk_idx": topk_idx.cpu().tolist(),
            "teacher_topk_lp": topk_lp.cpu().tolist(),
            "rollout_log_probs": rollout_lp.cpu().tolist(),
            "reward": rollouts[best_idx]["reward"],
        })

        if (idx + 1) % 20 == 0:
            print(f"  Phase A [{idx+1}/{len(records)}] cache={len(dataset)}", flush=True)

    return dataset


# ---------------------------------------------------------------------------
# Phase B — JSD distillation training
# ---------------------------------------------------------------------------

def compute_jsd_loss(
    student_logits,  # (n_resp, V)
    teacher_topk_idx,  # (n_resp, top_k)
    teacher_topk_lp,  # (n_resp, top_k)
    rollout_log_probs,  # (n_resp,)
    response_ids,  # (n_resp,)
    alpha: float,
    is_clip: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    student_full_lp = F.log_softmax(student_logits, dim=-1)  # (n_resp, V)

    # Gather student log probs at teacher top-k positions
    student_topk_lp = student_full_lp.gather(1, teacher_topk_idx)  # (n_resp, top_k)

    # Renormalize to top-k subspace
    student_topk_lp = student_topk_lp - torch.logsumexp(student_topk_lp, dim=-1, keepdim=True)
    teacher_topk_lp_norm = teacher_topk_lp - torch.logsumexp(teacher_topk_lp, dim=-1, keepdim=True)

    # JSD mixture M = (1-alpha)*student + alpha*teacher  (log space)
    log_M = torch.logaddexp(
        student_topk_lp + math.log(max(1 - alpha, 1e-8)),
        teacher_topk_lp_norm + math.log(max(alpha, 1e-8)),
    )

    student_p = torch.exp(student_topk_lp)
    kl_s_M = (student_p * (student_topk_lp - log_M)).sum(dim=-1)  # (n_resp,)

    teacher_p = torch.exp(teacher_topk_lp_norm)
    kl_t_M = (teacher_p * (teacher_topk_lp_norm - log_M)).sum(dim=-1)  # (n_resp,)

    jsd_per_token = (1.0 - alpha) * kl_s_M + alpha * kl_t_M

    # Importance sampling correction
    student_resp_lp = student_full_lp.gather(1, response_ids.unsqueeze(1)).squeeze(1)
    log_ratio = (student_resp_lp - rollout_log_probs).clamp(-20.0, 20.0)
    clipped_ratio = torch.exp(log_ratio).clamp(max=is_clip)
    jsd_per_token = jsd_per_token * clipped_ratio

    return jsd_per_token.mean()


def _wandb_init(args: argparse.Namespace) -> None:
    if args.report_to != "wandb":
        return
    try:
        import wandb
        wandb.init(
            project="telco-fc",
            name="m4-sdpo",
            config={
                "model": args.model,
                "teacher_adapter": args.teacher_adapter,
                "top_k": args.top_k,
                "alpha": args.alpha,
                "is_clip": args.is_clip,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "grad_accum_steps": args.grad_accum_steps,
                "success_threshold": args.success_threshold,
            },
        )
    except ImportError:
        print("wandb not installed — skipping logging", flush=True)


def _wlog(metrics: dict, step: int | None = None) -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def train_sdpo(
    dataset: list[dict],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    output_dir: Path,
    anchor_data: list[dict] | None = None,
) -> None:
    import random
    import torch
    from torch.optim import AdamW

    output_dir.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    anchor_data = anchor_data or []

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    model.train()

    total_steps = args.epochs * len(dataset)
    global_step = 0
    grad_accum_steps = args.grad_accum_steps

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, sample in enumerate(dataset):
            student_ids = torch.tensor(sample["student_ids"], device=dev)
            response_ids = torch.tensor(sample["response_ids"], device=dev)
            teacher_topk_idx = torch.tensor(sample["teacher_topk_idx"], device=dev)
            teacher_topk_lp = torch.tensor(sample["teacher_topk_lp"], dtype=torch.float32, device=dev)
            rollout_lp = torch.tensor(sample["rollout_log_probs"], dtype=torch.float32, device=dev)

            student_full = torch.cat([student_ids, response_ids]).unsqueeze(0)
            out = model(student_full)
            student_logits = out.logits[0]  # (T, V)
            resp_start = len(student_ids) - 1
            n_resp = sample["n_response"]
            student_resp_logits = student_logits[resp_start: resp_start + n_resp]

            loss = compute_jsd_loss(
                student_resp_logits, teacher_topk_idx, teacher_topk_lp,
                rollout_lp, response_ids, args.alpha, args.is_clip,
            )
            loss = loss / grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0:
                # Anchor SFT loss on abstain sample — added before optimizer step
                # to prevent catastrophic forgetting of abstain capability
                if anchor_data:
                    anchor = random.choice(anchor_data)
                    a_loss = _compute_sft_nll(model, tokenizer, anchor, dev)
                    (args.anchor_weight * a_loss / grad_accum_steps).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                _wlog({"train/loss": loss.item() * grad_accum_steps}, step=global_step)

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"  epoch={epoch+1}/{args.epochs} step={batch_idx+1}/{len(dataset)} "
                    f"loss={loss.item()*grad_accum_steps:.4f}  {sample['id']}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(len(dataset), 1)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")
        _wlog({"train/epoch_avg_loss": avg_loss, "epoch": epoch + 1})
        _save_adapter(model, output_dir)

    print(f"\nAdapter saved → {output_dir}")


def _save_adapter(model: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))


def _load_anchor_samples(anchor_file: str, data_dir: Path) -> list[dict]:
    path = Path(anchor_file)
    if not path.exists():
        print(f"Anchor file not found: {path} — skipping anchor loss", flush=True)
        return []

    try:
        from src.evaluation.routing import build_sample_prompt, load_real_assets
        from src.registry.contract_registry import ContractRegistry
        from src.registry.tool_registry import ToolRegistry
        real_assets = load_real_assets(data_dir)
        tool_registry = ToolRegistry([])
        contract_registry = ContractRegistry([])
    except Exception as exc:
        print(f"Could not load real_assets for anchor: {exc} — skipping", flush=True)
        return []

    samples = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    abstain = [s for s in samples if s.get("expected_action") == "abstain"]

    anchor_data: list[dict] = []
    for s in abstain:
        try:
            messages = build_sample_prompt(s, tool_registry, contract_registry, real_assets)
            pred = s.get("prediction", {})
            gold = (
                json.dumps(pred, ensure_ascii=False)
                if isinstance(pred, dict) and pred.get("action") == "abstain"
                else json.dumps({"action": "abstain", "reason": "ngoài phạm vi công cụ KPI"}, ensure_ascii=False)
            )
            anchor_data.append({"messages": messages, "gold": gold})
        except Exception:
            continue

    print(f"Loaded {len(anchor_data)} anchor abstain samples from {path}", flush=True)
    return anchor_data


def _compute_sft_nll(model: Any, tokenizer: Any, anchor: dict, dev: Any) -> Any:
    import torch
    import torch.nn.functional as F

    try:
        text = tokenizer.apply_chat_template(
            anchor["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            anchor["messages"], tokenize=False, add_generation_prompt=True
        )

    eos = tokenizer.eos_token or ""
    gold_text = anchor["gold"] + eos

    prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    gold_ids = tokenizer(gold_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(gold_ids) == 0:
        return torch.tensor(0.0, device=dev)

    full_ids = torch.cat([prompt_ids, gold_ids]).unsqueeze(0).to(dev)
    out = model(full_ids)
    resp_start = len(prompt_ids) - 1
    resp_logits = out.logits[0][resp_start: resp_start + len(gold_ids)]
    return F.cross_entropy(resp_logits, gold_ids.to(dev))


def main() -> None:
    args = _parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rollout_path = Path(args.rollouts)
    if not rollout_path.exists():
        raise SystemExit(f"Rollouts not found: {rollout_path}\nRun: run_m4_rollouts.sh first.")

    records = [json.loads(l) for l in rollout_path.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} rollout groups from {rollout_path}")

    _wandb_init(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache_path = Path(args.sdpo_cache)

    # -----------------------------------------------------------------------
    # Phase A: precompute teacher top-k logits
    # -----------------------------------------------------------------------
    if cache_path.exists() and not args.recompute:
        print(f"Loading cached SDPO logits from {cache_path}")
        sdpo_dataset = [json.loads(l) for l in cache_path.open(encoding="utf-8") if l.strip()]
    else:
        print(f"\n=== Phase A: precomputing teacher logits ({args.teacher_adapter}) ===")
        base = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        teacher = PeftModel.from_pretrained(base, args.teacher_adapter)
        teacher.eval()

        sdpo_dataset = compute_teacher_logits(
            records, teacher, tokenizer, top_k=args.top_k,
            success_threshold=args.success_threshold,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            for s in sdpo_dataset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"SDPO cache ({len(sdpo_dataset)} samples) → {cache_path}")

        del teacher, base
        torch.cuda.empty_cache()

    print(f"SDPO dataset: {len(sdpo_dataset)} training samples")
    if not sdpo_dataset:
        raise SystemExit("No SDPO training samples — check rollouts and success_threshold.")

    # Load abstain anchor samples for anti-forgetting regularization
    anchor_data: list[dict] = []
    if args.anchor_file:
        anchor_data = _load_anchor_samples(args.anchor_file, ROOT / "data")

    # -----------------------------------------------------------------------
    # Phase B: train student
    # -----------------------------------------------------------------------
    print(f"\n=== Phase B: SDPO student training (resume {args.student_resume}) ===")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    student = PeftModel.from_pretrained(base, args.student_resume, is_trainable=True)

    train_sdpo(sdpo_dataset, student, tokenizer, args, Path(args.output_dir), anchor_data=anchor_data)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default=str(ROOT / "data" / "sdpo_rollouts_m4.jsonl"))
    p.add_argument("--sdpo-cache", default=str(ROOT / "data" / "sdpo_cache_m4.jsonl"))
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--model", required=True)
    p.add_argument("--teacher-adapter", required=True)
    p.add_argument("--student-resume", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--is-clip", type=float, default=2.0)
    p.add_argument("--success-threshold", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no-load-in-4bit", action="store_true")
    p.add_argument("--report-to", default="none")
    p.add_argument("--anchor-file", default=None, help="Path to train file with abstain samples for anchor SFT loss")
    p.add_argument("--anchor-weight", type=float, default=0.2, help="Weight for anchor SFT loss (default 0.2)")
    return p.parse_args()


if __name__ == "__main__":
    main()

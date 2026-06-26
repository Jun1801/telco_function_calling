"""VPD-lite — Variational Policy Distillation (HF Transformers port).

EM loop:
  E-step: update teacher using reward signal + KL trust-region penalty vs student
  M-step: distill teacher→student (top-k JSD with IS correction)
  Progressive reward lambda: 0.3 → 0.9 over training

Teacher sees: prompt + feedback (corrective context)
Student sees: prompt only (learns from teacher's distribution)

Both teacher and student are LoRA adapters on the same frozen base model.

python scripts/train_vpd_hf.py \
  --config configs/vpd.yaml \
  --model /workspace/models/Qwen3-4B \
  --adapter outputs/sft/m1b_qwen3-4b \
  --rollouts data/sdpo_rollouts_m4.jsonl \
  --output-dir outputs/sft/m5b_qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing import REAL_SOURCE
from src.reward.feedback_renderer import render_teacher_feedback

VPD_FEEDBACK_PREFIX = (
    "\n\n[VPD: Environment Feedback]\n{feedback_text}"
)


def _format_feedback(feedback: dict, lang: str = "vi") -> str:
    if feedback.get("errors"):
        return render_teacher_feedback(feedback, lang)
    texts = feedback.get("feedback_text", [])
    return " ".join(texts) if texts else "The response was incorrect."


def _build_teacher_messages(record: dict, best_rollout: dict) -> list[dict] | None:
    lang = "vi" if record.get("source") == REAL_SOURCE else "en"
    feedback_text = _format_feedback(best_rollout["feedback"], lang)
    original_msgs = record["prompt_messages"]
    if not original_msgs or original_msgs[0]["role"] != "system":
        return None
    teacher_system = original_msgs[0]["content"] + VPD_FEEDBACK_PREFIX.format(
        feedback_text=feedback_text
    )
    return [{"role": "system", "content": teacher_system}] + original_msgs[1:]


def _progressive_lambda(step: int, total_steps: int, lam_start: float, lam_end: float) -> float:
    """Sigmoid ramp from lam_start to lam_end over training steps."""
    frac = step / max(total_steps - 1, 1)
    # Sigmoid transition: flat 0–20%, sigmoid 20–50%, flat 50–100%
    if frac < 0.2:
        return lam_start
    elif frac > 0.5:
        return lam_end
    t = (frac - 0.2) / 0.3  # 0→1 in the transition window
    sig = 1 / (1 + math.exp(-10 * (t - 0.5)))
    return lam_start + (lam_end - lam_start) * sig


def compute_top_k_kl(
    p_lp,  # (n, V) — log probs of P (teacher)
    q_lp,  # (n, V) — log probs of Q (student)
    top_k: int,
) -> Any:
    """KL(P || Q) restricted to top-k tokens of P."""
    import torch
    topk_lp, topk_idx = torch.topk(p_lp, top_k, dim=-1)
    # Renormalize P to top-k subspace
    topk_lp_norm = topk_lp - torch.logsumexp(topk_lp, dim=-1, keepdim=True)
    # Gather Q at same positions
    q_topk_lp = q_lp.gather(1, topk_idx)
    q_topk_lp_norm = q_topk_lp - torch.logsumexp(q_topk_lp, dim=-1, keepdim=True)

    p_topk = torch.exp(topk_lp_norm)
    kl = (p_topk * (topk_lp_norm - q_topk_lp_norm)).sum(dim=-1)  # (n,)
    return kl.mean()


def compute_jsd_loss(
    student_logits,
    teacher_topk_idx,
    teacher_topk_lp,
    rollout_lp,
    response_ids,
    alpha: float,
    is_clip: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    student_full_lp = F.log_softmax(student_logits, dim=-1)
    student_topk_lp = student_full_lp.gather(1, teacher_topk_idx)
    student_topk_lp = student_topk_lp - torch.logsumexp(student_topk_lp, dim=-1, keepdim=True)
    teacher_norm = teacher_topk_lp - torch.logsumexp(teacher_topk_lp, dim=-1, keepdim=True)

    log_M = torch.logaddexp(
        student_topk_lp + math.log(max(1 - alpha, 1e-8)),
        teacher_norm + math.log(max(alpha, 1e-8)),
    )
    kl_s_M = (torch.exp(student_topk_lp) * (student_topk_lp - log_M)).sum(dim=-1)
    kl_t_M = (torch.exp(teacher_norm) * (teacher_norm - log_M)).sum(dim=-1)
    jsd = (1 - alpha) * kl_s_M + alpha * kl_t_M

    student_resp_lp = student_full_lp.gather(1, response_ids.unsqueeze(1)).squeeze(1)
    log_ratio = (student_resp_lp - rollout_lp).clamp(-20, 20)
    clipped_ratio = torch.exp(log_ratio).clamp(max=is_clip)
    return (jsd * clipped_ratio).mean()


def e_step(
    teacher_model: Any,
    student_model: Any,
    tokenizer: Any,
    records: list[dict],
    cfg: Any,
    opt_teacher: Any,
    lam: float,
    top_k: int,
) -> float:
    """Update teacher using reward signal + KL(teacher || student) trust-region."""
    import torch
    import torch.nn.functional as F

    dev = next(teacher_model.parameters()).device
    total_loss = 0.0
    n = 0

    teacher_model.train()
    student_model.eval()

    for record in records:
        rollouts = record["rollouts"]
        best_idx = record.get("best_idx", -1)
        if best_idx < 0 or best_idx >= len(rollouts):
            best_idx = max(range(len(rollouts)), key=lambda j: rollouts[j]["reward"])

        best_rollout = rollouts[best_idx]
        reward = best_rollout["reward"]

        # Only use samples with some signal
        if reward == 0.0 and record.get("avg_reward", 0.0) == 0.0:
            continue

        teacher_msgs = _build_teacher_messages(record, best_rollout)
        if teacher_msgs is None:
            continue

        raw_out = best_rollout.get("raw_output", "").strip()
        if not raw_out:
            raw_out = json.dumps(best_rollout["prediction"], ensure_ascii=False)
        eos = tokenizer.eos_token or ""

        teacher_text = tokenizer.apply_chat_template(
            teacher_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        response_text = raw_out + eos

        teacher_ids = tokenizer(teacher_text, return_tensors="pt")["input_ids"][0].to(dev)
        response_ids = tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(dev)
        n_resp = len(response_ids)
        if n_resp == 0:
            continue

        teacher_full = torch.cat([teacher_ids, response_ids]).unsqueeze(0)
        student_full = teacher_full  # same context for KL computation

        teacher_out = teacher_model(teacher_full)
        teacher_logits = teacher_out.logits[0]
        resp_start = len(teacher_ids) - 1
        teacher_resp_logits = teacher_logits[resp_start: resp_start + n_resp]
        teacher_resp_lp = F.log_softmax(teacher_resp_logits, dim=-1)

        with torch.no_grad():
            student_out = student_model(student_full)
            student_logits = student_out.logits[0]
            student_resp_lp = F.log_softmax(student_logits[resp_start: resp_start + n_resp], dim=-1)

        # Reward-weighted NLL loss
        nll = -teacher_resp_lp.gather(1, response_ids.unsqueeze(1)).squeeze(1).mean()
        if reward >= cfg.success_threshold:
            reward_loss = lam * reward * nll
        else:
            # Negative trajectory: small push-away (min probability of this bad action)
            reward_loss = cfg.neg_loss_weight * nll

        # KL(teacher || student) trust-region penalty
        kl_penalty = compute_top_k_kl(teacher_resp_lp, student_resp_lp, top_k)

        loss = reward_loss + cfg.teacher_trust_region_beta * kl_penalty
        loss.backward()
        total_loss += loss.item()
        n += 1

    if n > 0:
        torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1.0)
        opt_teacher.step()
        opt_teacher.zero_grad()

    return total_loss / max(n, 1)


def m_step(
    student_model: Any,
    teacher_model: Any,
    tokenizer: Any,
    records: list[dict],
    cfg: Any,
    opt_student: Any,
    top_k: int,
    alpha: float,
    is_clip: float,
    anchor_data: list[dict] | None = None,
    anchor_weight: float = 0.2,
) -> float:
    """Distill teacher→student (top-k JSD with IS correction)."""
    import random
    import torch
    import torch.nn.functional as F

    dev = next(student_model.parameters()).device
    total_loss = 0.0
    n = 0
    anchor_data = anchor_data or []

    student_model.train()
    teacher_model.eval()

    opt_student.zero_grad()

    for step_idx, record in enumerate(records):
        rollouts = record["rollouts"]
        best_idx = record.get("best_idx", -1)
        if best_idx < 0 or best_idx >= len(rollouts):
            best_idx = max(range(len(rollouts)), key=lambda j: rollouts[j]["reward"])

        if rollouts[best_idx]["reward"] < cfg.success_threshold:
            continue

        teacher_msgs = _build_teacher_messages(record, rollouts[best_idx])
        student_msgs = record["prompt_messages"]
        if teacher_msgs is None:
            teacher_msgs = student_msgs

        raw_out = rollouts[best_idx].get("raw_output", "").strip()
        if not raw_out:
            raw_out = json.dumps(rollouts[best_idx]["prediction"], ensure_ascii=False)
        eos = tokenizer.eos_token or ""

        teacher_text = tokenizer.apply_chat_template(
            teacher_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        student_text = tokenizer.apply_chat_template(
            student_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        response_text = raw_out + eos

        teacher_ids = tokenizer(teacher_text, return_tensors="pt")["input_ids"][0].to(dev)
        student_ids = tokenizer(student_text, return_tensors="pt")["input_ids"][0].to(dev)
        response_ids = tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(dev)
        n_resp = len(response_ids)
        if n_resp == 0:
            continue

        # Teacher top-k logits (no grad)
        with torch.no_grad():
            teacher_full = torch.cat([teacher_ids, response_ids]).unsqueeze(0)
            teacher_out = teacher_model(teacher_full)
            teacher_logits = teacher_out.logits[0]
            resp_start_t = len(teacher_ids) - 1
            teacher_resp_lp = F.log_softmax(
                teacher_logits[resp_start_t: resp_start_t + n_resp], dim=-1
            )
            topk_lp, topk_idx = torch.topk(teacher_resp_lp, top_k, dim=-1)

        # Rollout IS log-probs (teacher on student context, no grad)
        with torch.no_grad():
            student_full_rollout = torch.cat([student_ids, response_ids]).unsqueeze(0)
            rollout_out = teacher_model(student_full_rollout)
            rollout_logits = rollout_out.logits[0]
            resp_start_s = len(student_ids) - 1
            rollout_lp = F.log_softmax(
                rollout_logits[resp_start_s: resp_start_s + n_resp], dim=-1
            ).gather(1, response_ids.unsqueeze(1)).squeeze(1)

        # Student forward (with grad)
        student_full = torch.cat([student_ids, response_ids]).unsqueeze(0)
        student_out = student_model(student_full)
        student_logits = student_out.logits[0][resp_start_s: resp_start_s + n_resp]

        loss = compute_jsd_loss(
            student_logits, topk_idx, topk_lp, rollout_lp, response_ids, alpha, is_clip
        )
        loss.backward()
        total_loss += loss.item()
        n += 1

        if (step_idx + 1) % cfg.grad_accum_steps == 0:
            if anchor_data:
                anchor = random.choice(anchor_data)
                a_loss = _compute_sft_nll(student_model, tokenizer, anchor, dev)
                (anchor_weight * a_loss / cfg.grad_accum_steps).backward()

            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            opt_student.step()
            opt_student.zero_grad()

    if n > 0:
        if anchor_data:
            anchor = random.choice(anchor_data)
            a_loss = _compute_sft_nll(student_model, tokenizer, anchor, dev)
            (anchor_weight * a_loss / max(cfg.grad_accum_steps, 1)).backward()

        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        opt_student.step()
        opt_student.zero_grad()

    return total_loss / max(n, 1)


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


def _wandb_init(args: argparse.Namespace, cfg: Any) -> None:
    if args.report_to != "wandb":
        return
    try:
        import wandb
        wandb.init(
            project="telco-fc",
            name="m5-vpd",
            config={
                "model": args.model,
                "adapter": args.adapter,
                "epochs": cfg.epochs,
                "e_steps_per_cycle": cfg.e_steps_per_cycle,
                "m_steps_per_cycle": cfg.m_steps_per_cycle,
                "learning_rate_e": cfg.learning_rate_e,
                "learning_rate_m": cfg.learning_rate_m,
                "distillation_top_k": cfg.distillation_top_k,
                "distillation_alpha": cfg.distillation_alpha,
                "lambda_start": cfg.lambda_start,
                "lambda_end": cfg.lambda_end,
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


def main() -> None:
    args = _parse_args()

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _load_config(args)

    rollout_path = Path(args.rollouts)
    if not rollout_path.exists():
        raise SystemExit(f"Rollouts not found: {rollout_path}\nRun: run_m4_rollouts.sh first.")

    records = [json.loads(l) for l in rollout_path.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} rollout groups from {rollout_path}")

    _wandb_init(args, cfg)

    anchor_data: list[dict] = []
    if args.anchor_file:
        anchor_data = _load_anchor_samples(args.anchor_file, ROOT / "data")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.model} ...")
    base_teacher = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    base_student = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # Both teacher and student start from M1b adapter
    teacher = PeftModel.from_pretrained(base_teacher, args.adapter, is_trainable=True)
    student = PeftModel.from_pretrained(base_student, args.adapter, is_trainable=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt_teacher = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, teacher.parameters()),
        lr=cfg.learning_rate_e,
    )
    opt_student = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=cfg.learning_rate_m,
    )
    setattr(cfg, "grad_accum_steps", 4)

    total_cycles = cfg.epochs
    print(f"\n=== VPD-lite EM loop: {total_cycles} cycles "
          f"({cfg.e_steps_per_cycle}E + {cfg.m_steps_per_cycle}M per cycle) ===")

    for cycle in range(total_cycles):
        lam = _progressive_lambda(cycle, total_cycles, cfg.lambda_start, cfg.lambda_end)
        print(f"\n--- Cycle {cycle+1}/{total_cycles}  lambda={lam:.3f} ---")

        # E-step(s)
        e_loss_last = 0.0
        for e in range(cfg.e_steps_per_cycle):
            e_loss_last = e_step(
                teacher, student, tokenizer, records, cfg, opt_teacher, lam,
                top_k=cfg.distillation_top_k,
            )
            print(f"  E-step {e+1}/{cfg.e_steps_per_cycle}  loss={e_loss_last:.4f}")

        # M-step(s)
        m_loss_last = 0.0
        for m in range(cfg.m_steps_per_cycle):
            m_loss_last = m_step(
                student, teacher, tokenizer, records, cfg, opt_student,
                top_k=cfg.distillation_top_k,
                alpha=cfg.distillation_alpha,
                is_clip=cfg.is_clip,
                anchor_data=anchor_data,
                anchor_weight=args.anchor_weight,
            )
            print(f"  M-step {m+1}/{cfg.m_steps_per_cycle}  loss={m_loss_last:.4f}")

        _wlog({
            "train/e_loss": e_loss_last,
            "train/m_loss": m_loss_last,
            "train/lambda": lam,
        }, step=cycle + 1)

        # Save student after each cycle
        student.save_pretrained(str(output_dir))
        print(f"  Saved student adapter → {output_dir}")

    print(f"\nVPD-lite done. Final student adapter → {output_dir}")


def _load_config(args: argparse.Namespace) -> Any:
    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text())

    class Cfg:
        pass

    c = Cfg()
    defaults = {
        "e_steps_per_cycle": 1,
        "m_steps_per_cycle": 3,
        "epochs": 3,
        "learning_rate_e": 1e-5,
        "learning_rate_m": 5e-6,
        "distillation_top_k": 20,
        "distillation_alpha": 0.5,
        "is_clip": 2.0,
        "teacher_trust_region_beta": 0.02,
        "neg_loss_weight": 0.5,
        "success_threshold": 1.0,
        "progressive_reward": True,
        "lambda_start": 0.3,
        "lambda_end": 0.9,
    }
    _float_fields = {"learning_rate_e", "learning_rate_m", "distillation_alpha",
                     "is_clip", "teacher_trust_region_beta", "neg_loss_weight",
                     "lambda_start", "lambda_end"}
    merged = {**defaults, **raw}
    for k, v in merged.items():
        if k in _float_fields and isinstance(v, str):
            merged[k] = float(v)
    for k, v in merged.items():
        setattr(c, k, v)
    return c


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "vpd.yaml"))
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True, help="M1b adapter (student + teacher start)")
    p.add_argument("--rollouts", default=str(ROOT / "data" / "sdpo_rollouts_m4.jsonl"))
    p.add_argument("--eval-file", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--report-to", default="none")
    p.add_argument("--anchor-file", default=None, help="Path to train file with abstain samples for anchor SFT loss")
    p.add_argument("--anchor-weight", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    main()

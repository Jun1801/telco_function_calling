"""VPD — Variational Policy Distillation (HF Transformers port).

EM loop (paper-faithful, arxiv 2605.15113):
  E-step: BCO loss on ALL K rollouts per group using implicit reward
          r̃ = β * (log q_teacher(y|x,C) - log π_student(y|x))
          ℒ_E = -E_y+[log σ(r̃+ - δ)] - E_y-[log σ(-(r̃- - δ))]
  M-step: Forward KL(student || teacher) without IS correction (Eq. 10)

Teacher sees: prompt + feedback C (corrective context)
Student sees: prompt only (learns from teacher's distribution)

Both teacher and student are LoRA adapters on the same frozen base model.

python scripts/train_vpd_hf.py \
  --config configs/vpd.yaml \
  --model /workspace/models/Qwen3-4B \
  --adapter /content/outputs/m5 \
  --student-adapter /content/adapters/m1 \
  --rollouts /content/data/rollout/rollouts_m6.jsonl \
  --output-dir /content/outputs/m6v2
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


def _build_teacher_messages(record: dict, rollout: dict) -> list[dict] | None:
    lang = "vi" if record.get("source") == REAL_SOURCE else "en"
    feedback_text = _format_feedback(rollout["feedback"], lang)
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
    if frac < 0.2:
        return lam_start
    elif frac > 0.5:
        return lam_end
    t = (frac - 0.2) / 0.3
    sig = 1 / (1 + math.exp(-10 * (t - 0.5)))
    return lam_start + (lam_end - lam_start) * sig


def compute_top_k_kl(
    p_lp,  # (n, V) — log probs of P (teacher)
    q_lp,  # (n, V) — log probs of Q (student)
    top_k: int,
) -> Any:
    """KL(P || Q) restricted to top-k tokens of P — trust-region penalty."""
    import torch
    topk_lp, topk_idx = torch.topk(p_lp, top_k, dim=-1)
    topk_lp_norm = topk_lp - torch.logsumexp(topk_lp, dim=-1, keepdim=True)
    q_topk_lp = q_lp.gather(1, topk_idx)
    q_topk_lp_norm = q_topk_lp - torch.logsumexp(q_topk_lp, dim=-1, keepdim=True)
    p_topk = torch.exp(topk_lp_norm)
    kl = (p_topk * (topk_lp_norm - q_topk_lp_norm)).sum(dim=-1)
    return kl.mean()


def e_step(
    teacher_model: Any,
    student_model: Any,
    tokenizer: Any,
    records: list[dict],
    cfg: Any,
    opt_teacher: Any,
    top_k: int,
    max_records: int | None = None,
) -> float:
    """E-step: update teacher via BCO loss on ALL K rollouts (paper Eq. 7+9).

    For each group, compute implicit reward per rollout:
        r̃ = bco_beta * (log q_teacher(y|x,C) - log π_student(y|x))  [token avg]
    Then apply BCO loss:
        ℒ = -log σ(r̃+ - δ)  for positive rollouts (reward >= threshold)
           -log σ(-(r̃- - δ)) for negative rollouts
    where δ = mean(r̃) over the group (reward shift, Eq. 9).
    Trust-region: KL(teacher || student) penalty.
    """
    import random
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    dev = next(teacher_model.parameters()).device
    total_loss = 0.0
    n = 0

    teacher_model.train()
    student_model.eval()

    if max_records and len(records) > max_records:
        records = random.sample(records, max_records)

    for record in tqdm(records, desc="E-step", leave=False):
        rollouts = record["rollouts"]

        # Skip groups with no contrastive signal (all rollouts same reward)
        rewards = [ro["reward"] for ro in rollouts]
        if len(set(round(r, 3) for r in rewards)) == 1:
            continue

        eos = tokenizer.eos_token or ""
        student_msgs = record["prompt_messages"]
        try:
            student_text = tokenizer.apply_chat_template(
                student_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            student_text = tokenizer.apply_chat_template(
                student_msgs, tokenize=False, add_generation_prompt=True
            )
        student_ids = tokenizer(student_text, return_tensors="pt")["input_ids"][0].to(dev)

        # Collect implicit rewards for all rollouts in this group
        group_entries: list[tuple[Any, bool, Any, Any]] = []
        # (implicit_reward_tensor, is_positive, teacher_resp_lp, student_resp_lp_detached)

        for rollout in rollouts:
            is_positive = rollout["reward"] >= cfg.success_threshold
            teacher_msgs = _build_teacher_messages(record, rollout)
            if teacher_msgs is None:
                continue

            raw_out = rollout.get("raw_output", "").strip()
            if not raw_out:
                raw_out = json.dumps(rollout["prediction"], ensure_ascii=False)

            try:
                teacher_text = tokenizer.apply_chat_template(
                    teacher_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                teacher_text = tokenizer.apply_chat_template(
                    teacher_msgs, tokenize=False, add_generation_prompt=True
                )

            response_text = raw_out + eos
            teacher_ids = tokenizer(teacher_text, return_tensors="pt")["input_ids"][0].to(dev)
            response_ids = tokenizer(
                response_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].to(dev)
            n_resp = len(response_ids)
            if n_resp == 0:
                continue

            # Teacher forward with feedback context (gradients enabled)
            teacher_full = torch.cat([teacher_ids, response_ids]).unsqueeze(0)
            teacher_out = teacher_model(teacher_full)
            t_resp_start = len(teacher_ids) - 1
            teacher_resp_lp = F.log_softmax(
                teacher_out.logits[0][t_resp_start: t_resp_start + n_resp], dim=-1
            )
            teacher_token_lp = teacher_resp_lp.gather(
                1, response_ids.unsqueeze(1)
            ).squeeze(1)  # (n_resp,)

            # Student forward WITHOUT feedback (stop-grad)
            student_full = torch.cat([student_ids, response_ids]).unsqueeze(0)
            with torch.no_grad():
                student_out = student_model(student_full)
                s_resp_start = len(student_ids) - 1
                student_resp_lp = F.log_softmax(
                    student_out.logits[0][s_resp_start: s_resp_start + n_resp], dim=-1
                )
                student_token_lp = student_resp_lp.gather(
                    1, response_ids.unsqueeze(1)
                ).squeeze(1).detach()  # (n_resp,)

            # Implicit reward: β * (log q_teacher - log π_student) averaged over tokens (Eq. 7)
            implicit_reward = cfg.bco_beta * (teacher_token_lp - student_token_lp).mean()
            group_entries.append((implicit_reward, is_positive, teacher_resp_lp, student_resp_lp))

        if len(group_entries) < 2:
            continue

        # δ = mean implicit reward over the group (reward shift for centering, Eq. 9)
        delta = torch.stack([e[0] for e in group_entries]).mean().detach()

        # BCO loss (Eq. 9)
        group_loss = torch.tensor(0.0, device=dev)
        for implicit_reward, is_positive, _, _ in group_entries:
            if is_positive:
                group_loss = group_loss + (-F.logsigmoid(implicit_reward - delta))
            else:
                group_loss = group_loss + (-F.logsigmoid(-(implicit_reward - delta)))
        group_loss = group_loss / len(group_entries)

        # KL trust-region penalty on last rollout's distributions (Eq. A.22)
        last_student_lp = group_entries[-1][3].detach()
        kl_penalty = compute_top_k_kl(
            group_entries[-1][2],  # teacher_resp_lp (has grad)
            last_student_lp,
            top_k,
        )

        loss = group_loss + cfg.teacher_trust_region_beta * kl_penalty
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
    max_records: int | None = None,
) -> float:
    """M-step: distill teacher→student via forward KL (paper Eq. 10).

    Forward KL(student || teacher) per token, no IS correction.
    Teacher uses feedback context; student uses prompt only.
    loss = mean_t [log q_teacher(y_t|x,C,y<t) - log π_student(y_t|x,y<t)]
    Minimise (≥0 when student < teacher) → student_lp rises toward teacher_lp.
    """
    import random
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    dev = next(student_model.parameters()).device
    total_loss = 0.0
    n = 0
    anchor_data = anchor_data or []

    student_model.train()
    teacher_model.eval()

    opt_student.zero_grad()

    if max_records and len(records) > max_records:
        records = random.sample(records, max_records)

    for step_idx, record in enumerate(tqdm(records, desc="M-step", leave=False)):
        rollouts = record["rollouts"]
        best_idx = record.get("best_idx", -1)
        if best_idx < 0 or best_idx >= len(rollouts):
            best_idx = max(range(len(rollouts)), key=lambda j: rollouts[j]["reward"])

        if rollouts[best_idx]["reward"] < cfg.success_threshold:
            continue

        best_rollout = rollouts[best_idx]
        teacher_msgs = _build_teacher_messages(record, best_rollout)
        student_msgs = record["prompt_messages"]
        if teacher_msgs is None:
            teacher_msgs = student_msgs

        raw_out = best_rollout.get("raw_output", "").strip()
        if not raw_out:
            raw_out = json.dumps(best_rollout["prediction"], ensure_ascii=False)
        eos = tokenizer.eos_token or ""

        try:
            teacher_text = tokenizer.apply_chat_template(
                teacher_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            student_text = tokenizer.apply_chat_template(
                student_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            teacher_text = tokenizer.apply_chat_template(
                teacher_msgs, tokenize=False, add_generation_prompt=True
            )
            student_text = tokenizer.apply_chat_template(
                student_msgs, tokenize=False, add_generation_prompt=True
            )

        response_text = raw_out + eos
        teacher_ids = tokenizer(teacher_text, return_tensors="pt")["input_ids"][0].to(dev)
        student_ids = tokenizer(student_text, return_tensors="pt")["input_ids"][0].to(dev)
        response_ids = tokenizer(
            response_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(dev)
        n_resp = len(response_ids)
        if n_resp == 0:
            continue

        # Teacher log-probs with feedback context (stop-grad — teacher is the target)
        with torch.no_grad():
            teacher_full = torch.cat([teacher_ids, response_ids]).unsqueeze(0)
            teacher_out = teacher_model(teacher_full)
            t_resp_start = len(teacher_ids) - 1
            teacher_token_lp = F.log_softmax(
                teacher_out.logits[0][t_resp_start: t_resp_start + n_resp], dim=-1
            ).gather(1, response_ids.unsqueeze(1)).squeeze(1)  # (n_resp,)

        # Student forward with grad
        student_full = torch.cat([student_ids, response_ids]).unsqueeze(0)
        student_out = student_model(student_full)
        s_resp_start = len(student_ids) - 1
        student_resp_lp = F.log_softmax(
            student_out.logits[0][s_resp_start: s_resp_start + n_resp], dim=-1
        )
        student_token_lp = student_resp_lp.gather(
            1, response_ids.unsqueeze(1)
        ).squeeze(1)  # (n_resp,)

        # Forward KL distillation: minimize teacher_lp - student_lp (≥0 when student<teacher)
        # Gradient pushes student_lp UP toward teacher_lp; converges to 0 at student=teacher
        loss = (teacher_token_lp.detach() - student_token_lp).mean()

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
            name="m6v2-vpd-bco",
            config={
                "model": args.model,
                "adapter": args.adapter,
                "student_adapter": args.student_adapter,
                "epochs": cfg.epochs,
                "e_steps_per_cycle": cfg.e_steps_per_cycle,
                "m_steps_per_cycle": cfg.m_steps_per_cycle,
                "learning_rate_e": cfg.learning_rate_e,
                "learning_rate_m": cfg.learning_rate_m,
                "bco_beta": cfg.bco_beta,
                "teacher_trust_region_beta": cfg.teacher_trust_region_beta,
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
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _load_config(args)

    rollout_path = Path(args.rollouts)
    if not rollout_path.exists():
        raise SystemExit(f"Rollouts not found: {rollout_path}")

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
    _model_kwargs: dict = dict(device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
    try:
        import flash_attn  # noqa: F401
        _model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  Using flash_attention_2")
    except ImportError:
        print("  flash-attn not installed — using default attention")

    base_teacher = AutoModelForCausalLM.from_pretrained(args.model, **_model_kwargs)
    base_student = AutoModelForCausalLM.from_pretrained(args.model, **_model_kwargs)

    teacher = PeftModel.from_pretrained(base_teacher, args.adapter, is_trainable=True)
    student_path = args.student_adapter or args.adapter
    student = PeftModel.from_pretrained(base_student, student_path, is_trainable=True)
    if args.student_adapter:
        print(f"  Asymmetric init: teacher={args.adapter}, student={student_path}")

    teacher.enable_input_require_grads()
    teacher.gradient_checkpointing_enable()
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable()

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
    print(f"\n=== VPD EM loop (BCO): {total_cycles} cycles "
          f"({cfg.e_steps_per_cycle}E + {cfg.m_steps_per_cycle}M per cycle) ===")
    print(f"  bco_beta={cfg.bco_beta}  trust_beta={cfg.teacher_trust_region_beta}")

    for cycle in range(total_cycles):
        lam = _progressive_lambda(cycle, total_cycles, cfg.lambda_start, cfg.lambda_end)
        print(f"\n--- Cycle {cycle+1}/{total_cycles}  lambda={lam:.3f} ---")

        e_loss_last = 0.0
        for e in range(cfg.e_steps_per_cycle):
            e_loss_last = e_step(
                teacher, student, tokenizer, records, cfg, opt_teacher,
                top_k=cfg.distillation_top_k,
                max_records=args.max_records_per_step,
            )
            print(f"  E-step {e+1}/{cfg.e_steps_per_cycle}  loss={e_loss_last:.4f}")

        torch.cuda.empty_cache()

        m_loss_last = 0.0
        for m in range(cfg.m_steps_per_cycle):
            m_loss_last = m_step(
                student, teacher, tokenizer, records, cfg, opt_student,
                top_k=cfg.distillation_top_k,
                alpha=cfg.distillation_alpha,
                is_clip=cfg.is_clip,
                anchor_data=anchor_data,
                anchor_weight=args.anchor_weight,
                max_records=args.max_records_per_step,
            )
            print(f"  M-step {m+1}/{cfg.m_steps_per_cycle}  loss={m_loss_last:.4f}")

        _wlog({
            "train/e_loss": e_loss_last,
            "train/m_loss": m_loss_last,
            "train/lambda": lam,
        }, step=cycle + 1)

        student.save_pretrained(str(output_dir))
        print(f"  Saved student adapter → {output_dir}")

    print(f"\nVPD done. Final student adapter → {output_dir}")


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
        "bco_beta": 0.5,
        "neg_loss_weight": 0.5,
        "success_threshold": 1.0,
        "progressive_reward": True,
        "lambda_start": 0.3,
        "lambda_end": 0.9,
    }
    _float_fields = {
        "learning_rate_e", "learning_rate_m", "distillation_alpha",
        "is_clip", "teacher_trust_region_beta", "bco_beta",
        "neg_loss_weight", "lambda_start", "lambda_end",
    }
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
    p.add_argument("--adapter", required=True, help="Teacher adapter")
    p.add_argument("--student-adapter", default=None, help="Student adapter (default: same as --adapter)")
    p.add_argument("--rollouts", default=str(ROOT / "data" / "sdpo_rollouts_m4.jsonl"))
    p.add_argument("--eval-file", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--report-to", default="none")
    p.add_argument("--anchor-file", default=None)
    p.add_argument("--anchor-weight", type=float, default=0.2)
    p.add_argument("--max-records-per-step", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    main()

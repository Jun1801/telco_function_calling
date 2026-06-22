"""
Phase 5 — Standard SDPO training (offline single-device MLX adaptation).

Algorithm (faithful to lasgroup/SDPO):
  Teacher context = prompt + env_feedback + best_sibling_demo (if reward >= threshold)
  Student context = prompt only (no feedback, no demo)
  Loss = JSD(student_logits || teacher_logits) on response tokens, top-k=20
  IS correction = clip(student_logp / rollout_logp, max=is_clip)

Single-device adaptation (M3 Pro 18GB):
  Phase A: Load teacher (M3 adapter) → precompute top-k logits → unload
  Phase B: Load student (M3 adapter as start) → train with stored logits → save

Usage:
  python3.11 -m src.training.train_sdpo \\
      --rollouts data/sdpo_rollouts.jsonl \\
      --model Qwen/Qwen3-4B \\
      --teacher-adapter outputs/sft_mlx/qwen3-4b-feedback-sdft \\
      --student-resume outputs/sft_mlx/qwen3-4b-feedback-sdft \\
      --output-dir outputs/sft_mlx/qwen3-4b-sdpo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing import REAL_SOURCE
from src.reward.feedback_renderer import render_teacher_feedback


# ---------------------------------------------------------------------------
# Teacher context builder (mirrors SDPO reprompt_template)
# ---------------------------------------------------------------------------

SDPO_TEACHER_SUFFIX_TEMPLATE = (
    "\n\n[SDPO: Previous Attempt]\n{best_response}"
    "\n\n[SDPO: Environment Feedback]\n{feedback_text}"
)


def _format_feedback(feedback: dict[str, Any], lang: str = "vi") -> str:
    # Rich structured rendering (path/expected/actual/suggested) beats a flat
    # join — the teacher reasons better from it. Falls back for empty feedback.
    if feedback.get("errors"):
        return render_teacher_feedback(feedback, lang)
    texts = feedback.get("feedback_text", [])
    if texts:
        return " ".join(texts)
    return "The response was incorrect."


def _record_lang(record: dict[str, Any]) -> str:
    return "vi" if record.get("source") == REAL_SOURCE else "en"


def build_teacher_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    """
    Build teacher context = original prompt + best sibling demo + feedback.
    Returns None if no successful sibling exists (avg_reward=0, best_idx=-1).
    """
    best_idx = record["best_rollout_idx"]
    rollouts = record["rollouts"]
    lang = _record_lang(record)

    if best_idx < 0:
        # No successful sibling — use feedback-only teacher context
        # Pick the rollout with highest reward as "closest attempt"
        best_idx = max(range(len(rollouts)), key=lambda j: rollouts[j]["reward"])
        best_response = json.dumps(rollouts[best_idx]["prediction"], ensure_ascii=False)
        feedback_text = _format_feedback(rollouts[best_idx]["feedback"], lang)
        has_demo = False
    else:
        best_response = json.dumps(rollouts[best_idx]["prediction"], ensure_ascii=False)
        feedback_text = _format_feedback(rollouts[best_idx]["feedback"], lang)
        has_demo = True

    original_msgs = record["prompt"]
    if not original_msgs or original_msgs[0]["role"] != "system":
        return None

    teacher_system = original_msgs[0]["content"] + SDPO_TEACHER_SUFFIX_TEMPLATE.format(
        best_response=best_response,
        feedback_text=feedback_text,
    )
    teacher_msgs = [{"role": "system", "content": teacher_system}] + original_msgs[1:]
    return teacher_msgs, has_demo


def build_teacher_messages_safe(record: dict[str, Any]):
    result = build_teacher_messages(record)
    if result is None:
        return None, False
    return result


# ---------------------------------------------------------------------------
# Phase A — precompute teacher top-k logits
# ---------------------------------------------------------------------------

def precompute_teacher_logits(
    rollout_records: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    top_k: int,
    success_threshold: float,
) -> list[dict[str, Any]]:
    """
    For each rollout record, build teacher context, run forward pass,
    extract top-k logits at each response position.

    Returns list of SDPO training samples:
      student_input_ids, response_start, n_response,
      teacher_topk_idx, teacher_topk_lp,
      rollout_log_probs (for IS correction),
      reward, has_demo
    """
    import mlx.core as mx
    import mlx.nn as nn

    dataset = []

    for rec_idx, record in enumerate(rollout_records):
        # Student context = original prompt (no feedback, no demo)
        student_msgs = record["prompt"]

        # Teacher context = prompt + feedback + best demo
        teacher_result = build_teacher_messages_safe(record)
        teacher_msgs, has_demo = teacher_result

        # Use best successful rollout as response target. If no rollout clears the
        # success threshold, skip the group entirely: distilling toward a reward-0
        # "best attempt" pushes the student toward bad outputs (root cause of M4
        # 47.9% collapse).
        best_idx = record["best_rollout_idx"]
        if best_idx < 0:
            best_idx = max(range(len(record["rollouts"])),
                           key=lambda j: record["rollouts"][j]["reward"])
        if record["rollouts"][best_idx]["reward"] < success_threshold:
            continue

        best_rollout = record["rollouts"][best_idx]
        # Use raw_output (the actual generation, possibly including <think>...</think>).
        # This is critical: response_tokens must match what the model actually generates
        # at inference time so that teacher logits at each position are meaningful.
        # Using json.dumps(prediction) would skip the <think> prefix, misaligning
        # teacher logits with the model's actual first-token prediction.
        raw_out = best_rollout.get("raw_output", "").strip()
        if not raw_out:
            raw_out = json.dumps(best_rollout["prediction"], ensure_ascii=False)
        target_response_raw = raw_out
        reward = best_rollout["reward"]

        # Tokenize student context
        student_text = tokenizer.apply_chat_template(
            student_msgs, tokenize=False, add_generation_prompt=True
        )
        student_tokens = tokenizer.encode(student_text, add_special_tokens=False)

        # Tokenize teacher context
        if teacher_msgs is not None:
            teacher_text = tokenizer.apply_chat_template(
                teacher_msgs, tokenize=False, add_generation_prompt=True
            )
            teacher_tokens = tokenizer.encode(teacher_text, add_special_tokens=False)
        else:
            teacher_tokens = student_tokens

        # Tokenize response — use actual raw generation (includes <think> if present)
        # EOS is added explicitly since mlx_generate strips it from decoded text
        eos = tokenizer.eos_token or ""
        response_tokens = tokenizer.encode(
            target_response_raw + eos,
            add_special_tokens=False
        )
        n_resp = len(response_tokens)
        if n_resp == 0:
            continue

        # --- Teacher forward pass on teacher context ---
        # teacher context = prompt + feedback + sibling demo → more informed distribution
        teacher_full = mx.array(teacher_tokens + response_tokens)[None]  # (1, T)
        teacher_logits = model(teacher_full)[0]  # (T, V)
        resp_start_teacher = len(teacher_tokens) - 1
        teacher_resp_logits = teacher_logits[resp_start_teacher: resp_start_teacher + n_resp]  # (n_resp, V)
        teacher_resp_lp = nn.log_softmax(teacher_resp_logits, axis=-1)  # (n_resp, V)

        teacher_topk_idx = mx.argsort(-teacher_resp_lp, axis=-1)[:, :top_k]  # (n_resp, top_k)
        teacher_topk_lp = mx.take_along_axis(teacher_resp_lp, teacher_topk_idx, axis=-1)  # (n_resp, top_k)
        mx.eval(teacher_topk_idx, teacher_topk_lp)

        # --- Rollout log probs for IS correction ---
        # IS ratio = π_student(a_t | student_context) / π_rollout(a_t | student_context)
        # rollout policy = teacher (M3) on STUDENT context (original prompt, no feedback/demo)
        # This ensures: at training start (student = M3), IS ratio ≈ 1.0 at all tokens.
        student_full_for_rollout = mx.array(student_tokens + response_tokens)[None]
        student_rollout_logits = model(student_full_for_rollout)[0]  # (T, V) — teacher on student ctx
        resp_start_student = len(student_tokens) - 1
        student_rollout_resp_logits = student_rollout_logits[resp_start_student: resp_start_student + n_resp]
        student_rollout_resp_lp = nn.log_softmax(student_rollout_resp_logits, axis=-1)
        response_mx = mx.array(response_tokens)
        rollout_lp = mx.take_along_axis(
            student_rollout_resp_lp, response_mx[:, None], axis=-1
        ).squeeze(-1)  # (n_resp,) — P_M3(a_t | student_context)
        mx.eval(rollout_lp)

        dataset.append({
            "id": record["id"],
            "split": record["split"],
            "student_tokens": student_tokens,
            "response_tokens": response_tokens,  # needed for IS ratio in compute_jsd_loss
            "n_response": n_resp,
            "teacher_topk_idx": teacher_topk_idx.tolist(),
            "teacher_topk_lp": teacher_topk_lp.tolist(),
            "rollout_log_probs": rollout_lp.tolist(),
            "reward": reward,
            "avg_reward": record["avg_reward"],
            "has_demo": has_demo,
        })

        print(f"  [{rec_idx+1}/{len(rollout_records)}] {record['id']}  "
              f"reward={reward:.2f}  avg={record['avg_reward']:.2f}  has_demo={has_demo}")

    return dataset


# ---------------------------------------------------------------------------
# Phase B — SDPO training with custom MLX JSD loss
# ---------------------------------------------------------------------------

def compute_jsd_loss(
    student_logits: Any,      # (n_resp, V) — has gradient
    teacher_topk_idx: Any,    # (n_resp, top_k) — no grad, int
    teacher_topk_lp: Any,     # (n_resp, top_k) — no grad
    rollout_log_probs: Any,   # (n_resp,) — no grad, π_rollout(a_t) at actual response tokens
    response_token_ids: Any,  # (n_resp,) — no grad, actual response token ids for IS ratio
    alpha: float = 0.5,
    is_clip: float = 2.0,
) -> Any:
    """
    JSD distillation loss over top-k teacher tokens.
    Mirrors compute_self_distillation_loss() from lasgroup/SDPO.

    alpha=0.5 → JSD (symmetric)
    alpha=0.0 → forward KL (student || teacher)
    alpha=1.0 → reverse KL (teacher || student)

    IS correction: weight each token by clip(π_student(a_t) / π_rollout(a_t), max=is_clip)
    where a_t is the actual response token (not teacher's top-1).
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Student log probs — full vocabulary
    student_full_lp = nn.log_softmax(student_logits, axis=-1)  # (n_resp, V)

    # Gather student log probs at teacher top-k positions
    student_topk_lp = mx.take_along_axis(
        student_full_lp, teacher_topk_idx, axis=-1
    )  # (n_resp, top_k)

    # Renormalize both distributions to top-k subspace (Equation 4 in SDPO paper)
    student_topk_lp = student_topk_lp - mx.logsumexp(student_topk_lp, axis=-1, keepdims=True)
    teacher_topk_lp_norm = teacher_topk_lp - mx.logsumexp(teacher_topk_lp, axis=-1, keepdims=True)

    # JSD mixture: M = (1-alpha)*student + alpha*teacher  (in log space)
    log_1_minus_alpha = mx.log(mx.array(1.0 - alpha + 1e-8))
    log_alpha = mx.log(mx.array(alpha + 1e-8))

    log_M = mx.logaddexp(
        student_topk_lp + log_1_minus_alpha,
        teacher_topk_lp_norm + log_alpha,
    )  # (n_resp, top_k)

    # KL(student || M) = sum_k s_k * (log s_k - log M_k)
    student_p = mx.exp(student_topk_lp)
    kl_s_M = mx.sum(student_p * (student_topk_lp - log_M), axis=-1)  # (n_resp,)

    # KL(teacher || M) = sum_k t_k * (log t_k - log M_k)
    teacher_p = mx.exp(teacher_topk_lp_norm)
    kl_t_M = mx.sum(teacher_p * (teacher_topk_lp_norm - log_M), axis=-1)  # (n_resp,)

    # JSD per token
    jsd_per_token = (1.0 - alpha) * kl_s_M + alpha * kl_t_M  # (n_resp,)

    # Importance sampling correction (SDPO repo: is_clip parameter)
    # ratio_t = π_student(a_t) / π_rollout(a_t)  at the actual response token a_t
    student_resp_lp = mx.take_along_axis(
        student_full_lp, response_token_ids[:, None], axis=-1
    ).squeeze(-1)  # (n_resp,)

    log_ratio = mx.clip(student_resp_lp - rollout_log_probs, -20.0, 20.0)
    ratio = mx.exp(log_ratio)
    clipped_ratio = mx.minimum(ratio, mx.array(is_clip))

    jsd_per_token = jsd_per_token * clipped_ratio

    # Sequence mean
    loss = mx.mean(jsd_per_token)
    return loss


def train_sdpo(
    sdpo_dataset: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    config: argparse.Namespace,
    output_dir: Path,
) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only train LoRA weights (lora_a, lora_b) — keep base linear weights frozen
    model.freeze()
    try:
        from mlx_lm.tuner.lora import LoRALinear
        for _, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze()        # unfreeze entire LoRALinear
                module["linear"].freeze()  # re-freeze base linear weight
    except (ImportError, AttributeError, KeyError):
        # fallback: unfreeze last N layers entirely
        try:
            for layer in model.model.layers[-config.lora_layers:]:
                layer.unfreeze()
        except AttributeError:
            model.unfreeze()

    flat_tp = dict(nn.utils.tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for v in flat_tp.values())
    print(f"Trainable parameters: {n_trainable:,}")

    optimizer = optim.Adam(learning_rate=config.learning_rate)

    def loss_fn(mdl, batch):
        student_full = mx.array(batch["student_tokens"] + batch["response_tokens"])[None]
        logits = mdl(student_full)[0]  # (T, V)

        # Response logits: positions [len(student)-1 : len(student)+n_resp-1]
        resp_start = len(batch["student_tokens"]) - 1
        n_resp = batch["n_response"]
        resp_logits = logits[resp_start: resp_start + n_resp]  # (n_resp, V)

        teacher_idx = mx.array(batch["teacher_topk_idx"])
        teacher_lp = mx.array(batch["teacher_topk_lp"])
        rollout_lp = mx.array(batch["rollout_log_probs"])
        response_tok_ids = mx.array(batch["response_tokens"])  # actual response token ids

        return compute_jsd_loss(
            resp_logits, teacher_idx, teacher_lp, rollout_lp, response_tok_ids,
            alpha=config.alpha, is_clip=config.is_clip,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    total_steps = config.epochs * len(sdpo_dataset)
    step = 0
    best_loss = float("inf")

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch in sdpo_dataset:
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            epoch_loss += loss_val
            step += 1

            if step % 5 == 0 or step == 1:
                print(f"  step {step}/{total_steps}  loss={loss_val:.4f}  "
                      f"epoch={epoch+1}/{config.epochs}  "
                      f"{batch['id']}  reward={batch['reward']:.2f}")

        avg_loss = epoch_loss / max(len(sdpo_dataset), 1)
        print(f"Epoch {epoch+1}/{config.epochs}  avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_adapter(model, output_dir / "adapters.safetensors",
                          student_resume=Path(config.student_resume) if hasattr(config, "student_resume") else None)
            print(f"  Saved adapter (best loss so far)")

    _save_adapter(model, output_dir / "adapters.safetensors",
                  student_resume=Path(config.student_resume) if hasattr(config, "student_resume") else None)
    print(f"\nFinal adapter saved → {output_dir}")


def _save_adapter(model: Any, path: Path, student_resume: Path | None = None) -> None:
    """Save LoRA adapter weights using mlx-lm convention."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import shutil
        flat = dict(nn.utils.tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(path), {k: v for k, v in flat.items()})
        # Copy adapter_config.json from resume checkpoint so mlx-lm can load this adapter
        if student_resume is not None:
            cfg_src = student_resume / "adapter_config.json"
            cfg_dst = path.parent / "adapter_config.json"
            if cfg_src.exists() and not cfg_dst.exists():
                shutil.copy2(cfg_src, cfg_dst)
    except Exception as e:
        print(f"  WARNING: could not save adapter: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    try:
        from mlx_lm import load
        import mlx.core as mx
    except ImportError:
        raise SystemExit("mlx-lm not found. Run with python3.11.")

    rollout_path = Path(args.rollouts)
    if not rollout_path.exists():
        raise SystemExit(f"Rollouts not found: {rollout_path}\n"
                         f"Run: python3.11 scripts/run_sdpo_rollouts.py first.")

    rollout_records = [json.loads(l) for l in rollout_path.open() if l.strip()]
    print(f"Loaded {len(rollout_records)} rollout groups from {rollout_path}")

    dataset_path = Path(args.sdpo_dataset)

    # -----------------------------------------------------------------------
    # Phase A: precompute teacher logits (or load cached)
    # -----------------------------------------------------------------------
    if dataset_path.exists() and not args.recompute:
        print(f"Loading cached SDPO dataset from {dataset_path}")
        sdpo_dataset = [json.loads(l) for l in dataset_path.open() if l.strip()]
    else:
        print(f"\n=== Phase A: precomputing teacher logits (adapter: {args.teacher_adapter}) ===")
        teacher_model, teacher_tokenizer = load(args.model, adapter_path=args.teacher_adapter)
        sdpo_dataset = precompute_teacher_logits(
            rollout_records, teacher_model, teacher_tokenizer,
            top_k=args.top_k, success_threshold=args.success_threshold,
        )
        # Cache to disk
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w", encoding="utf-8") as f:
            for s in sdpo_dataset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"SDPO dataset ({len(sdpo_dataset)} samples) saved → {dataset_path}")

        # Unload teacher before loading student (free memory)
        del teacher_model
        mx.clear_cache()

    print(f"\nSDPO dataset: {len(sdpo_dataset)} training samples")
    n_with_demo = sum(1 for s in sdpo_dataset if s.get("has_demo", False))
    avg_reward = np.mean([s["reward"] for s in sdpo_dataset]) if sdpo_dataset else 0.0
    print(f"  With teacher demo: {n_with_demo}/{len(sdpo_dataset)}")
    print(f"  Average best-rollout reward: {avg_reward:.3f}")

    if not sdpo_dataset:
        raise SystemExit("No SDPO training samples — check rollouts file.")

    # -----------------------------------------------------------------------
    # Phase B: train student
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    print(f"\n=== Phase B: SDPO student training (resume: {args.student_resume}) ===")
    student_model, student_tokenizer = load(args.model, adapter_path=args.student_resume)

    train_sdpo(sdpo_dataset, student_model, student_tokenizer, args, output_dir)

    print(f"\nEval with:")
    print(f"  python3.11 scripts/run_baseline.py --backend mlx --model {args.model} --adapter {output_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5: SDPO offline distillation training.")
    p.add_argument("--rollouts", default=str(ROOT / "data" / "sdpo_rollouts.jsonl"))
    p.add_argument("--sdpo-dataset", default=str(ROOT / "data" / "sdpo_dataset.jsonl"),
                   help="Cached dataset with precomputed teacher logits.")
    p.add_argument("--recompute", action="store_true",
                   help="Recompute teacher logits even if cached dataset exists.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--teacher-adapter",
                   default=str(ROOT / "outputs" / "sft_mlx" / "qwen3-4b-feedback-sdft"),
                   help="M3 (Feedback-SDFT) adapter for teacher logit computation.")
    p.add_argument("--student-resume",
                   default=str(ROOT / "outputs" / "sft_mlx" / "qwen3-4b-feedback-sdft"),
                   help="Starting checkpoint for student (default: M3).")
    p.add_argument("--output-dir",
                   default=str(ROOT / "outputs" / "sft_mlx" / "qwen3-4b-sdpo"))
    # Distillation
    p.add_argument("--top-k", type=int, default=20, help="Top-k teacher tokens for JSD.")
    p.add_argument("--alpha", type=float, default=0.5, help="JSD alpha (0.5=JSD, 0=fwd-KL, 1=rev-KL).")
    p.add_argument("--is-clip", type=float, default=2.0, help="Importance sampling clip threshold.")
    p.add_argument("--success-threshold", type=float, default=1.0)
    # Training
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--lora-layers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()

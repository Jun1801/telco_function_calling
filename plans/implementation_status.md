# Implementation Status — Codebase vs Master Plan

> Đối chiếu `plans/contract_guided_vpd_telco_full_plan.md` với code thực tế. Cập nhật khi tiến độ đổi.
> Liên quan: `plans/real_data_generation.md`, `plans/feedback_design.md`, `plans/synthetic_data_strategy_done.md`.

## A. ĐÃ IMPLEMENT (✅)

**Môi trường & đánh giá**
- Schema validator (`src/validation/schema_validator.py`); routing synthetic↔real (`src/evaluation/routing.py`).
- Real evaluator schema-only + gold-diff (source A) + reference-code (source B) + `reveal_gold` (`src/evaluation/real_evaluator.py`).
- Synthetic 4-layer evaluator + contract_checker + mock executor — **legacy, giữ tham chiếu**.
- Reward soft/strict (real_evaluator + `src/reward/reward_feedback.py`); structured feedback + renderer vi/en (`src/reward/feedback_renderer.py`) **đã wire vào SDPO + build_corrections** (lang theo `record.source`).
- `src/evaluation/metrics.py` (arg_key_f1, arg_value_accuracy…) — mức cơ bản.

**Data real (hoàn chỉnh)**
- Registry 26 hàm (`scripts/parse_function_xlsx.py`), catalogue (loc 48 / kpi 12 / unit 9 / station 144).
- Generation: ArgSampler + 7 gen_* families + DLV + split + ReAct decompose + audit + repair
  (`src/generation/real_*`, `scripts/generate_real_data.py`, `build_multistep_react.py`, `audit_real_data.py`, `repair_missing_slot_dates.py`).
- Dataset: **2613 train / ~1195 eval, audit 0 defect**; SFT messages (`scripts/build_real_sft.py`); public warmup; Kaggle mirror.
- Tests **88 pass**; `run_eval` gold **1243/1243**.

**Training hạ tầng (chưa chạy trên real)**
- M0 baseline (`scripts/run_baseline.py`, transformers/mlx, `--splits real`).
- M1 SFT QLoRA (`scripts/train_sft.py`) + MLX (`scripts/train_sft_mlx.py`).
- Rollouts real (`scripts/run_sdpo_rollouts.py --splits real_train/real_eval`); SDPO (`src/training/train_sdpo.py`, `success_threshold`); Feedback-SDFT (`src/training/build_corrections.py` + `scripts/train_feedback_sdft.py`).

## B. CÒN THIẾU (❌ / ⚠️)

| Thành phần (plan) | TT | Ghi chú |
|---|---|---|
| M0/M1 chạy trên real | ⚠️ infra sẵn, chưa chạy | Stage A (Colab A100) |
| M2 masking curriculum | ⚠️ data có (350/170), thiếu script đa-pha | optional ablation |
| M3 Feedback-SDFT real | ⚠️ default mix synthetic | đổi default→real (Stage B) |
| M4 SDPO real | ⚠️ infra sẵn, chưa chạy | Stage B |
| M5/M6 VPD-lite | ❌ `train_vpd.py`/`teacher_update.py` chỉ ở `train_vpd_colab.ipynb` | port→src (Stage C) |
| Progressive scheduler wiring | ❌ `src/reward/scheduler.py` có, chưa gọi | wire M-step (M6) |
| Reward penalties §9.8 | ❌ chưa có (unsafe/hallucinated/deprecated/unnecessary/extra/cost) | reward đầy đủ |
| Hungarian matching §9.2 | ❌ dùng set-by-name; `call_matching.py` không có | parallel/multi metric |
| Metrics aggregation §17 | ❌ chưa có lớp tổng hợp (safety/generalization/distillation/efficiency) | báo cáo |
| configs đọc bởi code | ❌ sdpo/vpd.yaml chưa load; thiếu sft/bandit.yaml | config loader |
| Retrieval §3 (BM25+BGE+reranker) | ❌ không có `retrieval/` | scope sau |
| Bandit M7 §13 | ❌ không có `bandit/` | optional |
| Streamlit demo §18 | ❌ không có `app/` | scope sau |
| Tool-evolution eval (real) | ❌ deprecated/schema_changed/expanded chỉ synthetic | RQ8 sau |
| Đổi tên dự án (Contract-Guided) | ⏳ chờ user | §25 đã flag |

## C. KẾ HOẠCH (MVP §14: M0→M1→M2→M3→M5)

**Stage A — M0 + M1 real (Colab A100) — SẴN SÀNG**
- A1 `build_real_sft` ✅ (2357 domain + 2967 warmup, 0 leak, prompt sạch contract).
- A2 M0 (bf16) → `reports/m0_real.jsonl`; A3 M1 QLoRA §11.2 → adapter; A4 eval → `reports/real_experiments.md`.
- Acceptance: M1 > M0 (seen); unseen không sụt nặng.

**Stage B — M3 Feedback-SDFT + M4 SDPO (A100, sau A)**
- Sửa nhỏ: `train_feedback_sdft --mix-domain` default → real; xác nhận backend GPU (transformers).
- Rollout real → build_corrections → SDFT; SDPO trên rollout real.

**Stage C — M5/M6 VPD-lite (A100, cần PORT)**
- Port `train_vpd_colab.ipynb` → `src/training/{train_vpd,teacher_update}.py` + CLI.
- Wire `scheduler.py` vào M-step (M6 progressive §9.7); đọc `configs/vpd.yaml`.

**Cross-cutting (không chặn MVP)**
- Reward penalties §9.8 + Hungarian `src/reward/call_matching.py` §9.2 — nên làm **trước Stage C** (M6 cần reward đầy đủ).
- `src/evaluation/metrics_aggregate.py` §17 — cho bảng báo cáo cuối.
- Optional/cuối: retrieval, bandit (M7), streamlit demo, tool-evolution eval real.

**Khuyến nghị:** chạy **Stage A ngay** (đã sẵn) → có số M0/M1 thật → quyết B/C. Penalties/Hungarian/metrics làm trước Stage C. Retrieval/bandit/demo để cuối.

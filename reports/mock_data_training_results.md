# Kết quả Training trên Mock/Synthetic Telco Data

> **Phạm vi**: Tổng hợp toàn bộ kết quả thí nghiệm trên **bộ dữ liệu synthetic 82 tools** (telco mô phỏng: read/write/contract/permission) — TRƯỚC khi có data thật từ `Function.xlsx`.
> **Eval set**: 48 samples (`data/sft_eval.jsonl`) trải 14 splits.
> **Metric chính**: `avg reward_total` (= 0.5·soft + 0.5·strict, partial credit). Strict-pass (reward==1.0) báo kèm để tham chiếu.
> **Model**: Qwen3-4B (text-only) + MLX LoRA. Hardware: M3 Pro 18GB.

---

## 1. Experimental Matrix

| ID | Method | Status | avg reward | strict-pass |
|----|--------|--------|-----------|-------------|
| M0 | Prompt-only (zero-shot) | ✅ | 60.8% | — |
| **M1** | **Minimal SFT (2-stage: warmup→domain)** | ✅ | **83.5%** | 81.2% |
| M2 | SFT + masking curriculum | ⏳ chưa chạy | — | — |
| M3 | Feedback-SDFT | ⏳ chưa chạy | — | — |
| M4 | Standard SDPO | ✅ | 47.9% ❌ | — |
| M5 | VPD-lite | ✅ | 80.9% | 79.2% |
| M6 | VPD + progressive reward | ⏳ scheduler đã có, chưa chạy | — | — |
| M7 | M6 + contextual bandit | ⏳ chưa làm | — | — |

**Xếp hạng**: M1 (83.5%) > M5 (80.9%) > M0 (60.8%) > M4 (47.9%).

**MVP tối thiểu** (theo plan §14): M0, M1, M2, M3, M5 — hiện có M0, M1, M5; còn thiếu M2, M3 (code sẵn, chưa đo).

---

## 2. M1 vs M5 — Breakdown theo từng split

| Split | M1 | M5 | Δ (M5−M1) |
|-------|----|----|-----------|
| eval_missing_slot | 72% | **100%** | **+28pp** ✅ |
| eval_seen | 91% | 89% | −2pp |
| eval_unseen | 100% | 100% | = |
| eval_masked_tools | 100% | 100% | = |
| eval_multi_step | 100% | 100% | = |
| eval_parallel | 100% | 100% | = |
| eval_deprecated | 100% | 100% | = |
| eval_abstention | 100% | 100% | = |
| eval_contract | 57% | 43% | **−14pp** ❌ |
| eval_expanded_library | 100% | 74% | **−26pp** ❌ |
| eval_schema_changed | 67% | 33% | **−33pp** ❌ |
| eval_evolution_deprecated | 100% | 100% | = |
| eval_evolution_new_tools | 0% | 0% | = (1 sample) |
| eval_evolution_schema_changed | 0% | 0% | = (1 sample) |

*(Nguồn: `reports/m1_results.jsonl`, `reports/prompt_only_results.jsonl` — file thứ 2 chứa output M5 VPD-lite.)*

---

## 3. Findings chính

### M1 — Minimal SFT (83.5%, baseline mạnh nhất)
- **Two-stage SFT bắt buộc**: warmup (LR 2e-4, 3082 samples, dạy JSON format + abstain/clarify) → domain fine-tune (LR 5e-5, 115 samples, `--resume-adapter-file`, dạy telco contracts).
- Qwen3-4B (text-only) >> Qwen2.5-3B: 72% vs 46.7% ở warmup.
- **Catastrophic forgetting**: naive domain fine-tune trên 38 samples → tụt còn 63% (−9pp). Fix bằng +77 synthetic abstain/clarify → warmup-aug 79.1%.
- Điểm yếu còn lại: `eval_contract` 57%, `eval_schema_changed` 67%, `eval_missing_slot` 72%.

### M4 — Standard SDPO (47.9%, thất bại)
- Train JSD distillation chỉ trên 38 rollout samples (chỉ seen-tool split).
- **Catastrophic forgetting**: student mất khả năng đọc-schema cho unseen/masked tools.
- Nguyên nhân: JSD-only loss trên phân phối hẹp kéo LoRA weights lệch khỏi general capability.
- Kết luận: SDPO đơn thuần không đủ nếu rollout thiếu đa dạng → củng cố sự cần thiết của VPD.

### M5 — VPD-lite (80.9%)
- **E-step học đúng tín hiệu feedback-conditioned**: `eval_missing_slot` +28pp xác nhận teacher học được cách hỏi slot thiếu.
- **Regression** trên `eval_schema_changed` (−33pp), `eval_expanded_library` (−26pp), `eval_contract` (−14pp): do rollout thiếu đa dạng (31 positive samples, không có schema-change scenario trong rollout train).
- Tổng M5 (80.9%) < M1 (83.5%): VPD-lite cần rollout coverage đa dạng mới vượt SFT.
- Train: Colab A100 (PEFT multi-adapter QLoRA), 3 epochs, lr_e=5e-4, β=0.1.
- Adapter: `outputs/vpd/qwen3-4b-vpd/`.

---

## 4. Data assets (mock/synthetic)

| File | Mô tả | Số lượng |
|------|-------|----------|
| `data/tools.json` | Synthetic telco tools | 82 |
| `data/tool_contracts.json` | Business contracts | — |
| `data/mock_telco_db.json` | Subscriber/plan/network state | — |
| `data/sft_train_with_warmup_augmented.jsonl` | Warmup (format + abstain/clarify) | 3,082 |
| `data/sft_train_augmented.jsonl` | Domain (38 telco + 77 synthetic) | 115 |
| `data/sft_eval.jsonl` | Held-out eval | 48 |

---

## 5. Câu hỏi nghiên cứu đã/chưa trả lời

| RQ | Nội dung | Trạng thái |
|----|----------|-----------|
| RQ1 | Gọi đúng tool seen sau SFT? | ✅ M1 eval_seen 91% |
| RQ2 | Gọi unseen tool nhờ đọc schema? | ✅ M1/M5 eval_unseen 100% |
| RQ3 | Masking giảm học thuộc tên? | ✅ eval_masked_tools 100% |
| RQ4 | Contract-aware reward giảm sai nghiệp vụ? | ⚠️ eval_contract còn yếu (57%) |
| RQ5 | Rich feedback > scalar reward? | ⚠️ M5 thắng cục bộ (missing_slot) nhưng tổng chưa vượt M1 |
| RQ6 | VPD > SFT/SDFT/SDPO? | ⚠️ VPD > SDPO (80.9 > 47.9) nhưng VPD < SFT |
| RQ7 | Progressive reward ổn định hơn? | ⏳ scheduler đã code (`src/reward/scheduler.py`), M6 chưa chạy |
| RQ8 | Thích nghi khi API đổi/deprecated? | ⚠️ evolution splits chỉ 1 sample/split, chưa kết luận |
| RQ9 | Bandit routing? | ⏳ chưa làm (M7) |

---

## 6. Lưu ý quan trọng

1. **Eval set nhỏ** (48 samples, một số split chỉ 1-3 samples) → các con số per-split độ tin cậy thống kê hạn chế, đặc biệt `eval_evolution_*` (1 sample/split).
2. Kết quả này **hoàn toàn trên synthetic data**. Data thật từ `Function.xlsx` (324 train + 37 eval, xem `reports/real_data_generation.json`) **chưa được train** — cần wire `real_tools.json` vào pipeline trước khi có điểm trên tool thật.
3. M2/M3 có code sẵn (`masking.py`, `build_corrections.py`, `train_feedback_sdft.py`) nhưng chưa chạy → là gap gần nhất để hoàn thiện MVP matrix.

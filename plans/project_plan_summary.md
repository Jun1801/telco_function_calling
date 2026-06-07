# Project Plan Summary

## Mục tiêu

Xây dựng một **contract-aware telco function-calling agent**: hệ thống có thể đọc tool schema và business contract động, chọn đúng function, tránh call vi phạm điều kiện nghiệp vụ, tự sửa lỗi từ feedback kiểm chứng được, và thích nghi khi tool library thay đổi.

## Vấn đề cần giải quyết

Function calling thông thường thường chỉ tối ưu để sinh JSON đúng format. Project này đi xa hơn: model phải hiểu khi nào một call hợp lệ về schema nhưng vẫn sai về nghiệp vụ, ví dụ thuê bao bị suspended, khách hàng chưa verified, tool đã deprecated, hoặc action có side effect rủi ro.

## Kiến trúc cốt lõi

Pipeline gồm 5 lớp chính:

1. **Tool Registry + Contracts**: định nghĩa telco functions, tham số, enum, required fields, preconditions, permission và side effects.
2. **Verifier Layer**: schema validator, contract checker và mock executor kiểm chứng mọi function call.
3. **Synthetic Data + Benchmark**: sinh train/eval data cho valid calls, missing args, wrong enum/type, ask clarification, abstention, contract violation, unseen tools và evolving tools.
4. **Model Layer**: prompt-only baseline, retrieval-augmented SFT baseline, sau đó feedback-guided self-distillation / SDPO-lite.
5. **Routing + Demo**: contextual bandit chọn strategy gọi hàm; Streamlit demo hiển thị query, selected tools, validation result, feedback và reward.

## MVP Must-Have

- 30-40 telco tools.
- Contract đầy đủ cho 10-15 tools quan trọng.
- Mock telco database.
- Synthetic JSONL datasets: `train`, `eval_seen`, `eval_unseen`, `eval_contract`, `eval_evolution_*`.
- Schema validator, contract checker, mock executor, reward scorer.
- Hybrid retrieval: BM25 + embedding + contract-aware scoring.
- SFT baseline.
- Feedback-guided self-distillation / SDPO-lite.
- PI-SA-CS-LinUCB router.
- Streamlit demo và evaluation report.

## Không làm trong MVP

- Real telco backend.
- PPO/GRPO đầy đủ.
- Neural reward model.
- Multi-agent phức tạp.
- Hàng nghìn tools.
- Gọi external teacher rồi gọi đó là self-distillation.

## Roadmap ngắn

**Phase 1: Verifiable Foundation**
Hoàn thiện registry, contracts, mock DB, validator, checker, executor, reward và tests.

**Phase 2: Dataset MVP**
Mở rộng tool library, sinh synthetic data, tạo seen/unseen/evolution split, kiểm leakage và validate toàn bộ JSONL.

**Phase 3: Retrieval + SFT**
Xây hybrid retriever, chạy prompt-only baseline, train SFT bằng QLoRA, đo function accuracy và Tool Recall@k.

**Phase 4: Feedback Learning**
Rollout SFT, sinh rich feedback, tạo corrected samples, lọc bằng verifier, rồi train Feedback-SDFT / SDPO-lite.

**Phase 5: Router + Demo**
Triển khai bandit router, Streamlit demo, dashboard metrics, ablation table và final report.

## Trạng thái hiện tại

MVP nền đã được khởi tạo: có tools/contracts mẫu, mock DB, generator JSONL, validator, checker, executor, reward scorer, evaluator và tests. Bước tiếp theo là mở rộng lên 30-40 tools và dataset đủ lớn cho SFT baseline.

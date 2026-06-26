# Prompt-Only Baseline Error Analysis

- total_records: 1806
- strict_success: 828/1806
- parse_errors: 1

## Real (read-only KPI) — Reward By Split

- eval_real_abstain: reward=0.988 over 250  (func=0.00 | key_f1=0.00 | val_acc=0.00 | schema=0.00)
- eval_real_masked: reward=0.759 over 300  (func=1.00 | key_f1=0.95 | val_acc=0.92 | schema=0.96)
- eval_real_missing_slot: reward=0.003 over 350  (func=0.00 | key_f1=0.00 | val_acc=0.00 | schema=0.00)
- eval_real_multi_step: reward=0.251 over 156  (func=1.00 | key_f1=0.92 | val_acc=0.41 | schema=0.78)
- eval_real_parallel: reward=0.714 over 150  (func=1.00 | key_f1=0.99 | val_acc=0.89 | schema=0.89)
- eval_real_seen: reward=0.741 over 350  (func=1.00 | key_f1=0.95 | val_acc=0.89 | schema=0.93)
- eval_real_unseen: reward=0.778 over 250  (func=1.00 | key_f1=0.96 | val_acc=0.94 | schema=1.00)

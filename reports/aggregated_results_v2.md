# Aggregated Telco Function Calling Evaluation Results (V2)

This report aggregates and compares the evaluation metrics of all trained models (M0b to M5b) across different evaluation splits of the real Viettel KPI dataset.

## Overall Success Summary

| Model | Strict Success | Accuracy (%) | Parse Errors |
| :--- | :---: | :---: | :---: |
| M0b (Prompt-Only) | 732 / 1786 | 41.0% | 1 |
| M1b (SFT) | 1297 / 1786 | 72.6% | 0 |
| M2b (SFT + Masking) | 1300 / 1786 | 72.8% | 0 |
| M3b (Feedback-SDFT) | 1298 / 1786 | 72.7% | 0 |
| M3b-test (SDFT-test) | 1295 / 1786 | 72.5% | 0 |
| M4b (SDPO+anchor, run 2) | 1326 / 1786 | 74.24% | 0 |
| M5b (VPD-lite) | 1369 / 1786 | 76.65% | 0 |

## Split-by-Split Average Reward Comparison

| Evaluation Split | Count | M0b (Prompt-Only) | M1b (SFT) | M2b (SFT + Masking) | M3b (Feedback-SDFT) | M3b-test (SDFT-test) | M4b (SDPO+anchor, run 1) | M4b (SDPO+anchor, run 2) | M5b (VPD-lite) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `eval_real_abstain` | 75 | 0.987 | 1.000 | 1.000 | 1.000 | 1.000 | 0.973 | 1.000 | 1.000 |
| `eval_real_hard_abstain` | 185 | 0.568 | 0.968 | 0.978 | 0.984 | 0.968 | 0.519 | 1.000 | 1.000 |
| `eval_real_hard_missing` | 134 | 0.007 | 0.749 | 0.751 | 0.748 | 0.748 | 0.770 | 0.756 | 0.749 |
| `eval_real_hard_parallel` | 139 | 0.482 | 0.534 | 0.534 | 0.531 | 0.528 | 0.539 | 0.537 | 0.613 |
| `eval_real_hard_seen` | 272 | 0.682 | 0.830 | 0.834 | 0.830 | 0.828 | 0.806 | 0.809 | 0.821 |
| `eval_real_masked` | 300 | 0.759 | 0.946 | 0.949 | 0.944 | 0.948 | 0.932 | 0.932 | 0.945 |
| `eval_real_missing_slot` | 75 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `eval_real_multi_step` | 156 | 0.251 | 0.543 | 0.547 | 0.543 | 0.549 | 0.530 | 0.531 | 0.593 |
| `eval_real_parallel` | 100 | 0.703 | 0.802 | 0.792 | 0.802 | 0.792 | 0.808 | 0.808 | 0.878 |
| `eval_real_seen` | 100 | 0.728 | 0.938 | 0.927 | 0.938 | 0.938 | 0.875 | 0.870 | 0.896 |
| `eval_real_unseen` | 250 | 0.778 | 0.974 | 0.974 | 0.974 | 0.974 | 0.964 | 0.968 | 0.984 |
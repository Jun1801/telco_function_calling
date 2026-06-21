# `data/` layout

Canonical files are kept **flat** (tests, `scripts/run_eval.py`, and ~24 code paths reference
`data/<file>` directly). Large / external / derived files are **gitignored** (see `.gitignore`),
so `git status` stays clean while the files remain on disk for training.

## Real Viettel KPI (active work)
- `Function.xlsx` — source of truth: 26 KPI functions + reference-code tables
- `real_tools.json`, `real_tool_contracts.json`, `real_reference_codes.json`, `real_station_catalogue.json` — registries/catalogues (built by `scripts/parse_function_xlsx.py`)
- `eval_real_*.jsonl` — 7 eval splits (seen/unseen/masked/missing_slot/multi_step/parallel/abstain)
- `sft_train_real.jsonl` — 2613 train samples
- `real_data/outputs-2/` — Kaggle (vLLM) canonical generation output; `data/*.jsonl` mirrors it + local ReAct decomposition of `multi_step` (gitignored)

## Synthetic 82-tool set (M0–M7 history)
- `tools.json`, `tool_contracts.json`, `mock_telco_db.json`
- `eval_*.jsonl` (14 splits), `sft_train*.jsonl`, `sft_eval.jsonl`, `train.jsonl`

## Warm-up data (public function-calling — gitignored, NOT auto-downloaded)
`public_warmup_*.jsonl` are the **normalized** form of public HF datasets; `sft_train_with_warmup*.jsonl`
are built from them and feed the **warm-up stage** of two-stage SFT. The repo only *normalizes* — it does
**not** download. To rebuild after a fresh clone:

1. Manually download the raw dataset from HuggingFace (by `raw_source`):
   - `toolace` → Team-ACE/ToolACE
   - `hermes_fc` → NousResearch/hermes-function-calling-v1
   - `apigen_mt` → Salesforce/APIGen-MT (xLAM)
   - `xlam` → Salesforce/xlam-function-calling-60k
2. Normalize → `data/public_warmup_<source>.jsonl`:
   ```bash
   python scripts/prepare_public_warmup.py --source <source> --input <raw_file> \
       --output data/public_warmup_<source>.jsonl
   ```
   (Without `--input` the script only writes a 2-row demo subset.)
3. The warm-up SFT files (`sft_train_with_warmup*.jsonl`) are assembled from these via
   `scripts/generate_synthetic_actions.py`.

## Derived / pipeline artifacts (gitignored, safe to delete — regenerated on run)
- `mlx_sft*/` — written by `train_sft_mlx.py --mlx-data-dir` each train run
- `rollouts.jsonl`, `corrections.jsonl`, `sdpo_rollouts.jsonl`, `sdpo_dataset.jsonl` — outputs of rollout/SDPO steps

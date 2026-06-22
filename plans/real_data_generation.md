# Real Data Generation — 26 hàm KPI Viettel

---

## 1. Tổng quan & nguyên tắc

**Triết lý: "construct-then-paraphrase" (ToolACE-adapted).**
Gold (tool + arguments) được **dựng tất định từ catalogue đóng** → *gold không thể sai*; LLM **chỉ viết câu hỏi
tiếng Việt** mô tả lại gold đó. Đảo ngược so với "LLM sinh cả query lẫn call rồi mới verify" → loại bỏ tận gốc
lỗi gold (mã sai, ngày sai, tham số bịa).

26 hàm read-only:
```
seen (18):  tram_nha_mang_khac_province · ke_hoach_trien_khai · vung_phu_province · kqi_province ·
            sub_attached_station · download_throughput_oss · speedtest_province · vung_lom_all · pakh_all ·
            sub_attached_all · regional_station_info · alarm_count · radio_traffic · tong_quan_kpi_vien_thong ·
            top_port · alarm_unresolved · thong_ke_kpi · nguong_kpi
unseen (8): radio_kpi · thong_ke_cntt · top_tram_max · top_tram_min · top_cell_max · top_cell_min ·
            top_sub_attached_max · top_sub_attached_min
```
Catalogue đóng: **location 48 · kpi 12 · unit 9 · station 144** (`EHB000NN`, phủ đủ 48 location).

---

## 2. Luồng end-to-end

```text
data/Function.xlsx
   │  scripts/parse_function_xlsx.py   (enum inject location/kpi/unit/station; KHÔNG object_code)
   ▼
real_tools.json · real_tool_contracts.json (stub) · real_reference_codes.json · real_station_catalogue.json(144)
   │
   ▼  scripts/generate_real_data.py  (orchestrator)
   ├─ ArgSampler.sample(tool, tier, seed)            → gold args hợp lệ + hints VI   (§3)
   ├─ RealToolLLMGenerator.gen_*(...)                 → LLM paraphrase query/family   (§4)
   ├─ DualLayerRealVerifier.run(...)                  → rule + semantic + dedupe       (§5)
   ├─ _hold(rows, k, seed=0)                          → tách EVAL trước (leak-free)
   └─ gen_masking(train_single, eval_seen/unseen)     → masking derive sau split
   ▼
scripts/build_multistep_react.py   → multi_step gold_steps → R1/R2 single-turn (mã trạm THẬT, src/executor/kpi_mock.py)
   ▼
scripts/audit_real_data.py (0 defect)  ·  scripts/repair_missing_slot_dates.py (bug-9)
   ▼
data/sft_train_real.jsonl + data/eval_real_*.jsonl     (canonical: data/real_data/outputs-2/, Kaggle vLLM)
```
Mirror chạy GPU: `kaggle/generate_real_data_vllm.py` (giữ parity với local generator+verifier).

---

## 3. ArgSampler — dựng gold từ catalogue (`src/generation/real_arg_sampler.py`)

- Với mỗi required param → chọn giá trị hợp lệ từ catalogue: location/object (theo tier), kpi, unit, station, date-period, data_level, enum (tech_type/order/rank_by/…).
- Trả `(arguments, hints)`: `hints` = cụm tiếng Việt cho query writer (vd `location_code→"Hà Nội"`, `_time→"quý 2/2023"`). Cặp `from_date/to_date` gói trong một hint `_time`.
- **3 tier độ khó** (`_TIER_MIX` xoay vòng): `simple` (tỉnh lớn HNI/HCM…) · `medium` (tỉnh khác) · `complex` (khu vực/quốc gia/tỉnh hiếm) → đa dạng độ phủ địa lý + thời gian.

---

## 4. Bảy họ generator (`RealToolLLMGenerator`, LLM chỉ paraphrase)

| Family | Cách sinh | Gold / action |
|---|---|---|
| **single_step** | sample args → `_spec(hints)` → `_paraphrase_batch` (batch 15) | `call_function`, gold = args đã sample |
| **missing_slot** | bỏ 1 slot bắt buộc khỏi query (date bỏ **cả cặp**) | `ask_clarification(asked_slots)` + `checker_call` (schema_invalid) |
| **masking** | derive từ single; 4 mode `fn/fn/param/renamed` | `call_function` tên `func_X`/`kpi_query_X`, kèm `masked_tool` (schema nhúng) |
| **parallel** | 3 cặp (PARALLEL_PAIRS), share location/date | `call_functions` = [a, b] |
| **multi_step** | 4 chain (MULTI_STEP_CHAINS) `regional_station_info → dep`; step2 mang `STEP1_REF` | `call_functions` gold_steps → ReAct decompose (R1/R2) |
| **abstain** | LLM sinh câu hỏi NGOÀI phạm vi KPI | `abstain` |
| **from_seed_examples** | dùng query+call expert có sẵn trong xlsx (unseen) | `call_function` cho eval unseen |

PARALLEL_PAIRS: `(kqi_province, download_throughput_oss)`, `(vung_lom_all, pakh_all)`, `(tram_nha_mang_khac_province, vung_phu_province)`.
MULTI_STEP_CHAINS: `regional_station_info →` `{sub_attached_station(station_code), radio_traffic, alarm_count, top_port(object_code)}`.

---

## 5. Chất lượng — Dual-Layer Verification (`src/generation/real_tool_verifier.py`)

**Lớp rule (tất định):**
- SchemaValidator: enum/type/required/pattern.
- `object_code ∈ location ∪ station`; `station_code ∈ 144`; `location/kpi/unit ∈ catalogue`.
- Date `YYYY-MM-DD` + `from_date ≤ to_date`.
- **city↔code word-boundary**: nếu query nêu tên địa danh trong catalogue thì gold location_code phải khớp (regex `\b…\b`, tránh false-match).
- Guard độ dài instruction `≥ 8`.
- Placeholder `<from_step_1>` ở step2 được thay tạm để vẫn kiểm các arg khác.

**Lớp semantic (LLM-judge, tùy chọn `--no-semantic`):** hỏi "query có đủ & đúng thông tin để suy ra call?" / "abstain có thật sự ngoài scope?". **Skip** cho masking / multi_step / missing_slot (đã chứng minh bằng rule).

**Dedupe:** Jaccard 0.85 trong (family, tool) + **cap 3 phrasing** mỗi tổ hợp gold-arg (giúp tool ít tham số không trùng lặp).

**Kiểm chứng độc lập:** `scripts/audit_real_data.py` re-derive mọi invariant (KHÔNG tái dùng verifier) → **0 defect** mọi family (schema/code/date/city/placeholder/masking/missing-slot). `reveal_gold=False` mặc định ở eval (không lộ đáp án).

---

## 6. Phân phối số lượng

### 6.1 Sinh thô (DEFAULTS, `--scale 1.0`, trước verify/split)
```
single seen     : 90 × 18 hàm seen
single unseen   : 26 × 8 hàm  + from_seed_examples
missing_slot    : 90 × 18 hàm seen
parallel        : 110 × 3 cặp
multi_step      : 300 × 4 chain
abstain         : 980
masking         : train 350 · eval seen 110 · eval unseen 60   (derive sau split)
```
(Over-generate để hấp thụ phần rớt khi verify/dedupe; eval **hold trước**, leak-free, seed 0.)

### 6.2 Sau verify + split + ReAct + repair (THỰC TẾ)

| Family | Train | Eval |
|---|---|---|
| single_step_valid | 828 | seen **250** · unseen **150** |
| missing_slot | 430 | **234** |
| multi_step (ReAct R1/R2) | 520 | **156** |
| masking | 350 | **170** |
| abstain | 289 | **155** |
| parallel | 196 | **80** |
| **Tổng** | **2613** | **~1195** |

---

## 7. Đảm bảo chất lượng (tóm tắt)

- ✅ **Gold không thể sai** (dựng từ catalogue; LLM chỉ viết query).
- ✅ DLV rule + semantic + dedupe; audit độc lập **0 defect**.
- ✅ Multi-step R2 dùng **mã trạm thật** từ observation (geographic-consistent), không placeholder lọt vào SFT gold.
- ✅ Masking nhúng schema `func_X` + shadow tool thật (test RQ3 schema-only).
- ✅ Eval `reveal_gold=False` (teacher/eval không leak đáp án); 7 split tách sạch khỏi train.
- ✅ bug-9 (missing_slot date-pair) đã repair tất định; 8 mẫu mơ hồ (còn năm cụ thể) bị drop.
- ✅ Regression: **86 tests pass**; `run_eval` gold self-check **1243/1243**.

---

## 8. Local vs Kaggle
- **Local (MLX, Qwen3-4B):** `generate_real_data.py` — tiện cô lập từng family; M3 chậm/nhiệt cho quy mô lớn.
- **Kaggle (vLLM/transformers):** `kaggle/generate_real_data_vllm.py` — sinh bộ canonical `data/real_data/outputs-2/` (mọi mẫu `generator="vllm"`). Giữ **parity** với local (cùng ArgSampler/verifier/_GLOSS; đã sync `.strip()`, station-code guard, word-boundary, date-pair).

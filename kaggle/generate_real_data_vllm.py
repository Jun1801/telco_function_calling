"""Self-contained vLLM data generator for real Viettel KPI tools (Kaggle/CUDA).

Same methodology as the local MLX pipeline (ToolACE-adapted, construct-then-
paraphrase) but the LLM backend is vLLM (fast batched CUDA generation). Gold args
are constructed from the closed catalogues (always valid); the model only writes a
natural Vietnamese query per spec — ALL prompts are generated in ONE vLLM batch.

Inputs (Kaggle Dataset dir): real_tools.json, real_reference_codes.json,
real_station_catalogue.json.
Outputs (/kaggle/working): sft_train_real.jsonl + eval_real_*.jsonl + report.

Env vars:
  GEN_MODEL   HuggingFace model ID (default: Qwen/Qwen3-32B-Instruct)
  SCALE       raw generation multiplier for train families (default: 1.0)
  EVAL_SCALE  multiplier for eval hold-out targets (default: 1.0)
  BACKEND       vllm | transformers (default: transformers)
  BATCH         batch size for transformers backend (default: 4)
  QUANTIZATION  vLLM quantization method: "bitsandbytes" for 4-bit on A100 40GB,
                unset for fp16 on A100 80GB / H100 (default: unset)
  DATA_DIR      path to input data (default: /kaggle/input/telco-real-tools)
  OUT_DIR       output path (default: /kaggle/working)
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/kaggle/input/telco-real-tools"))
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen3-32B-Instruct")
SCALE = float(os.environ.get("SCALE", "1.0"))
EVAL_SCALE = float(os.environ.get("EVAL_SCALE", "1.0"))
# QUANTIZATION: "bitsandbytes" for 4-bit (A100 40GB), None for fp16 (A100 80GB / H100).
QUANTIZATION = os.environ.get("QUANTIZATION", None) or None

STEP1_REF = "<from_step_1>"
TIER_MIX = ["simple", "simple", "simple", "medium", "medium", "medium", "medium", "complex", "complex", "complex"]

# Volume targets at scale=1.0 (before rule verify). Over-generated ~1.25x.
TARGET = dict(seen_single=90, unseen_single=26, missing=90, parallel_per_pair=110, multi_per_chain=300, abstain=980,
              eval_seen=250, eval_unseen=150, eval_missing=240, eval_multi=155, eval_parallel=80, eval_abstain=155,
              mask_train=350, mask_eval_seen=110, mask_eval_unseen=60)

# Parallel pairs: both tools share location+time context (or kpi_code for thong_ke/nguong).
PARALLEL_PAIRS = [
    ("kqi_province",              "download_throughput_oss"),
    ("vung_lom_all",              "pakh_all"),
    ("tram_nha_mang_khac_province", "vung_phu_province"),
    ("kqi_province",              "sub_attached_all"),          # KQI + thuê bao, share location+time
    ("download_throughput_oss",   "speedtest_province"),        # 2 nguồn throughput cùng tỉnh
    ("thong_ke_kpi",              "nguong_kpi"),                # thống kê + ngưỡng, share kpi_code
]

# Multi-step chains: all start with regional_station_info → dependent tool uses result key.
MULTI_CHAINS = [
    ("regional_station_info", "sub_attached_station", "station_code"),
    ("regional_station_info", "radio_traffic",        "object_code"),
    ("regional_station_info", "alarm_count",          "object_code"),
    ("regional_station_info", "top_port",             "object_code"),
    ("regional_station_info", "alarm_unresolved",     "object_code"),  # trạm → alarm chưa xử lý
]

# ============================== ArgSampler ==============================
_DATA_LEVEL_VN = {"day": "ngày", "week": "tuần", "month": "tháng", "quarter": "quý", "year": "năm"}
_PERIODS = {
    "simple": [("2026-06-01", "2026-06-30", "tháng 6/2026"), ("2026-01-01", "2026-12-31", "năm 2026"),
               ("2025-12-01", "2025-12-31", "tháng 12/2025"), ("2026-03-01", "2026-03-31", "tháng 3/2026"),
               ("2025-01-01", "2025-12-31", "năm 2025"), ("2026-01-01", "2026-03-31", "quý 1/2026")],
    "medium": [("2024-05-01", "2024-05-31", "tháng 5/2024"), ("2024-09-01", "2024-09-30", "tháng 9/2024"),
               ("2023-10-01", "2023-10-31", "tháng 10/2023"), ("2024-02-01", "2024-02-29", "tháng 2/2024"),
               ("2025-03-01", "2025-03-31", "tháng 3/2025"), ("2024-11-01", "2024-11-30", "tháng 11/2024"),
               ("2023-07-01", "2023-07-31", "tháng 7/2023"), ("2025-08-01", "2025-08-31", "tháng 8/2025"),
               ("2024-01-01", "2024-03-31", "quý 1/2024"), ("2024-04-01", "2024-06-30", "quý 2/2024"),
               ("2024-07-01", "2024-09-30", "quý 3/2024"), ("2024-10-01", "2024-12-31", "quý 4/2024"),
               ("2025-01-01", "2025-03-31", "quý 1/2025"), ("2023-04-01", "2023-06-30", "quý 2/2023"),
               ("2025-04-01", "2025-06-30", "quý 2/2025"), ("2023-06-01", "2023-06-30", "tháng 6/2023")],
    "complex": [("2022-01-01", "2022-12-31", "năm 2022"), ("2023-01-01", "2023-12-31", "năm 2023"),
                ("2024-01-01", "2024-12-31", "năm 2024"), ("2021-01-01", "2021-12-31", "năm 2021"),
                ("2024-07-15", "2024-08-20", "từ 15/7 đến 20/8/2024"), ("2023-01-01", "2023-06-30", "6 tháng đầu năm 2023"),
                ("2024-07-01", "2024-12-31", "6 tháng cuối năm 2024"), ("2022-03-10", "2022-05-25", "từ 10/3 đến 25/5/2022"),
                ("2025-02-01", "2025-04-30", "từ tháng 2 đến tháng 4/2025"), ("2023-09-15", "2023-11-30", "từ 15/9 đến 30/11/2023"),
                ("2022-06-01", "2022-06-30", "tháng 6/2022"), ("2021-07-01", "2021-09-30", "quý 3/2021"),
                ("2024-12-01", "2025-02-28", "từ tháng 12/2024 đến 2/2025"), ("2023-11-01", "2023-11-30", "tháng 11/2023")],
}
_DLV_TIER = {"simple": ["month", "quarter"], "medium": ["month", "week", "quarter"],
             "complex": ["quarter", "year", "day", "week", "month"]}
_TID = {"simple": 1, "medium": 2, "complex": 3}
_GLOSS = {
    "tech_type": {"2G": "2G", "3G": "3G", "4G": "4G", "5G": "5G", "all": "tất cả công nghệ"},
    "network_provider": {"viettel": "Viettel", "vinaphone": "Vinaphone", "mobifone": "Mobifone"},
    "speedtest_provider": {"ookla": "Ookla", "ispeed": "iSpeed"},
    "type_station": {"4g_vtt": "4G VTT", "4g_vtnet": "4G VTNet", "5g_vtnet": "5G VTNet", "all": "tất cả loại trạm"},
    "vendor": {"zte": "ZTE", "vttek": "VTTek", "ericsson": "Ericsson", "viettel": "Viettel", "nokia": "Nokia", "huawei": "Huawei"},
    "station_type": {"macro": "trạm macro", "smallcell": "small cell", "inbuilding": "in-building", "iot": "IoT", "rru": "RRU", "femtocell": "femtocell"},
    "order": {"max": "cao nhất", "min": "thấp nhất"},
    "rank_by": {"used": "lượng đã dùng", "total": "tổng", "performance": "hiệu năng"},
    "scope": {"station": "theo trạm", "district": "theo quận/huyện", "province": "theo tỉnh", "area": "theo khu vực"},
    "cell_type": {"2g": "2G", "3g": "3G", "4g": "4G", "5g": "5G"},
    "fault_level_name": {"critical": "nghiêm trọng", "major": "lớn", "minor": "nhỏ"},
}


class ArgSampler:
    def __init__(self, references, stations):
        self.references = references
        self.stations = stations
        self.loc_name = {}
        seen = set()
        provinces, regions, countries = [], [], []
        for loc in references["location_code"]:
            self.loc_name.setdefault(loc["code"], loc["name"])
            if loc["code"] in seen:
                continue
            seen.add(loc["code"])
            g = loc.get("group", "")
            (provinces if g == "Tỉnh/Thành phố Việt Nam" else regions if g == "Khu vực"
             else countries if g == "Quốc gia" else []).append(loc["code"])
        self._loc = {"simple": ["VNM", "HNI", "HCM", "DNG", "HPG", "CTO", "NAN", "THA", "QNH", "BNH", "KHA", "LDG"],
                     "medium": [c for c in provinces if c not in ("HNI", "HCM", "DNG")],
                     "complex": regions + countries + provinces[-8:]}
        self.kpis = references["kpi_code"]
        self.units = references["unit_code"]

    def _loc_pick(self, tier, rng):
        pool = self._loc.get(tier) or self._loc["medium"]
        code = rng.choice(pool or ["VNM"])
        return code, self.loc_name.get(code, code)

    def _station(self, loc, rng):
        pool = [s for s in self.stations if s["location_code"] == loc] or self.stations
        s = rng.choice(pool)
        return s["station_code"], s["name"]

    def sample(self, tool, tier, idx):
        rng = random.Random(idx * 97 + _TID[tier] * 7 + hash(tool["name"]) % 1000)
        props = tool["parameters"].get("properties", {})
        required = list(tool["parameters"].get("required", []))
        args, hints = {}, {}
        period = rng.choice(_PERIODS[tier])
        level = rng.choice(_DLV_TIER[tier])
        loc_code = None
        for p in required:
            spec = props.get(p, {})
            if p in ("location_code", "object_code"):
                code, name = self._loc_pick(tier, rng)
                loc_code = code; args[p] = code; hints[p] = name
            elif p == "from_date":
                args[p] = period[0]
            elif p == "to_date":
                args[p] = period[1]; hints["_time"] = period[2]
            elif p == "date":
                args[p] = period[0]; hints["_time"] = period[2]
            elif p in ("data_level", "time_type"):
                lv = level if level in spec.get("enum", [level]) else (spec.get("enum") or ["month"])[0]
                args[p] = lv; hints["_granularity"] = _DATA_LEVEL_VN.get(lv, lv)
            elif p == "station_code":
                code, name = self._station(loc_code, rng); args[p] = code; hints[p] = name
            elif p == "kpi_code":
                k = rng.choice(self.kpis); args[p] = k["code"]; hints[p] = k["meaning"]
            elif p == "unit_code":
                u = rng.choice(self.units); args[p] = u["code"]; hints[p] = u["name"]
            elif p == "top_k":
                args[p] = {"simple": 5, "medium": 10, "complex": 20}[tier]; hints["_topk"] = f"top {args[p]}"
            elif "enum" in spec:
                opts = [v for v in spec["enum"] if v != ""]
                if opts:
                    v = rng.choice(opts); args[p] = v
                    h = _GLOSS.get(p, {}).get(v, v)
                    if h:
                        hints[p] = h
                else:
                    args[p] = spec.get("default", "")
            elif spec.get("type") == "integer":
                args[p] = 10
            elif "default" in spec:
                args[p] = spec["default"]
            else:
                args[p] = "VNM"
        return {"arguments": args, "hints": hints}


# ============================== rule verify ==============================
# Full SchemaValidator parity with src/validation/schema_validator.py.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TYPE_MAP = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


def _schema_issues(tool, args):
    if tool is None:
        return ["unknown_tool"]
    if tool.get("deprecated"):
        return ["deprecated_tool"]
    schema = tool.get("parameters", {})
    props = schema.get("properties", {})
    issues = []
    for name in schema.get("required", []):
        if name not in args:
            issues.append("missing_arg")
    for name in args:
        if name not in props:
            issues.append("unknown_arg")
    for name, value in args.items():
        if name not in props or STEP1_REF in str(value):  # placeholder resolved at runtime
            continue
        spec = props[name]
        et = spec.get("type")
        if et in _TYPE_MAP and not isinstance(value, _TYPE_MAP[et]):
            issues.append("invalid_type")
        if "enum" in spec and value not in spec["enum"]:
            issues.append("invalid_enum")
        if spec.get("type") == "string" and "pattern" in spec and isinstance(value, str) \
                and re.fullmatch(spec["pattern"], value) is None:
            issues.append("pattern_mismatch")
    return issues


def _schema_ok(tool, args):
    return not _schema_issues(tool, args)


# ============================== spec → query prompt ==============================
def hints_spec(tool, hints):
    desc = tool["description"].split(".")[0][:90]
    parts = []
    for p in ("location_code", "object_code"):
        if p in hints:
            parts.append(f"địa điểm: {hints[p]}")
    if "station_code" in hints:
        parts.append(f"trạm: {hints['station_code']}")
    if "_time" in hints:
        parts.append(f"thời gian: {hints['_time']}")
    if "_granularity" in hints:
        parts.append(f"mức thời gian: {hints['_granularity']}")
    if "_topk" in hints:
        parts.append(hints["_topk"])
    if "kpi_code" in hints:
        parts.append(f"chỉ số: {hints['kpi_code']}")
    if "unit_code" in hints:
        parts.append(f"đơn vị: {hints['unit_code']}")
    skip = {"location_code", "object_code", "station_code", "_time", "_granularity", "_topk", "kpi_code", "unit_code"}
    for k, v in hints.items():
        if k not in skip:
            parts.append(str(v))
    return f"{desc}. Phải nêu: " + "; ".join(parts)


SYS_PARA = ("Bạn đóng vai người dùng hỏi trợ lý phân tích mạng viễn thông Viettel. "
            "Viết MỘT câu hỏi tiếng Việt tự nhiên chứa ĐÚNG và ĐỦ thông tin được nêu, "
            "không thêm thông tin khác. Chỉ trả về câu hỏi, không giải thích.")
SYS_ABSTAIN = ("Sinh MỘT câu hỏi tiếng Việt NGOÀI phạm vi hệ thống chỉ tra cứu KPI/chỉ số/hạ tầng "
               "mạng viễn thông (vd: đăng ký gói cước, khiếu nại hoá đơn, đổi SIM, thời tiết, nấu ăn). "
               "Chỉ trả về câu hỏi.")


def main():
    print(f"DATA={DATA} OUT={OUT} MODEL={MODEL} SCALE={SCALE} EVAL_SCALE={EVAL_SCALE}", flush=True)
    tools = json.loads((DATA / "real_tools.json").read_text())
    refs = json.loads((DATA / "real_reference_codes.json").read_text())
    stations = json.loads((DATA / "real_station_catalogue.json").read_text())
    tbn = {t["name"]: t for t in tools}
    seen = [t for t in tools if t["split"] == "seen"]
    unseen = [t for t in tools if t["split"] == "unseen"]
    sampler = ArgSampler(refs, stations)

    _EVAL_KEYS = {"eval_seen", "eval_unseen", "eval_missing", "eval_multi",
                  "eval_parallel", "eval_abstain", "mask_eval_seen", "mask_eval_unseen"}
    C = {k: max(1, int(v * (EVAL_SCALE if k in _EVAL_KEYS else SCALE))) for k, v in TARGET.items()}

    rng = random.Random(0)
    valid_loc = {i["code"] for i in refs["location_code"]}
    valid_sta = {s["station_code"] for s in stations}
    name2code = {i["name"].lower().strip(): i["code"] for i in refs["location_code"] if len(i["name"].strip()) >= 4}
    name2pat = [(re.compile(rf"\b{re.escape(nm)}\b"), code) for nm, code in name2code.items()]

    # ---- build jobs (prompt + builder) ----
    jobs = []  # each: {"user","system","builder","family"}

    def add_single(tool, n, split):
        for i in range(n):
            s = sampler.sample(tool, TIER_MIX[i % len(TIER_MIX)], i + (0 if split != "unseen" else 99000))
            spec = hints_spec(tool, s["hints"])
            args = s["arguments"]

            def b(q, tool=tool, args=args, i=i, split=split):
                return {"id": f"real_{tool['name']}_single_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "valid_kpi_read", "scenario_family": "single_step_valid",
                        "instruction": q, "expected_action": "call_function",
                        "generator": "vllm", "gold_call": {"tool_name": tool["name"], "arguments": args}}
            jobs.append({"user": f"Viết câu hỏi cho: {spec}", "system": SYS_PARA, "builder": b,
                         "family": "single_step_valid", "_split": split})

    for t in seen:
        add_single(t, C["seen_single"], "seen")
    for t in unseen:
        add_single(t, C["unseen_single"], "unseen")

    # missing_slot (seen)
    for t in seen:
        req = t["parameters"].get("required", [])
        # from_date/to_date share one time phrase → drop the pair together (parity
        # with src verifier; avoids declaring one missing while query drops both).
        date_slots = [d for d in ("from_date", "to_date") if d in req]
        drop = (["_dates"] if date_slots else []) + [s for s in ("location_code", "kpi_code", "station_code") if s in req]
        if not drop:
            continue
        for i in range(C["missing"]):
            s = sampler.sample(t, TIER_MIX[i % len(TIER_MIX)], i + 5000)
            slot = drop[i % len(drop)]
            h = dict(s["hints"])
            if slot == "_dates":
                dropped, label = date_slots, "khoảng thời gian"
                h.pop("_time", None)
            else:
                dropped, label = [slot], slot
                h.pop(slot, None)
            spec = hints_spec(t, h) + f" (KHÔNG nêu {label})"
            checker = {k: v for k, v in s["arguments"].items() if k not in dropped}

            def b(q, t=t, dropped=dropped, checker=checker, i=i):
                return {"id": f"real_{t['name']}_missing_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "missing_parameter", "scenario_family": "missing_slot",
                        "instruction": q, "expected_action": "ask_clarification",
                        "generator": "vllm", "missing_slots": dropped,
                        "prediction": {"action": "ask_clarification", "asked_slots": dropped},
                        "checker_call": {"tool_name": t["name"], "arguments": checker},
                        "checker_expected_status": "schema_invalid"}
            jobs.append({"user": f"Viết câu hỏi cho: {spec}", "system": SYS_PARA, "builder": b,
                         "family": "missing_slot", "_split": "seen"})

    # parallel
    pairs = [(tbn[a], tbn[b]) for a, b in PARALLEL_PAIRS if a in tbn and b in tbn]
    for pi, (a, bb) in enumerate(pairs):
        for i in range(C["parallel_per_pair"]):
            tier = TIER_MIX[i % len(TIER_MIX)]
            sa = sampler.sample(a, tier, i + 1000)
            shared = {k: sa["arguments"][k] for k in ("location_code", "from_date", "to_date", "data_level", "kpi_code") if k in sa["arguments"]}
            sb = sampler.sample(bb, tier, i + 2000)
            for k, v in shared.items():
                if k in sb["arguments"]:
                    sb["arguments"][k] = v
            h = {k: v for k, v in sa["hints"].items() if k.startswith("_") or k in ("location_code", "object_code", "kpi_code")}
            spec = (f"Hỏi cả: (1){a['description'].split('.')[0][:55]}; (2){bb['description'].split('.')[0][:55]}. "
                    + hints_spec(a, h).split("Phải nêu:")[-1])

            def b(q, a=a, bb=bb, sa=sa, sb=sb, pi=pi, i=i):
                return {"id": f"real_parallel_{pi}_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "parallel_reads", "scenario_family": "parallel", "instruction": q,
                        "expected_action": "call_functions", "generator": "vllm",
                        "gold_calls": [{"tool_name": a["name"], "arguments": sa["arguments"]},
                                       {"tool_name": bb["name"], "arguments": sb["arguments"]}]}
            jobs.append({"user": f"Viết câu hỏi cho: {spec}", "system": SYS_PARA, "builder": b,
                         "family": "parallel", "_split": "seen"})

    # multi_step
    chains = [(tbn[s], tbn[d], k) for s, d, k in MULTI_CHAINS if s in tbn and d in tbn]
    for ci, (src, dep, key) in enumerate(chains):
        for i in range(C["multi_per_chain"]):
            tier = TIER_MIX[i % len(TIER_MIX)]
            s1 = sampler.sample(src, tier, i + 3000)
            loc = s1["hints"].get("location_code") or s1["hints"].get("object_code") or "Hà Nội"
            tm = s1["hints"].get("_time", "")
            s2 = sampler.sample(dep, tier, i + 4000)["arguments"]; s2[key] = STEP1_REF
            spec = (f"Quy trình 2 bước: tìm các trạm ở {loc}" + (f" trong {tm}" if tm else "")
                    + f", rồi tra cứu {dep['description'].split('.')[0][:55]} của các trạm đó")

            def b(q, src=src, dep=dep, s1=s1, s2=s2, ci=ci, i=i):
                return {"id": f"real_multi_{ci}_{i:04d}", "source": "real_tool_xlsx", "scenario": "dependency",
                        "scenario_family": "multi_step", "instruction": q,
                        "expected_action": "call_functions", "generator": "vllm",
                        "gold_steps": [{"tool_name": src["name"], "arguments": s1["arguments"]},
                                       {"tool_name": dep["name"], "arguments": s2, "depends_on_previous": True}]}
            jobs.append({"user": spec, "system": SYS_PARA, "builder": b, "family": "multi_step", "_split": "seen"})

    # abstain
    for i in range(C["abstain"]):
        def b(q, i=i):
            return {"id": f"real_abstain_{i:04d}", "source": "real_tool_xlsx", "scenario": "irrelevance",
                    "scenario_family": "abstain", "instruction": q,
                    "expected_action": "abstain", "generator": "vllm",
                    "prediction": {"action": "abstain", "reason": "ngoài phạm vi công cụ KPI"}}
        jobs.append({"user": f"Sinh 1 câu hỏi ngoài phạm vi (mẫu {i}).", "system": SYS_ABSTAIN, "builder": b,
                     "family": "abstain", "_split": "seen"})

    print(f"Total jobs (prompts): {len(jobs)}", flush=True)

    # ---- LLM batch generate ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def templ(system, user):
        try:
            return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)

    prompts = [templ(j["system"], j["user"]) for j in jobs]
    backend = os.environ.get("BACKEND", "transformers").lower()
    print(f"Backend: {backend} | model: {MODEL} | generating {len(prompts)} queries ...", flush=True)

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        _llm_kwargs: dict = dict(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.90, trust_remote_code=True)
        if QUANTIZATION:
            _llm_kwargs["quantization"] = QUANTIZATION
        else:
            _llm_kwargs["dtype"] = "half"
        llm = LLM(**_llm_kwargs)
        outs = llm.generate(prompts, SamplingParams(temperature=0.85, max_tokens=90, stop=["\n\n"]))
        texts = [o.outputs[0].text for o in outs]
    else:  # transformers — robust on Kaggle (no runtime compilation)
        import torch
        from transformers import AutoModelForCausalLM
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                     device_map="cuda", trust_remote_code=True)
        model.eval()
        B = int(os.environ.get("BATCH", "4"))  # 32B model: small batch to avoid OOM
        texts = []
        for i in range(0, len(prompts), B):
            chunk = prompts[i:i + B]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=90, do_sample=True, temperature=0.85, top_p=0.9,
                                   pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                texts.append(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
            if i % (B * 10) == 0:
                print(f"  generated {i + len(chunk)}/{len(prompts)}", flush=True)

    def clean(text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = text.strip().strip('"').strip()
        return text.split("\n")[0].strip()

    samples = [j["builder"](clean(t)) for j, t in zip(jobs, texts)]

    # unseen expert seed examples (deterministic, no LLM) — parity with local from_seed_examples
    for t in unseen:
        for i, ex in enumerate(t.get("seed_examples", [])):
            if ex.get("query") and ex.get("call"):
                samples.append({"id": f"real_{t['name']}_seed_{i:03d}", "source": "real_tool_xlsx",
                                "scenario": "valid_unseen_tool", "scenario_family": "single_step_valid",
                                "instruction": ex["query"].strip(),
                                "expected_action": "call_function", "generator": "seed",
                                "gold_call": {"tool_name": t["name"], "arguments": copy.deepcopy(ex["call"])}})

    # ---- rule verify ----
    def all_calls(s):
        if s.get("gold_call"):
            return [s["gold_call"]]
        return s.get("gold_calls") or s.get("gold_steps") or []

    def tool_for(s, name):
        mt = s.get("masked_tool")
        if mt and mt["name"] == name:
            return mt
        return tbn.get(name)

    def rule_ok(s):
        if not s["instruction"] or len(s["instruction"]) < 8:
            return False
        if s["expected_action"] == "ask_clarification":
            c = s["checker_call"]; return not _schema_ok(tbn.get(c["tool_name"]), c["arguments"])
        if s["expected_action"] == "abstain":
            return True
        calls = all_calls(s)
        for c in calls:
            tool = tool_for(s, c["tool_name"])
            if tool is None or not _schema_ok(tool, c["arguments"]):
                return False
            a = c["arguments"]
            oc = a.get("object_code")
            if oc is not None and STEP1_REF not in str(oc) and oc not in valid_loc and oc not in valid_sta:
                return False
            sc = a.get("station_code")
            if sc is not None and STEP1_REF not in str(sc) and sc not in valid_sta:
                return False
            for dk in ("from_date", "to_date", "date"):
                v = a.get(dk)
                if v is not None and STEP1_REF not in str(v) and not _DATE_RE.match(str(v)):
                    return False
        # city↔code word-boundary check
        q = s["instruction"].lower()
        matched = {code for pat, code in name2pat if pat.search(q)}
        gl = {c["arguments"].get("location_code") for c in calls if c["arguments"].get("location_code")}
        if matched and gl and not all(g in matched for g in gl):
            return False
        return True

    kept = [s for s in samples if rule_ok(s)]

    # dedupe: Jaccard 0.85 within (family, primary tool) — parity with src/generation/real_tool_verifier.py
    def _toks(t):
        return set(re.findall(r"\w+", t.lower()))

    def _jac(a, b):
        return len(a & b) / len(a | b) if a and b else 0.0

    def _arg_sig(s):
        cs = (s.get("gold_calls") or s.get("gold_steps")
              or ([s["gold_call"]] if s.get("gold_call") else [])
              or ([s["checker_call"]] if s.get("checker_call") else []))
        return json.dumps([[c.get("tool_name"), sorted(c.get("arguments", {}).items())] for c in cs],
                          sort_keys=True, default=str)

    groups = {}; arg_count = Counter(); ded = []
    MAX_PER_ARGS = 3  # cap phrasings per identical gold-arg combo — keep diversity (Option A)
    for s in kept:
        pt = (all_calls(s)[0]["tool_name"] if all_calls(s) else s.get("checker_call", {}).get("tool_name", ""))
        key = (s["scenario_family"], pt)
        sig = str(key) + "|" + _arg_sig(s)
        if s["scenario_family"] != "abstain" and arg_count[sig] >= MAX_PER_ARGS:
            continue
        toks = _toks(s["instruction"])
        if any(_jac(toks, t) > 0.85 for t in groups.get(key, [])):
            continue
        groups.setdefault(key, []).append(toks); arg_count[sig] += 1; ded.append(s)
    print(f"Generated {len(samples)} -> rule_ok {len(kept)} -> dedup {len(ded)}", flush=True)

    by_fam = {}
    for s in ded:
        by_fam.setdefault(s["scenario_family"], []).append(s)

    def hold(rows, k):
        rows = list(rows); rng.shuffle(rows); return rows[:k], rows[k:]

    single_seen = [s for s in by_fam.get("single_step_valid", []) if s["gold_call"]["tool_name"] in tbn and tbn[s["gold_call"]["tool_name"]]["split"] == "seen"]
    single_unseen = [s for s in by_fam.get("single_step_valid", []) if s["gold_call"]["tool_name"] in tbn and tbn[s["gold_call"]["tool_name"]]["split"] == "unseen"]
    eval_seen, train_single = hold(single_seen, C["eval_seen"])
    eval_unseen = single_unseen[: C["eval_unseen"]]
    eval_missing, train_missing = hold(by_fam.get("missing_slot", []), C["eval_missing"])
    eval_multi, train_multi = hold(by_fam.get("multi_step", []), C["eval_multi"])
    eval_parallel, train_parallel = hold(by_fam.get("parallel", []), C["eval_parallel"])
    eval_abstain, train_abstain = hold(by_fam.get("abstain", []), C["eval_abstain"])

    # masking derive (deterministic) from train single + eval single (leak-free)
    def mask(base_list, n, tag):
        out = []
        for i, base in enumerate(base_list[:n]):
            tool = tbn.get(base["gold_call"]["tool_name"])
            if not tool:
                continue
            mode = ("fn", "fn", "param", "renamed")[i % 4]
            mn = f"func_{i+1}" if mode != "renamed" else f"kpi_query_{i+1}"
            props = tool["parameters"].get("properties", {})
            base_args = base["gold_call"]["arguments"]
            if mode == "param":
                km = {k: f"param_{j+1}" for j, k in enumerate(props)}
                mp = {"type": "object", "required": [km[k] for k in tool["parameters"].get("required", []) if k in km],
                      "properties": {km[k]: copy.deepcopy(v) for k, v in props.items()}}
                gargs = {km[k]: v for k, v in base_args.items() if k in km}
                mt = {"name": mn, "description": tool["description"], "parameters": mp, "status": "active", "deprecated": False}
            else:
                gargs = copy.deepcopy(base_args)
                desc = ("Hàm tra cứu: " + tool["description"]) if mode == "renamed" else tool["description"]
                mt = {"name": mn, "description": desc, "parameters": copy.deepcopy(tool["parameters"]), "status": "active", "deprecated": False}
            out.append({"id": f"real_mask_{tag}_{mode}_{i:04d}", "source": "real_tool_xlsx", "scenario": f"masking_{mode}",
                        "scenario_family": "masking", "instruction": f"Dùng {mn} để: {base['instruction']}",
                        "expected_action": "call_function", "generator": "vllm",
                        "gold_call": {"tool_name": mn, "arguments": gargs}, "masked_tool": mt})
        return out

    mask_train = [s for s in mask(train_single, C["mask_train"], "tr") if rule_ok(s)]
    mask_eval = [s for s in (mask(eval_seen, C["mask_eval_seen"], "ev")
                             + mask(eval_unseen, C["mask_eval_unseen"], "evu")) if rule_ok(s)]

    train = train_single + train_missing + train_multi + train_parallel + train_abstain + mask_train
    for s in train:
        s["split"] = "train"
    evalsets = {"eval_real_seen": eval_seen, "eval_real_unseen": eval_unseen, "eval_real_masked": mask_eval,
                "eval_real_missing_slot": eval_missing, "eval_real_multi_step": eval_multi,
                "eval_real_parallel": eval_parallel, "eval_real_abstain": eval_abstain}
    for name, rows in evalsets.items():
        for s in rows:
            s["split"] = name

    OUT.mkdir(parents=True, exist_ok=True)

    def w(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    w(OUT / "sft_train_real.jsonl", train)
    for name, rows in evalsets.items():
        w(OUT / f"{name}.jsonl", rows)

    report = {"train_total": len(train), "train_by_family": dict(Counter(s["scenario_family"] for s in train)),
              "eval_total": sum(len(r) for r in evalsets.values()),
              "eval_counts": {k: len(v) for k, v in evalsets.items()},
              "model": MODEL, "scale": SCALE, "eval_scale": EVAL_SCALE}
    (OUT / "real_data_generation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("=== SUMMARY ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

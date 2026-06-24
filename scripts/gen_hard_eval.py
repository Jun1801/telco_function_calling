"""Generate hard eval splits only (does not touch train data or existing easy eval).

Outputs (4 files):
  eval_real_hard_abstain.jsonl   200 samples (hard_negative + irrelevance)
  eval_real_hard_seen.jsonl      ~300 samples (rare locations + disambiguation + no-overlap)
  eval_real_hard_missing.jsonl   ~300 samples (multi-slot + natural omission + rare slots)
  eval_real_hard_parallel.jsonl  150 samples (implicit multi-tool, no "cả hai" marker)

Env vars:
  GEN_MODEL    HuggingFace model ID (default: Qwen/Qwen3-32B)
  BACKEND      vllm | transformers (default: transformers)
  BATCH        batch size for transformers backend (default: 8)
  DATA_DIR     path to real_tools.json etc. (default: <project>/data)
  OUT_DIR      output directory (default: <project>/data)
  QUANTIZATION vLLM quantization (e.g. bitsandbytes for 40GB GPU)
  HARD_SCALE   float multiplier for target counts (default: 1.0)
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", str(_ROOT / "data")))
OUT = Path(os.environ.get("OUT_DIR", str(_ROOT / "data")))
MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen3-32B")
HARD_SCALE = float(os.environ.get("HARD_SCALE", "1.0"))
QUANTIZATION = os.environ.get("QUANTIZATION", None) or None

TIER_MIX = ["simple", "simple", "simple", "medium", "medium", "medium", "medium", "complex", "complex", "complex"]

# Targets at HARD_SCALE=1.0
T = {
    "hard_abstain_consumer": 50,
    "hard_abstain_revenue": 50,
    "hard_abstain_coverage": 50,
    "hard_abstain_irr": 50,
    "hard_seen_rare": 100,
    "hard_seen_disambig": 100,
    "hard_seen_nooverlap": 100,   # final count; jobs over-generated 1.6x
    "hard_missing_multi": 150,
    "hard_missing_natural": 100,
    "hard_missing_rare": 50,
    "hard_parallel_implicit": 150,
}

PARALLEL_PAIRS = [
    ("kqi_province",               "download_throughput_oss"),
    ("vung_lom_all",               "pakh_all"),
    ("tram_nha_mang_khac_province","vung_phu_province"),
    ("kqi_province",               "sub_attached_all"),
    ("download_throughput_oss",    "speedtest_province"),
    ("thong_ke_kpi",               "nguong_kpi"),
]

DISAMBIG_PAIRS = [
    ("download_throughput_oss", "speedtest_province",
     "tốc độ tải về từ hệ thống OSS nội bộ Viettel",
     "kết quả đo Speedtest ookla/iSpeed từ thiết bị đầu cuối"),
    ("kqi_province", "tong_quan_kpi_vien_thong",
     "chỉ số KQI (Key Quality Indicator) mạng di động",
     "tổng quan KPI viễn thông toàn bộ hạ tầng"),
    ("vung_lom_all", "vung_phu_province",
     "vùng lõm sóng (điểm yếu sóng, dead zone)",
     "vùng phủ sóng (diện tích phủ)"),
]

_DATA_LEVEL_VN = {"day": "ngày", "week": "tuần", "month": "tháng", "quarter": "quý", "year": "năm"}
_PERIODS = {
    "simple": [("2026-06-01","2026-06-30","tháng 6/2026"),("2026-01-01","2026-12-31","năm 2026"),
               ("2025-12-01","2025-12-31","tháng 12/2025"),("2026-03-01","2026-03-31","tháng 3/2026"),
               ("2025-01-01","2025-12-31","năm 2025"),("2026-01-01","2026-03-31","quý 1/2026")],
    "medium": [("2024-05-01","2024-05-31","tháng 5/2024"),("2024-09-01","2024-09-30","tháng 9/2024"),
               ("2023-10-01","2023-10-31","tháng 10/2023"),("2024-02-01","2024-02-29","tháng 2/2024"),
               ("2025-03-01","2025-03-31","tháng 3/2025"),("2024-11-01","2024-11-30","tháng 11/2024"),
               ("2023-07-01","2023-07-31","tháng 7/2023"),("2025-08-01","2025-08-31","tháng 8/2025"),
               ("2024-01-01","2024-03-31","quý 1/2024"),("2024-04-01","2024-06-30","quý 2/2024"),
               ("2024-07-01","2024-09-30","quý 3/2024"),("2024-10-01","2024-12-31","quý 4/2024"),
               ("2025-01-01","2025-03-31","quý 1/2025"),("2023-04-01","2023-06-30","quý 2/2023"),
               ("2025-04-01","2025-06-30","quý 2/2025"),("2023-06-01","2023-06-30","tháng 6/2023")],
    "complex": [("2022-01-01","2022-12-31","năm 2022"),("2023-01-01","2023-12-31","năm 2023"),
                ("2024-01-01","2024-12-31","năm 2024"),("2021-01-01","2021-12-31","năm 2021"),
                ("2024-07-15","2024-08-20","từ 15/7 đến 20/8/2024"),("2023-01-01","2023-06-30","6 tháng đầu năm 2023"),
                ("2024-07-01","2024-12-31","6 tháng cuối năm 2024"),("2022-03-10","2022-05-25","từ 10/3 đến 25/5/2022"),
                ("2025-02-01","2025-04-30","từ tháng 2 đến tháng 4/2025"),("2023-09-15","2023-11-30","từ 15/9 đến 30/11/2023"),
                ("2022-06-01","2022-06-30","tháng 6/2022"),("2021-07-01","2021-09-30","quý 3/2021"),
                ("2024-12-01","2025-02-28","từ tháng 12/2024 đến 2/2025"),("2023-11-01","2023-11-30","tháng 11/2023")],
}
_DLV_TIER = {"simple": ["month","quarter"], "medium": ["month","week","quarter"],
             "complex": ["quarter","year","day","week","month"]}
_TID = {"simple": 1, "medium": 2, "complex": 3, "rare": 4}
_GLOSS = {
    "tech_type": {"2G":"2G","3G":"3G","4G":"4G","5G":"5G","all":"tất cả công nghệ"},
    "network_provider": {"viettel":"Viettel","vinaphone":"Vinaphone","mobifone":"Mobifone"},
    "speedtest_provider": {"ookla":"Ookla","ispeed":"iSpeed"},
    "type_station": {"4g_vtt":"4G VTT","4g_vtnet":"4G VTNet","5g_vtnet":"5G VTNet","all":"tất cả loại trạm"},
    "vendor": {"zte":"ZTE","vttek":"VTTek","ericsson":"Ericsson","viettel":"Viettel","nokia":"Nokia","huawei":"Huawei"},
    "station_type": {"macro":"trạm macro","smallcell":"small cell","inbuilding":"in-building","iot":"IoT","rru":"RRU","femtocell":"femtocell"},
    "order": {"max":"cao nhất","min":"thấp nhất"},
    "rank_by": {"used":"lượng đã dùng","total":"tổng","performance":"hiệu năng"},
    "scope": {"station":"theo trạm","district":"theo quận/huyện","province":"theo tỉnh","area":"theo khu vực"},
    "cell_type": {"2g":"2G","3g":"3G","4g":"4G","5g":"5G"},
    "fault_level_name": {"critical":"nghiêm trọng","major":"lớn","minor":"nhỏ"},
}


class ArgSampler:
    def __init__(self, references, stations):
        self.references = references
        self.stations = stations
        self.loc_name = {}
        seen_codes: set = set()
        provinces, regions, countries = [], [], []
        for loc in references["location_code"]:
            self.loc_name.setdefault(loc["code"], loc["name"])
            if loc["code"] in seen_codes:
                continue
            seen_codes.add(loc["code"])
            g = loc.get("group", "")
            (provinces if g == "Tỉnh/Thành phố Việt Nam" else
             regions if g == "Khu vực" else
             countries if g == "Quốc gia" else []).append(loc["code"])
        self._loc = {
            "simple":  ["VNM","HNI","HCM","DNG","HPG","CTO","NAN","THA","QNH","BNH","KHA","LDG"],
            "medium":  [c for c in provinces if c not in ("HNI","HCM","DNG")],
            "complex": regions + countries + provinces[-8:],
            "rare":    countries + regions,   # country + region codes only
        }
        self.kpis  = references["kpi_code"]
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
        t_id = _TID.get(tier, 3)
        rng = random.Random(idx * 97 + t_id * 7 + hash(tool["name"]) % 1000)
        period = rng.choice(_PERIODS.get(tier, _PERIODS["complex"]))
        level  = rng.choice(_DLV_TIER.get(tier, _DLV_TIER["complex"]))
        props    = tool["parameters"].get("properties", {})
        required = list(tool["parameters"].get("required", []))
        args, hints = {}, {}
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
                args[p] = {"simple": 5, "medium": 10}.get(tier, 20)
                hints["_topk"] = f"top {args[p]}"
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
_DATE_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TYPE_MAP = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "object": dict, "array": list}
_STEP1_REF = "<from_step_1>"


def _schema_issues(tool, args):
    if tool is None:
        return ["unknown_tool"]
    if tool.get("deprecated"):
        return ["deprecated_tool"]
    schema = tool.get("parameters", {})
    props  = schema.get("properties", {})
    issues = []
    for name in schema.get("required", []):
        if name not in args:
            issues.append("missing_arg")
    for name in args:
        if name not in props:
            issues.append("unknown_arg")
    for name, value in args.items():
        if name not in props or _STEP1_REF in str(value):
            continue
        spec = props[name]
        et = spec.get("type")
        if et in _TYPE_MAP and not isinstance(value, _TYPE_MAP[et]):
            issues.append("invalid_type")
        if "enum" in spec and value not in spec["enum"]:
            issues.append("invalid_enum")
        if (spec.get("type") == "string" and "pattern" in spec
                and isinstance(value, str) and re.fullmatch(spec["pattern"], value) is None):
            issues.append("pattern_mismatch")
    return issues


def _schema_ok(tool, args):
    return not _schema_issues(tool, args)


def hints_spec(tool, hints):
    desc  = tool["description"].split(".")[0][:90]
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
    skip = {"location_code","object_code","station_code","_time","_granularity","_topk","kpi_code","unit_code"}
    for k, v in hints.items():
        if k not in skip:
            parts.append(str(v))
    return f"{desc}. Phải nêu: " + "; ".join(parts)


# ============================== system prompts ==============================
SYS_PARA = ("Bạn đóng vai người dùng hỏi trợ lý phân tích mạng viễn thông Viettel. "
            "Viết MỘT câu hỏi tiếng Việt tự nhiên chứa ĐÚNG và ĐỦ thông tin được nêu, "
            "không thêm thông tin khác. Chỉ trả về câu hỏi, không giải thích.")

SYS_ABSTAIN_CONSUMER = (
    "Sinh MỘT câu hỏi tiếng Việt ngắn theo góc nhìn nhân viên kỹ thuật viễn thông "
    "về DỊCH VỤ KHÁCH HÀNG: đăng ký/hủy gói cước, kiểm tra hóa đơn, đổi SIM, "
    "báo mất SIM, kích hoạt eSIM. "
    "Câu hỏi nghe kỹ thuật nhưng là provisioning/billing — "
    "KHÔNG thể trả lời bằng công cụ tra cứu KPI. Chỉ câu hỏi.")

SYS_ABSTAIN_REVENUE = (
    "Sinh MỘT câu hỏi tiếng Việt về chỉ số KINH DOANH không trong hệ thống KPI mạng: "
    "doanh thu, chi phí vận hành, điểm CSAT/NPS, số khiếu nại, ngân sách triển khai. "
    "Nghe như KPI nhưng là tài chính/kinh doanh ngoài phạm vi. Chỉ câu hỏi.")

SYS_ABSTAIN_COVERAGE = (
    "Sinh MỘT câu hỏi về chất lượng sóng từ góc nhìn NGƯỜI DÙNG CÁ NHÂN: "
    "sóng tại địa chỉ nhà/văn phòng cụ thể, cuộc gọi tại tòa nhà, báo mất sóng GPS. "
    "Hệ thống chỉ có KPI tỉnh/khu vực, không theo địa chỉ hộ gia đình. Chỉ câu hỏi.")

SYS_ABSTAIN_IRR = (
    "Sinh MỘT câu hỏi tiếng Việt NGOÀI phạm vi hệ thống tra cứu KPI/hạ tầng mạng "
    "(vd: nấu ăn, du lịch, học tập, thời tiết). Chỉ câu hỏi.")

SYS_IMPLICIT = (
    "Bạn đóng vai người dùng hỏi trợ lý phân tích mạng viễn thông Viettel. "
    "Viết MỘT câu hỏi tiếng Việt tự nhiên kết hợp hai nhu cầu dưới đây thành một câu liền mạch. "
    "KHÔNG dùng các từ: 'đồng thời', 'cả hai', 'cùng lúc', 'vừa...vừa', 'song song'. "
    "Câu phải chứa đủ thông tin để thực hiện cả hai tra cứu. Chỉ trả về câu hỏi, không giải thích.")


def _call_sig(tool_name, arguments):
    return json.dumps([tool_name, sorted(arguments.items())], sort_keys=True, default=str)


def main():
    print(f"DATA={DATA} OUT={OUT} MODEL={MODEL} HARD_SCALE={HARD_SCALE}", flush=True)
    tools    = json.loads((DATA / "real_tools.json").read_text())
    refs     = json.loads((DATA / "real_reference_codes.json").read_text())
    stations = json.loads((DATA / "real_station_catalogue.json").read_text())
    tbn      = {t["name"]: t for t in tools}
    seen_tools = [t for t in tools if t["split"] == "seen"]
    sampler  = ArgSampler(refs, stations)
    targets  = {k: max(1, int(v * HARD_SCALE)) for k, v in T.items()}

    valid_loc  = {i["code"] for i in refs["location_code"]}
    valid_sta  = {s["station_code"] for s in stations}
    name2code  = {i["name"].lower().strip(): i["code"] for i in refs["location_code"] if len(i["name"].strip()) >= 4}
    name2pat   = [(re.compile(rf"\b{re.escape(nm)}\b"), code) for nm, code in name2code.items()]

    # Load train arg-sigs for hard_seen_nooverlap
    train_sigs: set = set()
    train_path = DATA / "sft_train_real.jsonl"
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                gc = json.loads(line).get("gold_call")
                if gc:
                    train_sigs.add(_call_sig(gc["tool_name"], gc.get("arguments", {})))
        print(f"Loaded {len(train_sigs)} train arg-sigs", flush=True)

    rng  = random.Random(42)
    jobs = []

    # ---- H1: hard_abstain (4 sub-types × 50) ----
    for sub_key, n, sys_p, scenario in [
        ("hard_abstain_consumer", targets["hard_abstain_consumer"], SYS_ABSTAIN_CONSUMER, "hard_negative_consumer"),
        ("hard_abstain_revenue",  targets["hard_abstain_revenue"],  SYS_ABSTAIN_REVENUE,  "hard_negative_revenue"),
        ("hard_abstain_coverage", targets["hard_abstain_coverage"], SYS_ABSTAIN_COVERAGE, "hard_negative_coverage"),
        ("hard_abstain_irr",      targets["hard_abstain_irr"],      SYS_ABSTAIN_IRR,      "hard_irrelevance"),
    ]:
        for i in range(n):
            def b(q, sub_key=sub_key, scenario=scenario, i=i):
                return {"id": f"real_{sub_key}_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": scenario, "scenario_family": "hard_abstain",
                        "instruction": q, "expected_action": "abstain", "generator": "vllm",
                        "prediction": {"action": "abstain", "reason": "ngoài phạm vi công cụ KPI"}}
            jobs.append({"user": f"Sinh câu hỏi (mẫu {i}).", "system": sys_p,
                         "builder": b, "family": "hard_abstain"})

    # ---- H2a: hard_seen_rare (rare location codes) ----
    rare_per_tool = max(1, targets["hard_seen_rare"] // max(1, len(seen_tools)))
    for t in seen_tools:
        for i in range(rare_per_tool):
            s    = sampler.sample(t, "rare", i + 20000)
            spec = hints_spec(t, s["hints"])
            args = s["arguments"]
            def b(q, t=t, args=args, i=i):
                return {"id": f"real_hard_seen_rare_{t['name']}_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "hard_rare_location", "scenario_family": "hard_seen",
                        "instruction": q, "expected_action": "call_function", "generator": "vllm",
                        "gold_call": {"tool_name": t["name"], "arguments": args}}
            jobs.append({"user": f"Viết câu hỏi cho: {spec}", "system": SYS_PARA,
                         "builder": b, "family": "hard_seen"})

    # ---- H2b: hard_seen_disambig (3 pairs × ~33) ----
    per_pair_d = max(1, targets["hard_seen_disambig"] // max(1, len(DISAMBIG_PAIRS)))
    for di, (a_name, b_name, a_desc, b_desc) in enumerate(DISAMBIG_PAIRS):
        a_tool = tbn.get(a_name)
        if not a_tool:
            continue
        for i in range(per_pair_d):
            tier = TIER_MIX[i % len(TIER_MIX)]
            s    = sampler.sample(a_tool, tier, i + 21000 + di * 1000)
            loc  = s["hints"].get("location_code") or s["hints"].get("object_code") or "Hà Nội"
            tm   = s["hints"].get("_time", "")
            spec = (f"Viết câu hỏi về '{a_desc}' tại {loc}"
                    + (f" trong {tm}" if tm else "")
                    + f". Phân biệt rõ với '{b_desc}' — câu hỏi phải chỉ rõ đây là {a_desc}.")
            args = s["arguments"]
            def b(q, a_tool=a_tool, args=args, di=di, i=i):
                return {"id": f"real_hard_seen_disambig_{di}_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "hard_disambiguation", "scenario_family": "hard_seen",
                        "instruction": q, "expected_action": "call_function", "generator": "vllm",
                        "gold_call": {"tool_name": a_tool["name"], "arguments": args}}
            jobs.append({"user": spec, "system": SYS_PARA, "builder": b, "family": "hard_seen"})

    # ---- H2c: hard_seen_nooverlap (medium/complex, filter vs train after inference) ----
    nooverlap_gen   = max(1, int(targets["hard_seen_nooverlap"] * 1.6))
    nooverlap_final = targets["hard_seen_nooverlap"]
    nooverlap_per_t = max(1, nooverlap_gen // max(1, len(seen_tools)))
    for t in seen_tools:
        for i in range(nooverlap_per_t):
            tier = ["medium", "complex"][i % 2]
            s    = sampler.sample(t, tier, i + 22000)
            spec = hints_spec(t, s["hints"])
            args = s["arguments"]
            sig  = _call_sig(t["name"], args)
            def b(q, t=t, args=args, sig=sig, i=i):
                return {"id": f"real_hard_seen_nooverlap_{t['name']}_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "hard_no_train_overlap", "scenario_family": "hard_seen",
                        "instruction": q, "expected_action": "call_function", "generator": "vllm",
                        "gold_call": {"tool_name": t["name"], "arguments": args},
                        "_arg_sig": sig, "_filter_train_overlap": True}
            jobs.append({"user": f"Viết câu hỏi cho: {spec}", "system": SYS_PARA,
                         "builder": b, "family": "hard_seen"})

    # ---- H3 helpers ----
    def get_multi_drops(tool):
        """Return list of (dropped_slots, label) for 2+ slot drops."""
        req       = tool["parameters"].get("required", [])
        has_dates = "from_date" in req and "to_date" in req
        loc_slot  = "location_code" if "location_code" in req else ("object_code" if "object_code" in req else None)
        combos    = []
        if has_dates and loc_slot:
            combos.append((["from_date", "to_date", loc_slot], "thời gian và địa điểm"))
        if has_dates:
            combos.append((["from_date", "to_date"], "khoảng thời gian"))
        if "kpi_code" in req and loc_slot:
            combos.append(([loc_slot, "kpi_code"], "địa điểm và chỉ số KPI"))
        if "station_code" in req and has_dates:
            combos.append((["station_code", "from_date", "to_date"], "trạm và thời gian"))
        if "kpi_code" in req and "unit_code" in req:
            combos.append((["kpi_code", "unit_code"], "chỉ số và đơn vị"))
        return combos

    # H3a: multi-slot missing with explicit "(KHÔNG nêu)" marker
    n_multi = max(1, targets["hard_missing_multi"] // max(1, len(seen_tools)))
    for t in seen_tools:
        combos = get_multi_drops(t)
        if not combos:
            continue
        for i in range(n_multi):
            s        = sampler.sample(t, TIER_MIX[i % len(TIER_MIX)], i + 30000)
            dropped, label = combos[i % len(combos)]
            h        = dict(s["hints"])
            for slot in dropped:
                h.pop(slot, None)
            if any(d in dropped for d in ("from_date", "to_date")):
                h.pop("_time", None)
            spec    = hints_spec(t, h) + f" (KHÔNG nêu {label})"
            checker = {k: v for k, v in s["arguments"].items() if k not in dropped}
            def b(q, t=t, dropped=dropped, checker=checker, i=i):
                return {"id": f"real_hard_missing_multi_{t['name']}_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "hard_missing_multi", "scenario_family": "hard_missing",
                        "instruction": q, "expected_action": "ask_clarification", "generator": "vllm",
                        "missing_slots": dropped,
                        "prediction": {"action": "ask_clarification", "asked_slots": dropped},
                        "checker_call": {"tool_name": t["name"], "arguments": checker},
                        "checker_expected_status": "schema_invalid"}
            jobs.append({"user": f"Viết câu hỏi cho: {spec}", "system": SYS_PARA,
                         "builder": b, "family": "hard_missing"})

    # H3b: natural omission — NO explicit marker in spec
    n_nat = max(1, targets["hard_missing_natural"] // max(1, len(seen_tools)))
    for t in seen_tools:
        combos = get_multi_drops(t)
        if not combos:
            continue
        for i in range(n_nat):
            s        = sampler.sample(t, TIER_MIX[i % len(TIER_MIX)], i + 31000)
            dropped, _ = combos[i % len(combos)]
            h        = dict(s["hints"])
            for slot in dropped:
                h.pop(slot, None)
            if any(d in dropped for d in ("from_date", "to_date")):
                h.pop("_time", None)
            spec    = hints_spec(t, h)   # NO "(KHÔNG nêu ...)" — natural omission
            checker = {k: v for k, v in s["arguments"].items() if k not in dropped}
            def b(q, t=t, dropped=dropped, checker=checker, i=i):
                return {"id": f"real_hard_missing_nat_{t['name']}_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "hard_missing_natural", "scenario_family": "hard_missing",
                        "instruction": q, "expected_action": "ask_clarification", "generator": "vllm",
                        "missing_slots": dropped,
                        "prediction": {"action": "ask_clarification", "asked_slots": dropped},
                        "checker_call": {"tool_name": t["name"], "arguments": checker},
                        "checker_expected_status": "schema_invalid"}
            jobs.append({"user": f"Viết câu hỏi cho: {spec}", "system": SYS_PARA,
                         "builder": b, "family": "hard_missing"})

    # H3c: rare slots (station_code, kpi_code, unit_code)
    rare_slot_pairs = [(t, slot) for t in seen_tools
                       for slot in ("station_code", "kpi_code", "unit_code")
                       if slot in t["parameters"].get("required", [])]
    if rare_slot_pairs:
        n_rare = max(1, targets["hard_missing_rare"] // len(rare_slot_pairs))
        slot_labels = {"station_code": "mã trạm", "kpi_code": "chỉ số KPI", "unit_code": "đơn vị"}
        for t, slot in rare_slot_pairs:
            for i in range(n_rare):
                s       = sampler.sample(t, TIER_MIX[i % len(TIER_MIX)], i + 32000)
                h       = dict(s["hints"]); h.pop(slot, None)
                spec    = hints_spec(t, h) + f" (KHÔNG nêu {slot_labels[slot]})"
                checker = {k: v for k, v in s["arguments"].items() if k != slot}
                def b(q, t=t, slot=slot, checker=checker, i=i):
                    return {"id": f"real_hard_missing_rare_{t['name']}_{slot}_{i:04d}", "source": "real_tool_xlsx",
                            "scenario": "hard_missing_rare_slot", "scenario_family": "hard_missing",
                            "instruction": q, "expected_action": "ask_clarification", "generator": "vllm",
                            "missing_slots": [slot],
                            "prediction": {"action": "ask_clarification", "asked_slots": [slot]},
                            "checker_call": {"tool_name": t["name"], "arguments": checker},
                            "checker_expected_status": "schema_invalid"}
                jobs.append({"user": f"Viết câu hỏi cho: {spec}", "system": SYS_PARA,
                             "builder": b, "family": "hard_missing"})

    # ---- H4: hard_parallel_implicit ----
    pairs = [(tbn[a], tbn[b]) for a, b in PARALLEL_PAIRS if a in tbn and b in tbn]
    per_pair_p = max(1, targets["hard_parallel_implicit"] // max(1, len(pairs)))
    for pi, (a, bb) in enumerate(pairs):
        for i in range(per_pair_p):
            tier = TIER_MIX[i % len(TIER_MIX)]
            sa   = sampler.sample(a, tier, i + 40000 + pi * 1000)
            shared = {k: sa["arguments"][k] for k in ("location_code","from_date","to_date","data_level","kpi_code")
                      if k in sa["arguments"]}
            sb = sampler.sample(bb, tier, i + 41000 + pi * 1000)
            for k, v in shared.items():
                if k in sb["arguments"]:
                    sb["arguments"][k] = v
            h = {k: v for k, v in sa["hints"].items()
                 if k.startswith("_") or k in ("location_code", "object_code", "kpi_code")}
            a_metric  = a["description"].split(".")[0][:55]
            b_metric  = bb["description"].split(".")[0][:55]
            loc_hint  = h.get("location_code", h.get("object_code", ""))
            time_hint = h.get("_time", "")
            spec = (f"Cần biết: (1) {a_metric}; (2) {b_metric}."
                    + (f" Tại: {loc_hint}." if loc_hint else "")
                    + (f" Thời gian: {time_hint}." if time_hint else "")
                    + " KHÔNG dùng 'cả hai' hay 'đồng thời' trong câu.")
            def b(q, a=a, bb=bb, sa=sa, sb=sb, pi=pi, i=i):
                return {"id": f"real_hard_parallel_impl_{pi}_{i:04d}", "source": "real_tool_xlsx",
                        "scenario": "hard_parallel_implicit", "scenario_family": "hard_parallel",
                        "instruction": q, "expected_action": "call_functions", "generator": "vllm",
                        "gold_calls": [{"tool_name": a["name"], "arguments": sa["arguments"]},
                                       {"tool_name": bb["name"], "arguments": sb["arguments"]}]}
            jobs.append({"user": spec, "system": SYS_IMPLICIT, "builder": b, "family": "hard_parallel"})

    print(f"Total jobs: {len(jobs)}", flush=True)

    # ---- LLM inference ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def templ(system, user):
        try:
            return tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)

    prompts = [templ(j["system"], j["user"]) for j in jobs]
    backend = os.environ.get("BACKEND", "transformers").lower()
    print(f"Backend: {backend} | {len(prompts)} prompts", flush=True)

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        kw: dict = dict(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.88,
                        trust_remote_code=True, enable_prefix_caching=True, tensor_parallel_size=1)
        if QUANTIZATION:
            kw["quantization"] = QUANTIZATION
        else:
            kw["dtype"] = "bfloat16"
        llm  = LLM(**kw)
        outs = llm.generate(prompts, SamplingParams(temperature=0.85, max_tokens=120, stop=["\n\n"]))
        texts = [o.outputs[0].text for o in outs]
    else:
        import torch
        from transformers import AutoModelForCausalLM
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                   device_map="cuda", trust_remote_code=True)
        mdl.eval()
        B = int(os.environ.get("BATCH", "8"))
        texts = []
        for i in range(0, len(prompts), B):
            chunk = prompts[i:i + B]
            enc   = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=2048).to(mdl.device)
            with torch.no_grad():
                g = mdl.generate(**enc, max_new_tokens=90, do_sample=True, temperature=0.85,
                                 top_p=0.9, pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                texts.append(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
            if i % (B * 10) == 0:
                print(f"  {i + len(chunk)}/{len(prompts)}", flush=True)

    def clean(text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text.strip().strip('"').strip().split("\n")[0].strip()

    samples = [j["builder"](clean(t)) for j, t in zip(jobs, texts)]

    # ---- rule verify ----
    def all_calls(s):
        if s.get("gold_call"):   return [s["gold_call"]]
        if s.get("gold_calls"):  return s["gold_calls"]
        return []

    def rule_ok(s):
        if not s.get("instruction") or len(s["instruction"]) < 8:
            return False
        if s["expected_action"] == "ask_clarification":
            c = s["checker_call"]
            return not _schema_ok(tbn.get(c["tool_name"]), c["arguments"])
        if s["expected_action"] == "abstain":
            return True
        calls = all_calls(s)
        for c in calls:
            tool = tbn.get(c["tool_name"])
            if tool is None or not _schema_ok(tool, c["arguments"]):
                return False
            a  = c["arguments"]
            oc = a.get("object_code")
            if oc is not None and _STEP1_REF not in str(oc) and oc not in valid_loc and oc not in valid_sta:
                return False
            sc = a.get("station_code")
            if sc is not None and _STEP1_REF not in str(sc) and sc not in valid_sta:
                return False
            for dk in ("from_date", "to_date", "date"):
                v = a.get(dk)
                if v is not None and _STEP1_REF not in str(v) and not _DATE_RE.match(str(v)):
                    return False
        q       = s["instruction"].lower()
        matched = {code for pat, code in name2pat if pat.search(q)}
        gl      = {c["arguments"].get("location_code") for c in calls if c["arguments"].get("location_code")}
        if matched and gl and not all(g in matched for g in gl):
            return False
        return True

    kept = [s for s in samples if rule_ok(s)]
    print(f"Generated {len(samples)} -> rule_ok {len(kept)}", flush=True)

    # ---- dedup (Jaccard 0.85 within family+tool) ----
    def _toks(t): return set(re.findall(r"\w+", t.lower()))
    def _jac(a, b): return len(a & b) / len(a | b) if a and b else 0.0

    def _arg_sig_s(s):
        cs = all_calls(s) or ([s["checker_call"]] if s.get("checker_call") else [])
        return json.dumps([[c.get("tool_name"), sorted(c.get("arguments", {}).items())] for c in cs],
                          sort_keys=True, default=str)

    groups:    dict    = {}
    arg_count: Counter = Counter()
    ded = []
    MAX_PER_ARGS = 3
    for s in kept:
        calls = all_calls(s)
        pt    = (calls[0]["tool_name"] if calls
                 else s.get("checker_call", {}).get("tool_name", ""))
        key   = (s["scenario_family"], pt)
        sig   = str(key) + "|" + _arg_sig_s(s)
        if s["scenario_family"] != "hard_abstain" and arg_count[sig] >= MAX_PER_ARGS:
            continue
        tok_set = _toks(s["instruction"])
        if any(_jac(tok_set, t) > 0.85 for t in groups.get(key, [])):
            continue
        groups.setdefault(key, []).append(tok_set)
        arg_count[sig] += 1
        ded.append(s)
    print(f"After dedup: {len(ded)}", flush=True)

    # ---- apply nooverlap filter ----
    final:           list = []
    nooverlap_kept:  int  = 0
    for s in ded:
        if s.get("_filter_train_overlap"):
            sig = s.pop("_arg_sig", "")
            s.pop("_filter_train_overlap", None)
            if sig in train_sigs:
                continue
            if nooverlap_kept >= nooverlap_final:
                continue
            nooverlap_kept += 1
        final.append(s)

    # ---- split to 4 output files ----
    out_sets = {
        "eval_real_hard_abstain":  [s for s in final if s["scenario_family"] == "hard_abstain"],
        "eval_real_hard_seen":     [s for s in final if s["scenario_family"] == "hard_seen"],
        "eval_real_hard_missing":  [s for s in final if s["scenario_family"] == "hard_missing"],
        "eval_real_hard_parallel": [s for s in final if s["scenario_family"] == "hard_parallel"],
    }
    for name, rows in out_sets.items():
        for s in rows:
            s["split"] = name

    OUT.mkdir(parents=True, exist_ok=True)

    def w(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for name, rows in out_sets.items():
        rng.shuffle(rows)
        w(OUT / f"{name}.jsonl", rows)
        print(f"  {name}: {len(rows)}", flush=True)

    report = {"model": MODEL, "hard_scale": HARD_SCALE, "total": len(final),
              "counts": {k: len(v) for k, v in out_sets.items()},
              "by_scenario": dict(Counter(s["scenario"] for s in final))}
    (OUT / "hard_eval_generation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("=== HARD EVAL DONE ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

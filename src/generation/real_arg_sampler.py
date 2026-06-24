"""Deterministic argument constructor for real KPI tools (ToolACE-adapted).

Construct-then-paraphrase: instead of letting the LLM invent arguments (the source
of hallucinated codes), we CONSTRUCT valid gold arguments from the closed
catalogues across complexity tiers, then a User Agent writes a Vietnamese query
that matches them. Gold is correct by construction.

sample() returns (arguments, hints):
  - arguments: the gold call arguments (all valid per schema/catalogue)
  - hints: param -> human-readable Vietnamese phrase, so the query writer can
    mention exactly this information (e.g. location_code "HNI" -> "Hà Nội").
"""

from __future__ import annotations

import random
from typing import Any

TIERS = ("simple", "medium", "complex", "rare")
_TIER_ID = {"simple": 1, "medium": 2, "complex": 3, "rare": 4}

_DATA_LEVEL_VN = {"day": "ngày", "week": "tuần", "month": "tháng", "quarter": "quý", "year": "năm"}

# (from_date, to_date, phrase) period templates per tier — expanded for diversity.
_PERIODS = {
    "simple": [
        ("2026-06-01", "2026-06-30", "tháng 6/2026"),
        ("2026-01-01", "2026-12-31", "năm 2026"),
        ("2025-12-01", "2025-12-31", "tháng 12/2025"),
        ("2026-03-01", "2026-03-31", "tháng 3/2026"),
        ("2025-01-01", "2025-12-31", "năm 2025"),
        ("2026-01-01", "2026-03-31", "quý 1/2026"),
    ],
    "medium": [
        ("2024-05-01", "2024-05-31", "tháng 5/2024"),
        ("2024-09-01", "2024-09-30", "tháng 9/2024"),
        ("2023-10-01", "2023-10-31", "tháng 10/2023"),
        ("2024-02-01", "2024-02-29", "tháng 2/2024"),
        ("2025-03-01", "2025-03-31", "tháng 3/2025"),
        ("2024-11-01", "2024-11-30", "tháng 11/2024"),
        ("2023-07-01", "2023-07-31", "tháng 7/2023"),
        ("2025-08-01", "2025-08-31", "tháng 8/2025"),
        ("2024-01-01", "2024-03-31", "quý 1/2024"),
        ("2024-04-01", "2024-06-30", "quý 2/2024"),
        ("2024-07-01", "2024-09-30", "quý 3/2024"),
        ("2024-10-01", "2024-12-31", "quý 4/2024"),
        ("2025-01-01", "2025-03-31", "quý 1/2025"),
        ("2023-04-01", "2023-06-30", "quý 2/2023"),
        ("2025-04-01", "2025-06-30", "quý 2/2025"),
        ("2023-06-01", "2023-06-30", "tháng 6/2023"),
    ],
    "complex": [
        ("2022-01-01", "2022-12-31", "năm 2022"),
        ("2023-01-01", "2023-12-31", "năm 2023"),
        ("2024-01-01", "2024-12-31", "năm 2024"),
        ("2021-01-01", "2021-12-31", "năm 2021"),
        ("2024-07-15", "2024-08-20", "từ 15/7 đến 20/8/2024"),
        ("2023-01-01", "2023-06-30", "6 tháng đầu năm 2023"),
        ("2024-07-01", "2024-12-31", "6 tháng cuối năm 2024"),
        ("2022-03-10", "2022-05-25", "từ 10/3 đến 25/5/2022"),
        ("2025-02-01", "2025-04-30", "từ tháng 2 đến tháng 4/2025"),
        ("2023-09-15", "2023-11-30", "từ 15/9 đến 30/11/2023"),
        ("2022-06-01", "2022-06-30", "tháng 6/2022"),
        ("2021-07-01", "2021-09-30", "quý 3/2021"),
        ("2024-12-01", "2025-02-28", "từ tháng 12/2024 đến 2/2025"),
        ("2023-11-01", "2023-11-30", "tháng 11/2023"),
    ],
    "rare": [
        ("2026-06-01", "2026-06-30", "tháng 6/2026"),
        ("2025-01-01", "2025-12-31", "năm 2025"),
        ("2024-01-01", "2024-12-31", "năm 2024"),
        ("2025-06-01", "2025-06-30", "tháng 6/2025"),
        ("2024-07-01", "2024-09-30", "quý 3/2024"),
        ("2023-01-01", "2023-12-31", "năm 2023"),
        ("2026-01-01", "2026-03-31", "quý 1/2026"),
    ],
}
_DATA_LEVEL_BY_TIER = {"simple": ["month", "quarter"], "medium": ["month", "week", "quarter"],
                       "complex": ["quarter", "year", "day", "week", "month"],
                       "rare": ["month", "quarter", "year"]}

# Human glosses for the most user-facing enums (fallback = raw value).
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
    def __init__(self, references: dict, stations: list[dict]) -> None:
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
            if g == "Tỉnh/Thành phố Việt Nam":
                provinces.append(loc["code"])
            elif g == "Khu vực":
                regions.append(loc["code"])
            elif g == "Quốc gia":
                countries.append(loc["code"])
        self._loc_tiers = {
            "simple": ["VNM", "HNI", "HCM", "DNG", "HPG", "CTO", "NAN", "THA", "QNH", "BNH", "KHA", "LDG"],
            "medium": [c for c in provinces if c not in ("HNI", "HCM", "DNG")],
            "complex": regions + countries + provinces[-8:],
            "rare": countries + regions,
        }
        self.kpis = references["kpi_code"]
        self.units = references["unit_code"]

    # ---- per-param value providers (return (value, hint|None)) ----

    def _location(self, tier, rng):
        pool = self._loc_tiers.get(tier) or self._loc_tiers["medium"]
        code = rng.choice(pool or ["VNM"])
        return code, self.loc_name.get(code, code)

    def _enum(self, param, spec, tier, rng):
        opts = [v for v in spec.get("enum", []) if v != ""]
        if not opts:
            return spec.get("default", ""), None
        val = rng.choice(opts)
        hint = _GLOSS.get(param, {}).get(val, val)
        return val, hint

    def _station(self, location_code, rng):
        pool = [s for s in self.stations if s["location_code"] == location_code] or self.stations
        s = rng.choice(pool)
        return s["station_code"], s["name"]

    # ---- main ----

    def sample(self, tool: dict, tier: str, idx: int) -> dict[str, Any]:
        rng = random.Random(idx * 97 + _TIER_ID[tier] * 7 + hash(tool["name"]) % 1000)
        props = tool["parameters"].get("properties", {})
        required = list(tool["parameters"].get("required", []))
        args: dict[str, Any] = {}
        hints: dict[str, str] = {}

        # Period (shared by from_date/to_date).
        period = rng.choice(_PERIODS[tier])
        level = rng.choice(_DATA_LEVEL_BY_TIER[tier])
        loc_code = None

        for p in required:
            spec = props.get(p, {})
            if p in ("location_code", "object_code"):
                code, name = self._location(tier, rng)
                loc_code = code
                args[p] = code
                hints[p] = name
            elif p == "from_date":
                args[p] = period[0]
            elif p == "to_date":
                args[p] = period[1]
                hints["_time"] = f"{period[2]}"
            elif p == "date":
                args[p] = period[0]
                hints["_time"] = period[2]
            elif p in ("data_level", "time_type"):
                lv = level if level in spec.get("enum", [level]) else (spec.get("enum") or ["month"])[0]
                args[p] = lv
                hints["_granularity"] = _DATA_LEVEL_VN.get(lv, lv)
            elif p == "station_code":
                code, name = self._station(loc_code, rng)
                args[p] = code
                hints[p] = name
            elif p == "kpi_code":
                k = rng.choice(self.kpis)
                args[p] = k["code"]
                hints[p] = k["meaning"]
            elif p == "unit_code":
                u = rng.choice(self.units)
                args[p] = u["code"]
                hints[p] = u["name"]
            elif p == "top_k":
                args[p] = {"simple": 5, "medium": 10, "complex": 20}[tier]
                hints["_topk"] = f"top {args[p]}"
            elif "enum" in spec:
                val, hint = self._enum(p, spec, tier, rng)
                args[p] = val
                if hint:
                    hints[p] = hint
            elif spec.get("type") == "integer":
                args[p] = 10
            elif "default" in spec:
                args[p] = spec["default"]
            else:
                args[p] = "VNM"
        return {"tool_name": tool["name"], "arguments": args, "hints": hints, "complexity": tier}

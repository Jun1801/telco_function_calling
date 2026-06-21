"""Mock for the only identifier-returning seen tool: regional_station_info.

Real KPI tools are read-only with no executor. For ReAct multi-step training we
need step1 (regional_station_info) to "return" stations that step2 consumes.
regional_station_info is the ONLY seen tool that yields reusable identifiers
(station codes) — the other 25 return KPI metric values, so dependency chains are
always "list stations → query a station". This mock synthesizes a realistic
station-record observation from the station catalogue (deterministic).
"""

from __future__ import annotations

import random
from typing import Any

# query_type -> (field name, value generator)
_ATTR = {
    "tilt": ("tilt_deg", lambda r: round(r.uniform(2.0, 8.0), 1)),
    "antenna_height": ("antenna_height_m", lambda r: r.choice([18, 24, 30, 36, 42])),
    "station_distance": ("distance_km", lambda r: round(r.uniform(0.5, 5.0), 1)),
    "": ("status", lambda r: r.choice(["active", "active", "maintenance"])),
}


def build_observation(step1_call: dict[str, Any], catalogue: list[dict], seed: int = 0, k: int = 3) -> dict[str, Any]:
    """Simulated regional_station_info result: enriched station records.

    Stations belong to step1's location if the catalogue has any (17 provinces);
    otherwise k arbitrary catalogue stations (deterministic). Attribute depends on
    step1's query_type.
    """
    rng = random.Random(seed)
    args = step1_call.get("arguments", {})
    loc = args.get("location_code") or args.get("object_code")
    by_loc = [s for s in catalogue if s["location_code"] == loc]
    pool = by_loc if by_loc else rng.sample(catalogue, min(k, len(catalogue)))
    pool = pool[:k]
    field, fn = _ATTR.get(args.get("query_type", ""), _ATTR[""])
    stations = [
        {"station_code": s["station_code"], "name": s["name"], field: fn(rng)}
        for s in pool
    ]
    return {"count": len(stations), "location": loc, "stations": stations}

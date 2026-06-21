"""Deterministic repair of the bug-9 date-pair defect in missing_slot data.

The buggy generator dropped the whole time phrase from the query but declared only
ONE of from_date/to_date missing (leaving the other in the checker). Since the query
already carries no concrete date range, the fix is a pure relabel — no LLM:

  missing_slots  : single date  -> {from_date, to_date}
  checker args   : drop BOTH from_date and to_date
  prediction.asked_slots -> the new missing_slots

Samples whose query still names a concrete year/date are ambiguous and are DROPPED.
Non-date missing_slot samples and every other family are left untouched.

  python3 scripts/repair_missing_slot_dates.py data data/real_data/outputs-2
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATE_TOK = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|năm 20\d\d|tháng \d{1,2}/20\d\d")
DATES = {"from_date", "to_date"}


def _needs_repair(s: dict) -> bool:
    if s.get("scenario_family") != "missing_slot":
        return False
    ms = set(s.get("missing_slots", []))
    return bool(ms & DATES) and not DATES <= ms


def _repair(s: dict) -> dict | None:
    """Return repaired sample, or None to drop (ambiguous concrete date in query)."""
    if DATE_TOK.search(s.get("instruction", "")):
        return None
    ms = (set(s.get("missing_slots", [])) - DATES) | DATES
    s["missing_slots"] = sorted(ms)
    args = s.get("checker_call", {}).get("arguments", {})
    for d in DATES:
        args.pop(d, None)
    if "prediction" in s and isinstance(s["prediction"], dict):
        s["prediction"]["asked_slots"] = sorted(ms)
    return s


def _process(rows: list[dict]) -> tuple[list[dict], int, int]:
    out, repaired, dropped = [], 0, 0
    for s in rows:
        if _needs_repair(s):
            fixed = _repair(s)
            if fixed is None:
                dropped += 1
                continue
            repaired += 1
            out.append(fixed)
        else:
            out.append(s)
    return out, repaired, dropped


def _rewrite(path: Path) -> None:
    if not path.exists():
        print(f"  (skip, missing) {path}")
        return
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    out, repaired, dropped = _process(rows)
    shadow = path.with_suffix(path.suffix + ".shadow")
    with shadow.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shadow.replace(path)
    print(f"  {path}:  {len(rows)} -> {len(out)}  (repaired {repaired}, dropped {dropped})")


def main() -> None:
    dirs = [Path(d) for d in (sys.argv[1:] or ["data"])]
    for d in dirs:
        print(f"== {d} ==")
        _rewrite(d / "eval_real_missing_slot.jsonl")
        _rewrite(d / "sft_train_real.jsonl")


if __name__ == "__main__":
    main()

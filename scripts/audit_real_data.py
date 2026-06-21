"""Independent cleanliness audit for the real-tool dataset.

Does NOT reuse the generator's own DualLayerRealVerifier — it re-derives every
invariant from scratch so it can catch bugs the verifier itself might miss. Scans a
directory of *.jsonl (eval_real_*.jsonl) plus an optional train file and reports, per
file and per check, FAIL counts with examples.

  python3 scripts/audit_real_data.py data            # working dir
  python3 scripts/audit_real_data.py data/real_data/outputs-2   # Kaggle canonical
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation.schema_validator import SchemaValidator

DATA = ROOT / "data"
PLACEHOLDER = "<from_step_1>"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WORD_RE = re.compile(r"\w+", re.UNICODE)

TOOLS = json.loads((DATA / "real_tools.json").read_text(encoding="utf-8"))
TBN = {t["name"]: t for t in TOOLS}
REFS = json.loads((DATA / "real_reference_codes.json").read_text(encoding="utf-8"))
STATIONS = json.loads((DATA / "real_station_catalogue.json").read_text(encoding="utf-8"))
VALID_LOC = {i["code"] for i in REFS["location_code"]}
VALID_KPI = {k["code"] for k in REFS["kpi_code"]}
VALID_UNIT = {u["code"] for u in REFS["unit_code"]}
VALID_STA = {s["station_code"] for s in STATIONS}
NAME2PAT = [(re.compile(rf"\b{re.escape(i['name'].lower().strip())}\b"), i["code"])
            for i in REFS["location_code"] if len(i["name"].strip()) >= 4]
VALIDATOR = SchemaValidator()


def _tool_for(sample, name):
    mt = sample.get("masked_tool")
    if mt and mt.get("name") == name:
        return mt
    return TBN.get(name)


def _all_calls(sample):
    if sample.get("gold_call"):
        return [sample["gold_call"]]
    return sample.get("gold_calls") or sample.get("gold_steps") or []


def _subst_placeholder(tool, args):
    if not any(PLACEHOLDER in str(v) for v in args.values()):
        return args
    props = (tool or {}).get("parameters", {}).get("properties", {})
    out = {}
    for k, v in args.items():
        if PLACEHOLDER in str(v):
            sp = props.get(k, {})
            out[k] = sp["enum"][0] if sp.get("enum") else ("VNM" if sp.get("type") == "string" else 1)
        else:
            out[k] = v
    return out


def audit_sample(s, issues):
    """Append (check_name, sample_id) to issues[check] for each failure."""
    def fail(check):
        issues[check].append(s.get("id", "?"))

    fam = s.get("scenario_family", "")
    act = s.get("expected_action", "")
    instr = s.get("instruction", "")

    # 1. instruction length
    if len(instr.strip()) < 8:
        fail("short_instruction")

    # 2. structure ↔ action
    if act == "call_function" and not s.get("gold_call"):
        fail("missing_gold_call")
    if act == "call_functions" and not (s.get("gold_calls") or s.get("gold_steps")):
        fail("missing_gold_calls")
    if act == "ask_clarification" and (not s.get("missing_slots") or not s.get("checker_call")):
        fail("missing_clar_fields")
    if act == "abstain" and not (s.get("prediction") or s.get("reason")):
        fail("missing_abstain_reason")

    # 3-7: call-bearing samples
    calls = _all_calls(s)
    for c in calls:
        tool = _tool_for(s, c["tool_name"])
        if tool is None:
            fail("unknown_tool"); continue
        a = c.get("arguments", {})

        # 3. schema validity (placeholder substituted)
        issues_found = VALIDATOR.validate_call(tool, _subst_placeholder(tool, a), c["tool_name"])
        # a deprecated-only issue is not a cleanliness fail
        bad = [i for i in (issues_found or []) if getattr(i, "code", "") != "deprecated_tool"]
        if bad:
            fail("schema_invalid_gold")

        # 4. code catalogues
        oc = a.get("object_code")
        if oc is not None and PLACEHOLDER not in str(oc) and oc not in VALID_LOC and oc not in VALID_STA:
            fail("object_code_off_catalogue")
        for key, pool in (("location_code", VALID_LOC), ("kpi_code", VALID_KPI),
                          ("unit_code", VALID_UNIT), ("station_code", VALID_STA)):
            v = a.get(key)
            if v is not None and PLACEHOLDER not in str(v) and v not in pool:
                fail(f"{key}_off_catalogue")

        # 5. dates
        for k in ("from_date", "to_date", "date"):
            v = a.get(k)
            if v is not None and PLACEHOLDER not in str(v) and not DATE_RE.match(str(v)):
                fail("bad_date_format")
        f, t = a.get("from_date"), a.get("to_date")
        if f and t and DATE_RE.match(str(f)) and DATE_RE.match(str(t)) and f > t:
            fail("from_after_to")

    # 6. city ↔ code
    matched = {code for pat, code in NAME2PAT if pat.search(instr.lower())}
    gold_locs = {c["arguments"].get("location_code") for c in calls
                 if c.get("arguments", {}).get("location_code")}
    if matched and gold_locs and not all(g in matched for g in gold_locs):
        fail("city_code_mismatch")

    # 7. placeholder leak: only allowed inside raw call_functions gold_steps step2.
    raw_multi = act == "call_functions" and s.get("gold_steps")
    blob_calls = json.dumps(calls, ensure_ascii=False)
    if PLACEHOLDER in blob_calls and not raw_multi:
        fail("placeholder_leak")

    # 8. masking
    if fam == "masking" or s.get("masked_tool"):
        mt = s.get("masked_tool") or {}
        gold_name = (s.get("gold_call") or {}).get("tool_name", "")
        if mt.get("name") != gold_name:
            fail("masked_name_mismatch")
        if not mt.get("parameters"):
            fail("masked_no_schema")
        if mt.get("name") and mt["name"] not in instr:
            fail("masked_name_not_in_instruction")

    # 9. missing_slot coherence (bug-9 date pair)
    if act == "ask_clarification":
        ms = set(s.get("missing_slots", []))
        checker = s.get("checker_call") or {}
        tool = _tool_for(s, checker.get("tool_name", ""))
        req = set((tool or {}).get("parameters", {}).get("required", []))
        if not ms <= req:
            fail("missing_slot_not_required")
        cargs = set((checker.get("arguments") or {}).keys())
        if ms & cargs:
            fail("missing_slot_present_in_checker")
        if ms & {"from_date", "to_date"} and not {"from_date", "to_date"} <= ms:
            fail("date_pair_incomplete")
        # checker must be schema-INVALID (a required slot is missing)
        if tool and not VALIDATOR.validate_call(tool, checker.get("arguments", {}), checker.get("tool_name", "")):
            fail("clarification_checker_actually_valid")


def _norm(t):
    return frozenset(WORD_RE.findall(t.lower()))


def audit_file(path: Path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    issues = defaultdict(list)
    seen = defaultdict(list)  # (family, primary_tool) -> [normalized instruction]
    dup = 0
    for s in rows:
        if s.get("source") != "real_tool_xlsx":
            continue
        audit_sample(s, issues)
        calls = _all_calls(s)
        pt = calls[0]["tool_name"] if calls else (s.get("checker_call") or {}).get("tool_name", "")
        key = (s.get("scenario_family"), pt)
        norm = _norm(s.get("instruction", ""))
        if norm in seen[key]:
            dup += 1
        else:
            seen[key].append(norm)
    if dup:
        issues["exact_duplicate_instruction"] = [f"{dup} dupes"]
    return len(rows), issues


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA
    files = sorted(target.glob("eval_real_*.jsonl"))
    train = target / "sft_train_real.jsonl"
    if train.exists():
        files.append(train)
    print(f"Auditing {target}  ({len(files)} files)\n" + "=" * 60)
    grand = Counter()
    for f in files:
        n, issues = audit_file(f)
        if not issues:
            print(f"✅ {f.name:32s} {n:5d} samples  CLEAN")
        else:
            print(f"❌ {f.name:32s} {n:5d} samples")
            for check, ids in sorted(issues.items()):
                cnt = len(ids) if not (ids and "dupes" in str(ids[0])) else ids[0]
                print(f"     {check:38s} {cnt}   e.g. {ids[:2]}")
                grand[check] += (len(ids) if "dupes" not in str(ids[0]) else 0)
    print("=" * 60)
    print("TOTAL FAILS:", dict(grand) if grand else "NONE — dataset CLEAN")


if __name__ == "__main__":
    main()

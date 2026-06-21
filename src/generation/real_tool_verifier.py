"""Dual-Layer Verification (ToolACE DLV) for construct-then-paraphrase real data.

Rule layer (deterministic):
  - SchemaValidator: enum/required/type (gold built valid → mainly guards regressions).
  - city↔code: if the query names a catalogue location, gold location_code must match it.
  - object_code ∈ location ∪ station catalogue.
  - date format YYYY-MM-DD and from_date ≤ to_date.
Model layer (LLM, decomposed):
  - query↔args consistency: does the query contain exactly the gold facts?
  - abstain: is the query genuinely out of KPI scope?
Plus dedupe by (family, tool) Jaccard.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.validation.schema_validator import SchemaValidator

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PLACEHOLDER = "<from_step_1>"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _norm_tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class DualLayerRealVerifier:
    def __init__(self, tools_by_name, references=None, stations=None, generator=None,
                 dedupe_threshold=0.85, run_semantic=True) -> None:
        self.tools = tools_by_name
        self.validator = SchemaValidator()
        self.generator = generator
        self.dedupe_threshold = dedupe_threshold
        self.run_semantic = run_semantic and generator is not None
        self.references = references or {}
        self.valid_loc = {i["code"] for i in self.references.get("location_code", [])}
        self.valid_station = {s["station_code"] for s in (stations or [])}
        # name(lower) -> code, only names length>=4 to avoid spurious substring hits
        self.name2code = {}
        for i in self.references.get("location_code", []):
            nm = i["name"].lower().strip()
            if len(nm) >= 4:
                self.name2code.setdefault(nm, i["code"])
        # Word-boundary patterns: plain substring match false-positives (a short
        # name embedded in a longer word) → spurious wrong-city drops.
        self.name2pat = [(re.compile(rf"\b{re.escape(nm)}\b"), code)
                         for nm, code in self.name2code.items()]
        self.stats = {"input": 0, "schema_fail": 0, "city_fail": 0, "date_fail": 0,
                      "semantic_fail": 0, "dedup": 0, "kept": 0}

    # ---- helpers ----
    def _tool_for(self, sample, name):
        mt = sample.get("masked_tool")
        if mt and mt["name"] == name:
            return mt
        return self.tools.get(name)

    @staticmethod
    def _all_calls(sample):
        if sample.get("gold_call"):
            return [sample["gold_call"]]
        return sample.get("gold_calls") or sample.get("gold_steps") or []

    # ---- rule layer ----
    def _schema_ok(self, sample) -> bool:
        action = sample["expected_action"]
        if action == "ask_clarification":
            checker = sample.get("checker_call")
            if not checker:
                return False
            tool = self._tool_for(sample, checker["tool_name"])
            return bool(self.validator.validate_call(tool, checker["arguments"], checker["tool_name"]))
        calls = self._all_calls(sample)
        if not calls:
            return action == "abstain"
        for c in calls:
            tool = self._tool_for(sample, c["tool_name"])
            # multi_step step2 carries a <from_step_1> placeholder (resolved at runtime);
            # substitute a schema-valid stand-in so the OTHER args are still checked.
            args = self._subst_placeholder(tool, c["arguments"])
            if self.validator.validate_call(tool, args, c["tool_name"]):
                return False
        return True

    @staticmethod
    def _subst_placeholder(tool, args):
        if not any(_PLACEHOLDER in str(v) for v in args.values()):
            return args
        props = (tool or {}).get("parameters", {}).get("properties", {})
        out = {}
        for k, v in args.items():
            if _PLACEHOLDER in str(v):
                sp = props.get(k, {})
                out[k] = sp["enum"][0] if sp.get("enum") else ("VNM" if sp.get("type") == "string" else 1)
            else:
                out[k] = v
        return out

    def _codes_ok(self, sample) -> bool:
        for c in self._all_calls(sample):
            a = c.get("arguments", {})
            oc = a.get("object_code")
            if oc is not None and _PLACEHOLDER not in str(oc):
                if oc not in self.valid_loc and oc not in self.valid_station:
                    return False
            sc = a.get("station_code")
            if sc is not None and _PLACEHOLDER not in str(sc) and self.valid_station:
                if sc not in self.valid_station:
                    return False
        return True

    def _dates_ok(self, sample) -> bool:
        for c in self._all_calls(sample):
            a = c.get("arguments", {})
            for k in ("from_date", "to_date", "date"):
                v = a.get(k)
                if v is not None and _PLACEHOLDER not in str(v) and not _DATE_RE.match(str(v)):
                    return False
            f, t = a.get("from_date"), a.get("to_date")
            if f and t and _DATE_RE.match(str(f)) and _DATE_RE.match(str(t)) and f > t:
                return False
        return True

    def _city_code_ok(self, sample) -> bool:
        """If the query names catalogue locations, gold location_code must be among them."""
        q = sample["instruction"].lower()
        matched = {code for pat, code in self.name2pat if pat.search(q)}
        if not matched:
            return True
        gold_locs = {c["arguments"].get("location_code") for c in self._all_calls(sample)
                     if c.get("arguments", {}).get("location_code")}
        if not gold_locs:
            return True
        # every gold location mentioned must be a matched one (no wrong-city gold)
        return all(g in matched for g in gold_locs)

    # ---- model layer (batched) ----
    @staticmethod
    def _needs_semantic(sample) -> bool:
        # masking derived; multi placeholder; missing proven by rule → skip.
        return sample["scenario_family"] not in ("masking", "multi_step", "missing_slot")

    def _semantic_item(self, sample) -> str:
        if sample["expected_action"] == "abstain":
            return (f'Câu hỏi: "{sample["instruction"]}"\n   Hỏi: câu này NẰM NGOÀI phạm vi tra cứu '
                    "KPI/chỉ số/hạ tầng mạng viễn thông?")
        calls = self._all_calls(sample)
        facts = "; ".join(f"{c['tool_name']}({c['arguments']})" for c in calls)
        return (f'Câu hỏi: "{sample["instruction"]}"\n   Cần có: {facts}\n   '
                "Hỏi: câu hỏi chứa ĐỦ và ĐÚNG thông tin để suy ra lệnh gọi này?")

    def _semantic_batch(self, samples: list[dict], size: int = 10) -> list[bool]:
        from src.generation.real_tool_llm_generator import _extract_json_array
        verdicts: list[bool] = []
        sysmsg = ("Bạn là giám khảo dữ liệu. Với mỗi mục, trả lời CO hoặc KHONG. "
                  'Trả về JSON array các chuỗi "CO"/"KHONG" đúng thứ tự, đúng số lượng.')
        for start in range(0, len(samples), size):
            chunk = samples[start:start + size]
            listing = "\n".join(f"{i+1}. {self._semantic_item(s)}" for i, s in enumerate(chunk))
            ans = self.generator.chat(sysmsg, f"{len(chunk)} mục:\n{listing}\n\nJSON array:",
                                      temperature=0.0, max_tokens=15 * len(chunk) + 60)
            arr = _extract_json_array(ans)
            for i in range(len(chunk)):
                v = str(arr[i]).upper() if i < len(arr) else "CO"  # default keep on parse gap
                verdicts.append("CO" in v and "KHONG" not in v)
        return verdicts

    # ---- orchestration ----
    def run(self, candidates: list[dict]) -> list[dict]:
        self.stats["input"] = len(candidates)
        stage = []
        for s in candidates:
            if len(s.get("instruction", "").strip()) < 8:
                self.stats["schema_fail"] += 1; continue
            if not self._schema_ok(s):
                self.stats["schema_fail"] += 1; continue
            if not self._codes_ok(s):
                self.stats["schema_fail"] += 1; continue
            if not self._dates_ok(s):
                self.stats["date_fail"] += 1; continue
            if not self._city_code_ok(s):
                self.stats["city_fail"] += 1; continue
            stage.append(s)
        if not self.run_semantic:
            sem = stage
        else:
            need = [s for s in stage if self._needs_semantic(s)]
            skip = [s for s in stage if not self._needs_semantic(s)]
            verdicts = self._semantic_batch(need)
            sem = skip + [s for s, ok in zip(need, verdicts) if ok]
            self.stats["semantic_fail"] += sum(1 for ok in verdicts if not ok)
        kept = self._dedupe(sem)
        self.stats["dedup"] = len(sem) - len(kept)
        self.stats["kept"] = len(kept)
        return kept

    @staticmethod
    def _primary_tool(s):
        if s.get("gold_call"):
            return s["gold_call"].get("tool_name", "")
        for k in ("gold_calls", "gold_steps"):
            if s.get(k):
                return s[k][0].get("tool_name", "")
        if s.get("checker_call"):
            return s["checker_call"].get("tool_name", "")
        return ""

    @staticmethod
    def _arg_sig(s):
        calls = (s.get("gold_calls") or s.get("gold_steps")
                 or ([s["gold_call"]] if s.get("gold_call") else [])
                 or ([s["checker_call"]] if s.get("checker_call") else []))
        return json.dumps([[c.get("tool_name"), sorted(c.get("arguments", {}).items())] for c in calls],
                          sort_keys=True, default=str)

    _MAX_PER_ARGS = 3  # allow up to N phrasings per identical gold-arg combo (helps low-param tools)

    def _dedupe(self, samples):
        import collections
        kept, kept_tok = [], []
        arg_count = collections.Counter()  # cap exact gold-arg repeats within a (family, tool)
        for s in samples:
            key = s["scenario_family"] + "|" + self._primary_tool(s)
            sig = key + "|" + self._arg_sig(s)
            if s["scenario_family"] != "abstain" and arg_count[sig] >= self._MAX_PER_ARGS:
                continue
            toks = _norm_tokens(s["instruction"])
            if any(k == key and _jaccard(toks, t) > self.dedupe_threshold for k, t in kept_tok):
                continue
            kept.append(s); kept_tok.append((key, toks)); arg_count[sig] += 1
        return kept

"""ToolACE-adapted generator for real KPI tools (construct-then-paraphrase).

Gold arguments are CONSTRUCTED by ArgSampler (always valid). The LLM acts only as
the ToolACE "User Agent": it writes a natural Vietnamese question that contains
exactly the constructed facts. This removes the hallucinated-code failure mode
(gold can't be wrong) while keeping ToolACE structure (multi-agent + complexity
tiers + dual-layer verification done separately).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from src.generation.real_arg_sampler import ArgSampler, TIERS

STEP1_REF = "<from_step_1>"


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_json_array(text: str) -> list[Any]:
    text = _strip_thinking(text)
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    cand = m.group(1) if m else text[text.find("["): text.rfind("]") + 1]
    try:
        data = json.loads(cand)
    except (json.JSONDecodeError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _coerce_query(x: Any) -> str:
    """The model may return strings or {"question"/"query": ...} objects."""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("question", "query", "cau_hoi", "q", "text"):
            if isinstance(x.get(k), str):
                return x[k].strip()
        for v in x.values():
            if isinstance(v, str):
                return v.strip()
    return ""


# Tier mix for single-step generation.
_TIER_MIX = ["simple", "simple", "simple", "medium", "medium", "medium", "medium", "complex", "complex", "complex"]


class RealToolLLMGenerator:
    def __init__(self, model_name="Qwen/Qwen3-4B", references=None, stations=None,
                 adapter_path=None, temperature=0.8, max_tokens=2048) -> None:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load
        self._mlx_generate = mlx_generate
        self.model, self.tokenizer = load(model_name, adapter_path=adapter_path)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.references = references or {}
        self.stations = stations or []
        self.sampler = ArgSampler(self.references, self.stations)

    # ---- LLM plumbing ----
    def chat(self, system: str, user: str, temperature: float | None = None, max_tokens: int | None = None) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        kwargs: dict[str, Any] = {"max_tokens": max_tokens or self.max_tokens, "verbose": False}
        temp = self.temperature if temperature is None else temperature
        if temp > 0:
            from mlx_lm.sample_utils import make_sampler
            kwargs["sampler"] = make_sampler(temp=temp)
        result = self._mlx_generate(self.model, self.tokenizer, prompt=text, **kwargs)
        # MLX retains buffers across calls; over hundreds of generations this grows
        # until the process thrashes/stalls. Release the cache each call.
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        return result

    # ---- fact spec from hints (what the query must contain) ----
    @staticmethod
    def _spec(tool: dict, hints: dict) -> str:
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
        return f"Nội dung: {desc}. Phải nêu: " + "; ".join(parts)

    def _paraphrase_batch(self, specs: list[str], temperature=0.8) -> list[str]:
        """Write one natural Vietnamese query per spec (batched)."""
        if not specs:
            return []
        system = ("Bạn đóng vai người dùng hỏi trợ lý phân tích mạng viễn thông Viettel. "
                  "Viết câu hỏi tiếng Việt TỰ NHIÊN, đa dạng cách diễn đạt. "
                  "Mỗi câu phải chứa ĐÚNG và ĐỦ các thông tin được nêu, KHÔNG thêm thông tin khác.")
        listing = "\n".join(f"{i+1}. {s}" for i, s in enumerate(specs))
        user = (f"Viết {len(specs)} câu hỏi, mỗi câu cho 1 mục dưới đây:\n{listing}\n\n"
                'Trả về JSON array gồm đúng số câu, theo thứ tự. Chỉ JSON.')
        # ~50 tokens/query + JSON overhead; cap to avoid runaway generation.
        arr = _extract_json_array(self.chat(system, user, temperature, max_tokens=60 * len(specs) + 100))
        return [c for c in (_coerce_query(x) for x in arr) if c]

    def _batch_build(self, specs, builders, temperature=0.8, size=15):
        """Paraphrase specs in batches; build a sample per (query, builder)."""
        out = []
        for start in range(0, len(specs), size):
            qs = self._paraphrase_batch(specs[start:start + size], temperature)
            for j, q in enumerate(qs):
                k = start + j
                if k >= len(builders):
                    break
                s = builders[k](q)
                if s:
                    out.append(s)
        return out

    @staticmethod
    def _sample(idx, tool, args, query, scenario, family, split, extra=None) -> dict:
        s = {
            "id": f"real_{split}_{tool['name']}_{family}_{idx:04d}",
            "source": "real_tool_xlsx", "split": split, "scenario": scenario,
            "scenario_family": family, "instruction": query,
            "expected_action": "call_function", "generator": "real_tool_llm",
            "gold_call": {"tool_name": tool["name"], "arguments": args},
        }
        if extra:
            s.update(extra)
        return s

    # ---- Case: single_step_valid ----
    def gen_single_step(self, tool: dict, n: int) -> list[dict]:
        sampled = [self.sampler.sample(tool, _TIER_MIX[i % len(_TIER_MIX)], i) for i in range(n)]
        specs = [self._spec(tool, s["hints"]) for s in sampled]
        out = []
        for batch_start in range(0, len(specs), 15):
            chunk = specs[batch_start:batch_start + 15]
            queries = self._paraphrase_batch(chunk)
            for j, q in enumerate(queries):
                k = batch_start + j
                if k >= len(sampled):
                    break
                out.append(self._sample(k, tool, sampled[k]["arguments"], q,
                                        "valid_kpi_read", "single_step_valid", tool["split"]))
        return out

    # ---- Case: missing_slot ----
    def gen_missing_slot(self, tool: dict, n: int) -> list[dict]:
        required = tool["parameters"].get("required", [])
        # from_date/to_date share a single time phrase (hint "_time"); they must be
        # dropped together, else the query loses the whole period but only one date
        # is declared missing → the model rightly asks for both and is scored wrong.
        date_slots = [d for d in ("from_date", "to_date") if d in required]
        droppable: list[str] = (["_dates"] if date_slots else []) + [
            s for s in ("location_code", "kpi_code", "station_code") if s in required
        ]
        if not droppable:
            return []
        specs, builders = [], []
        for i in range(n):
            s = self.sampler.sample(tool, _TIER_MIX[i % len(_TIER_MIX)], i + 5000)
            slot = droppable[i % len(droppable)]
            hints = dict(s["hints"])
            if slot == "_dates":
                drop, label = date_slots, "khoảng thời gian"
                hints.pop("_time", None)
            else:
                drop, label = [slot], slot
                hints.pop(slot, None)
            specs.append(self._spec(tool, hints) + f"  (CỐ TÌNH KHÔNG nêu {label})")
            checker = {k: v for k, v in s["arguments"].items() if k not in drop}

            def build(q, drop=drop, checker=checker, i=i):
                return {
                    "id": f"real_{tool['split']}_{tool['name']}_missing_{i:04d}",
                    "source": "real_tool_xlsx", "split": tool["split"], "scenario": "missing_parameter",
                    "scenario_family": "missing_slot", "instruction": q,
                    "expected_action": "ask_clarification", "generator": "real_tool_llm",
                    "missing_slots": drop, "prediction": {"action": "ask_clarification", "asked_slots": drop},
                    "checker_call": {"tool_name": tool["name"], "arguments": checker},
                    "checker_expected_status": "schema_invalid",
                }
            builders.append(build)
        return self._batch_build(specs, builders)

    # ---- Case: masking (fn-name / fn+param / renamed) ----
    def gen_masking(self, single_samples: list[dict], tools_by_name: dict, n: int) -> list[dict]:
        out = []
        for i, base in enumerate(single_samples[:n]):
            tool = tools_by_name.get(base["gold_call"]["tool_name"])
            if tool is None:
                continue
            mode = ("fn", "fn", "param", "renamed")[i % 4]
            masked_name = f"func_{i+1}"
            props = tool["parameters"].get("properties", {})
            base_args = base["gold_call"]["arguments"]
            if mode == "param":
                keymap = {k: f"param_{j+1}" for j, k in enumerate(props)}
                masked_props = {keymap[k]: copy.deepcopy(v) for k, v in props.items()}
                masked_params = {"type": "object",
                                 "required": [keymap[k] for k in tool["parameters"].get("required", []) if k in keymap],
                                 "properties": masked_props}
                gold_args = {keymap[k]: v for k, v in base_args.items() if k in keymap}
                masked_tool = {"name": masked_name, "description": tool["description"],
                               "parameters": masked_params, "status": "active", "deprecated": False}
                instr = f"Dùng {masked_name} để: {base['instruction']}"
            elif mode == "renamed":
                masked_name = f"kpi_query_{i+1}"
                masked_tool = {"name": masked_name, "description": "Hàm tra cứu: " + tool["description"],
                               "parameters": copy.deepcopy(tool["parameters"]), "status": "active", "deprecated": False}
                gold_args = copy.deepcopy(base_args)
                instr = f"Dùng hàm {masked_name} để: {base['instruction']}"
            else:  # fn-name only
                masked_tool = {"name": masked_name, "description": tool["description"],
                               "parameters": copy.deepcopy(tool["parameters"]), "status": "active", "deprecated": False}
                gold_args = copy.deepcopy(base_args)
                instr = f"Dùng {masked_name} để: {base['instruction']}"
            out.append({
                "id": f"real_mask_{mode}_{i:04d}", "source": "real_tool_xlsx", "split": base["split"],
                "scenario": f"masking_{mode}", "scenario_family": "masking", "instruction": instr,
                "expected_action": "call_function", "generator": "real_tool_llm",
                "gold_call": {"tool_name": masked_name, "arguments": gold_args}, "masked_tool": masked_tool,
            })
        return out

    # ---- Case: parallel ----
    def gen_parallel(self, pairs: list[tuple[dict, dict]], per_pair: int) -> list[dict]:
        specs, builders = [], []
        for pi, (a, b) in enumerate(pairs):
            for i in range(per_pair):
                tier = _TIER_MIX[i % len(_TIER_MIX)]
                sa = self.sampler.sample(a, tier, i + 1000)
                shared = {k: sa["arguments"][k] for k in ("location_code", "from_date", "to_date", "data_level") if k in sa["arguments"]}
                sb = self.sampler.sample(b, tier, i + 2000)
                for k, v in shared.items():
                    if k in sb["arguments"]:
                        sb["arguments"][k] = v
                hints = {k: v for k, v in sa["hints"].items() if k.startswith("_") or k in ("location_code", "object_code")}
                specs.append(f"Hỏi CẢ HAI trong một câu: (1) {a['description'].split('.')[0][:60]}; "
                             f"(2) {b['description'].split('.')[0][:60]}. "
                             + self._spec(a, hints).split("Phải nêu:")[-1])

                def build(q, a=a, b=b, sa=sa, sb=sb, pi=pi, i=i):
                    return {
                        "id": f"real_parallel_{pi}_{i:04d}", "source": "real_tool_xlsx", "split": "seen",
                        "scenario": "parallel_reads", "scenario_family": "parallel", "instruction": q,
                        "expected_action": "call_functions", "generator": "real_tool_llm",
                        "gold_calls": [{"tool_name": a["name"], "arguments": sa["arguments"]},
                                       {"tool_name": b["name"], "arguments": sb["arguments"]}],
                    }
                builders.append(build)
        return self._batch_build(specs, builders)

    # ---- Case: multi_step (dependent) ----
    def gen_multi_step(self, chains: list[tuple[dict, dict, str]], per_chain: int) -> list[dict]:
        specs, builders = [], []
        for ci, (src, dep, dep_key) in enumerate(chains):
            for i in range(per_chain):
                tier = _TIER_MIX[i % len(_TIER_MIX)]
                s1 = self.sampler.sample(src, tier, i + 3000)
                loc_hint = s1["hints"].get("location_code") or s1["hints"].get("object_code") or "Hà Nội"
                tm = s1["hints"].get("_time", "")
                specs.append(f"Quy trình 2 bước: trước tiên tìm các trạm ở {loc_hint}"
                             + (f" trong {tm}" if tm else "")
                             + f", sau đó tra cứu {dep['description'].split('.')[0][:60]} của các trạm đó.")
                s2args = self.sampler.sample(dep, tier, i + 4000)["arguments"]
                s2args[dep_key] = STEP1_REF

                def build(q, src=src, dep=dep, s1=s1, s2args=s2args, ci=ci, i=i):
                    return {
                        "id": f"real_multistep_{ci}_{i:04d}", "source": "real_tool_xlsx", "split": "seen",
                        "scenario": "dependency", "scenario_family": "multi_step", "instruction": q,
                        "expected_action": "call_functions", "generator": "real_tool_llm",
                        "gold_steps": [{"tool_name": src["name"], "arguments": s1["arguments"]},
                                       {"tool_name": dep["name"], "arguments": s2args, "depends_on_previous": True}],
                    }
                builders.append(build)
        return self._batch_build(specs, builders, temperature=0.7)

    # ---- Case: abstain / irrelevance (non-tool-use) ----
    def gen_abstain(self, n: int) -> list[dict]:
        system = ("Bạn tạo dữ liệu. Sinh câu hỏi tiếng Việt NGOÀI phạm vi của một hệ thống "
                  "chỉ tra cứu KPI/chỉ số mạng viễn thông (vd: đăng ký gói cước, khiếu nại hoá đơn, "
                  "đổi SIM, hỏi thời tiết, nấu ăn...). Trả JSON array các câu hỏi. Chỉ JSON.")
        out = []
        for batch in range((n + 14) // 15):
            arr = _extract_json_array(self.chat(system, f"Sinh 15 câu hỏi ngoài phạm vi, đa dạng. Lần {batch+1}.", 0.9, max_tokens=900))
            for j, q in enumerate(arr):
                if not isinstance(q, str) or not q.strip():
                    continue
                out.append({
                    "id": f"real_abstain_{batch}_{j:03d}", "source": "real_tool_xlsx", "split": "seen",
                    "scenario": "irrelevance", "scenario_family": "abstain", "instruction": q.strip(),
                    "expected_action": "abstain", "generator": "real_tool_llm",
                    "prediction": {"action": "abstain", "reason": "ngoài phạm vi công cụ KPI"},
                })
                if len(out) >= n:
                    return out
        return out

    # ---- Case: hard_abstain (telco-adjacent negatives) ----
    def gen_hard_abstain(self, n_consumer: int = 50, n_revenue: int = 50, n_coverage: int = 50, n_irrelevance: int = 50) -> list[dict]:
        """Hard abstain: telco-adjacent queries that look KPI-related but aren't supported."""
        SYS_CONSUMER = ("Bạn tạo dữ liệu huấn luyện. Sinh câu hỏi tiếng Việt theo góc nhìn nhân viên kỹ thuật viễn thông "
                        "về DỊCH VỤ KHÁCH HÀNG: đăng ký/hủy gói cước, kiểm tra hóa đơn, đổi SIM, báo mất SIM, "
                        "kích hoạt eSIM, đăng ký 4G/5G cho thuê bao. "
                        "Câu hỏi nghe có vẻ kỹ thuật nhưng là tác vụ provisioning/billing, "
                        "KHÔNG thể trả lời bằng công cụ tra cứu KPI đọc dữ liệu. "
                        "Trả JSON array các câu hỏi. Chỉ JSON.")
        SYS_REVENUE = ("Bạn tạo dữ liệu huấn luyện. Sinh câu hỏi tiếng Việt về chỉ số KINH DOANH "
                       "không có trong hệ thống KPI mạng: doanh thu, chi phí vận hành, CSAT/NPS, "
                       "số khiếu nại, chi phí bảo trì, ngân sách triển khai. "
                       "Nghe như tra cứu KPI nhưng là chỉ số kinh doanh/tài chính ngoài phạm vi. "
                       "Trả JSON array các câu hỏi. Chỉ JSON.")
        SYS_COVERAGE = ("Bạn tạo dữ liệu huấn luyện. Sinh câu hỏi về chất lượng sóng từ góc nhìn NGƯỜI DÙNG CÁ NHÂN: "
                        "sóng tại địa chỉ nhà/văn phòng, chất lượng cuộc gọi tại tòa nhà, báo mất sóng GPS. "
                        "Hệ thống chỉ có KPI tỉnh/khu vực, không tra theo địa chỉ/hộ gia đình. "
                        "Trả JSON array các câu hỏi. Chỉ JSON.")
        SYS_IRR = ("Sinh câu hỏi tiếng Việt NGOÀI phạm vi hệ thống tra cứu KPI/hạ tầng mạng viễn thông "
                   "(vd: nấu ăn, du lịch, học tập, tài chính cá nhân). Trả JSON array. Chỉ JSON.")

        def _gen_batch(system, n, scenario):
            result = []
            for batch in range((n + 14) // 15):
                if len(result) >= n:
                    break
                arr = _extract_json_array(self.chat(system, f"Sinh 15 câu hỏi đa dạng. Lần {batch+1}.", 0.9, max_tokens=900))
                for q in arr:
                    if not isinstance(q, str) or not q.strip():
                        continue
                    idx = len(result)
                    result.append({
                        "id": f"real_hard_abstain_{scenario}_{idx:04d}",
                        "source": "real_tool_xlsx", "split": "eval_real_hard_abstain",
                        "scenario": scenario, "scenario_family": "hard_abstain",
                        "instruction": q.strip(), "expected_action": "abstain",
                        "generator": "real_tool_llm",
                        "prediction": {"action": "abstain", "reason": "ngoài phạm vi công cụ KPI"},
                    })
                    if len(result) >= n:
                        break
            return result

        out = []
        out += _gen_batch(SYS_CONSUMER, n_consumer, "consumer_billing")
        out += _gen_batch(SYS_REVENUE, n_revenue, "unsupported_kpi")
        out += _gen_batch(SYS_COVERAGE, n_coverage, "consumer_coverage")
        out += _gen_batch(SYS_IRR, n_irrelevance, "irrelevance")
        return out

    # ---- Case: hard_seen_rare (rare location codes) ----
    def gen_hard_seen_rare(self, tool: dict, n: int) -> list[dict]:
        """Hard seen: rare locations (countries, regions) instead of common provinces."""
        sampled = [self.sampler.sample(tool, "rare", i) for i in range(n)]
        specs = [self._spec(tool, s["hints"]) for s in sampled]
        out = []
        for batch_start in range(0, len(specs), 15):
            chunk = specs[batch_start:batch_start + 15]
            queries = self._paraphrase_batch(chunk)
            for j, q in enumerate(queries):
                k = batch_start + j
                if k >= len(sampled):
                    break
                out.append({
                    "id": f"real_hard_seen_rare_{tool['name']}_{k:04d}",
                    "source": "real_tool_xlsx", "split": "eval_real_hard_seen",
                    "scenario": "valid_kpi_read_rare", "scenario_family": "hard_seen",
                    "instruction": q, "expected_action": "call_function",
                    "generator": "real_tool_llm",
                    "gold_call": {"tool_name": tool["name"], "arguments": sampled[k]["arguments"]},
                })
        return out

    # ---- Case: hard_missing_multi (2+ required params absent) ----
    def gen_hard_missing_multi(self, tool: dict, n: int) -> list[dict]:
        """Hard missing_slot: 2+ required params absent simultaneously."""
        required = tool["parameters"].get("required", [])
        date_slots = [d for d in ("from_date", "to_date") if d in required]
        other_slots = [s for s in ("location_code", "kpi_code", "station_code") if s in required]
        combos = []
        if date_slots and other_slots:
            for other in other_slots:
                combos.append((date_slots + [other], f"khoảng thời gian và {other}"))
        if len(other_slots) >= 2:
            combos.append((other_slots[:2], " và ".join(other_slots[:2])))
        if not combos and date_slots:
            combos.append((date_slots, "khoảng thời gian"))
        if not combos:
            return []
        specs, builders = [], []
        for i in range(n):
            s = self.sampler.sample(tool, _TIER_MIX[i % len(_TIER_MIX)], i + 7000)
            drop_slots, label = combos[i % len(combos)]
            hints = dict(s["hints"])
            if any(d in drop_slots for d in ("from_date", "to_date")):
                hints.pop("_time", None)
            for slot in drop_slots:
                if slot not in ("from_date", "to_date"):
                    hints.pop(slot, None)
            actual_drop = [sl for sl in drop_slots if sl in s["arguments"]]
            if not actual_drop:
                continue
            checker = {k: v for k, v in s["arguments"].items() if k not in drop_slots}
            specs.append(self._spec(tool, hints) + f"  (KHÔNG nêu {label})")

            def build(q, tool=tool, actual_drop=actual_drop, checker=checker, i=i):
                return {
                    "id": f"real_hard_missing_{tool['name']}_{i:04d}",
                    "source": "real_tool_xlsx", "split": "eval_real_hard_missing",
                    "scenario": "missing_multi_parameter", "scenario_family": "hard_missing",
                    "instruction": q, "expected_action": "ask_clarification",
                    "generator": "real_tool_llm",
                    "missing_slots": actual_drop,
                    "prediction": {"action": "ask_clarification", "asked_slots": actual_drop},
                    "checker_call": {"tool_name": tool["name"], "arguments": checker},
                    "checker_expected_status": "schema_invalid",
                }
            builders.append(build)
        return self._batch_build(specs, builders)

    # ---- Case: hard_parallel_implicit (no explicit 'cả hai' marker) ----
    def gen_hard_parallel_implicit(self, pairs: list[tuple[dict, dict]], per_pair: int) -> list[dict]:
        """Hard parallel: implicit multi-tool request without 'cả hai'/'đồng thời' marker."""
        specs, builders = [], []
        for pi, (a, b) in enumerate(pairs):
            for i in range(per_pair):
                tier = _TIER_MIX[i % len(_TIER_MIX)]
                sa = self.sampler.sample(a, tier, i + 8000)
                shared = {k: sa["arguments"][k] for k in ("location_code", "from_date", "to_date", "data_level") if k in sa["arguments"]}
                sb = self.sampler.sample(b, tier, i + 9000)
                for k, v in shared.items():
                    if k in sb["arguments"]:
                        sb["arguments"][k] = v
                hints = {k: v for k, v in sa["hints"].items() if k.startswith("_") or k in ("location_code", "object_code")}
                spec = (f"Chủ đề 1: {a['description'].split('.')[0][:60]}. "
                        f"Chủ đề 2: {b['description'].split('.')[0][:60]}. "
                        f"Viết câu hỏi về CẢ HAI mà KHÔNG dùng từ 'đồng thời', 'cả hai', 'cùng lúc'. "
                        + self._spec(a, hints).split("Phải nêu:")[-1])
                specs.append(spec)

                def build(q, a=a, b=b, sa=sa, sb=sb, pi=pi, i=i):
                    return {
                        "id": f"real_hard_parallel_{pi}_{i:04d}",
                        "source": "real_tool_xlsx", "split": "eval_real_hard_parallel",
                        "scenario": "implicit_parallel", "scenario_family": "hard_parallel",
                        "instruction": q, "expected_action": "call_functions",
                        "generator": "real_tool_llm",
                        "gold_calls": [{"tool_name": a["name"], "arguments": sa["arguments"]},
                                       {"tool_name": b["name"], "arguments": sb["arguments"]}],
                    }
                builders.append(build)
        return self._batch_build(specs, builders)

    # ---- Unseen eval from real expert seed examples ----
    def from_seed_examples(self, unseen_tools: list[dict]) -> list[dict]:
        out = []
        for tool in unseen_tools:
            for i, ex in enumerate(tool.get("seed_examples", [])):
                if ex.get("query") and ex.get("call"):
                    out.append(self._sample(i, tool, copy.deepcopy(ex["call"]), ex["query"].strip(),
                                            "valid_unseen_tool", "single_step_valid", "eval_real_unseen"))
        return out

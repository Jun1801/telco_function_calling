"""Regression tests for the Tier-1 src review bug fixes.

Each test pins one confirmed bug from plans/encapsulated-scribbling-forest.md so a
future refactor can't silently reintroduce it.
"""

from src.evaluation.evaluator import evaluate_prediction
from src.evaluation.real_evaluator import evaluate_real_prediction
from src.evaluation.routing import build_sample_prompt, evaluate_sample, load_real_assets
from src.executor.mock_telco_api import MockTelcoApi
from src.generation.real_arg_sampler import ArgSampler
from src.generation.real_tool_llm_generator import RealToolLLMGenerator
from src.generation.real_tool_verifier import DualLayerRealVerifier
from src.model.prompt_builder import build_prompt_messages
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry

REAL_REG = ToolRegistry.from_file("data/real_tools.json")


def _syn_inputs():
    return (
        ToolRegistry.from_file("data/tools.json"),
        ContractRegistry.from_file("data/tool_contracts.json"),
        MockTelcoApi.from_file("data/mock_telco_db.json"),
    )


# ---- Bug 1: parse_error must NOT be scored as a correct abstain ----

def test_bug1_real_parse_error_is_format_error_not_abstain() -> None:
    sample = {"id": "p1", "expected_action": "abstain"}
    pred = {"action": "abstain", "reason": "parse_error", "parse_error": "not json"}
    res = evaluate_real_prediction(sample, pred, REAL_REG)
    assert res["reward_strict"] == 0.0
    assert res["feedback"]["machine_status"] == "format_error"


def test_bug1_synthetic_parse_error_scores_zero() -> None:
    registry, contracts, state = _syn_inputs()
    sample = {"expected_action": "abstain"}
    pred = {"action": "abstain", "reason": "parse_error", "parse_error": "garbage"}
    res = evaluate_prediction(sample, pred, registry, state, contracts)
    assert res["reward_strict"] == 0.0
    assert res["feedback"]["machine_status"] == "format_error"


# ---- Bug 3: a deprecated tool call is schema-valid JSON ----

def test_bug3_deprecated_call_keeps_schema_validity_and_flags_rate() -> None:
    registry, contracts, state = _syn_inputs()
    sample = {
        "expected_action": "call_function",
        "gold_call": {"tool_name": "legacy_change_plan",
                      "arguments": {"customer_id": "C001", "plan": "plan_basic"}},
    }
    pred = {"action": "call_function",
            "call": {"tool_name": "legacy_change_plan",
                     "arguments": {"customer_id": "C001", "plan": "plan_basic"}}}
    res = evaluate_prediction(sample, pred, registry, state, contracts)
    assert res["metrics"]["schema_validity"] == 1.0
    assert res["metrics"]["deprecated_tool_call_rate"] == 1.0


# ---- Bug 4: an unmatched parallel call → one unnecessary_call, not per-arg spam ----

def test_bug4_unmatched_multi_call_emits_single_unnecessary_call() -> None:
    sample = {
        "id": "m1", "expected_action": "call_functions",
        "gold_calls": [{"tool_name": "vung_phu_province",
                        "arguments": {"location_code": "HNI", "tech_type": "5G"}}],
    }
    pred = {"action": "call_functions", "calls": [
        {"tool_name": "vung_phu_province", "arguments": {"location_code": "HNI", "tech_type": "5G"}},
        {"tool_name": "kqi_province",
         "arguments": {"location_code": "HCM", "from_date": "2026-06-01",
                       "to_date": "2026-06-30", "data_level": "day"}},
    ]}
    res = evaluate_real_prediction(sample, pred, REAL_REG)
    codes = [e["code"] for e in res["feedback"]["errors"]]
    assert "unnecessary_call" in codes
    assert "extra_argument" not in codes  # no per-arg spam for the unmatched call


# ---- Bug 2 + 8: masking inject + fallback gating in the prompt builder ----

def test_bug2_synthetic_masked_injects_func_and_shadows_real_tool() -> None:
    registry, contracts, _ = _syn_inputs()
    masked = {"name": "func_7", "original_name": "activate_esim",
              "description": "masked", "parameters": {"type": "object", "properties": {}}}
    sample = {"source": "telco_toolace_mini", "instruction": "func_7 should activate eSIM for C001",
              "gold_call": {"tool_name": "activate_esim", "arguments": {"customer_id": "C001"}},
              "masked_tool": masked}
    msgs = build_prompt_messages(sample, registry, contracts, extra_tools=[masked])
    body = msgs[1]["content"]
    assert "func_7" in body
    assert "activate_esim" not in body  # real shadowed name must not leak


def test_bug8_real_sample_gets_no_synthetic_fallback_tools() -> None:
    # Real = read-only → no contract registry (contract_registry=None).
    sample = {"source": "real_tool_xlsx", "instruction": "Báo cáo sim roaming data package",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    msgs = build_prompt_messages(sample, REAL_REG, None)
    body = msgs[1]["content"]
    # synthetic keyword/hardcoded fallbacks must not pollute a real-tool prompt
    assert "get_balance" not in body
    assert "activate_esim" not in body
    # read-only real prompt carries no contract/customer_verified block
    assert "tool_contracts" not in body
    assert "customer_verified" not in body


# ---- Bug 7: routing dispatches a real sample to the schema-only evaluator ----

def test_bug7_routing_evaluates_real_sample_without_crash() -> None:
    real_assets = load_real_assets("data")
    assert real_assets is not None
    registry, contracts, _ = _syn_inputs()
    sample = {"id": "rr1", "source": "real_tool_xlsx", "expected_action": "call_function",
              "instruction": "KPI Hà Nội",
              "gold_call": {"tool_name": "vung_phu_province",
                            "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    pred = {"action": "call_function",
            "call": {"tool_name": "vung_phu_province", "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    res = evaluate_sample(sample, pred, registry, contracts, "data", real_assets)
    assert res["reward_strict"] == 1.0


# ---- Bug 9: dropping a date drops the whole from/to pair ----

def _gen_no_llm():
    import json
    refs = json.load(open("data/real_reference_codes.json"))
    stations = json.load(open("data/real_station_catalogue.json"))
    gen = RealToolLLMGenerator.__new__(RealToolLLMGenerator)  # bypass mlx load
    gen.sampler = ArgSampler(refs, stations)
    gen._paraphrase_batch = lambda specs, temperature=0.8: list(specs)  # identity
    return gen


def test_bug9_missing_slot_drops_date_pair_together() -> None:
    gen = _gen_no_llm()
    tool = REAL_REG.get("tram_nha_mang_khac_province")  # required: from_date,to_date,...
    samples = gen.gen_missing_slot(tool, 6)
    date_samples = [s for s in samples if set(s["missing_slots"]) & {"from_date", "to_date"}]
    assert date_samples, "expected at least one date-drop sample"
    for s in date_samples:
        assert set(s["missing_slots"]) == {"from_date", "to_date"}
        args = s["checker_call"]["arguments"]
        assert "from_date" not in args and "to_date" not in args


# ---- Bugs 10/11/13: verifier guards ----

def _verifier():
    import json
    refs = json.load(open("data/real_reference_codes.json"))
    stations = json.load(open("data/real_station_catalogue.json"))
    tbn = {t["name"]: t for t in REAL_REG.all()}
    return DualLayerRealVerifier(tbn, references=refs, stations=stations, run_semantic=False)


def test_bug10_codes_ok_rejects_unknown_station_code() -> None:
    v = _verifier()
    bad = {"expected_action": "call_function",
           "gold_call": {"tool_name": "x", "arguments": {"station_code": "EHB99999"}}}
    assert v._codes_ok(bad) is False


def test_bug11_run_drops_too_short_instruction() -> None:
    v = _verifier()
    short = {"scenario_family": "single_step_valid", "expected_action": "abstain",
             "instruction": "ngắn"}
    kept = v.run([short])
    assert kept == []


def test_bug13_city_match_uses_word_boundary() -> None:
    v = _verifier()
    # "Hà Nội" present as a whole phrase → matched; gold HNI is consistent.
    ok = {"scenario_family": "single_step_valid", "expected_action": "call_function",
          "instruction": "báo cáo tại hà nội",
          "gold_call": {"tool_name": "vung_phu_province",
                        "arguments": {"location_code": "HNI", "tech_type": "5G"}}}
    assert v._city_code_ok(ok) is True

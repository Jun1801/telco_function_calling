"""Route a sample to the synthetic or the real-tool evaluation path.

Synthetic telco tools (data/tools.json) are transactional: subscribers,
contracts, stateful mock execution. Real Viettel KPI tools
(data/real_tools.json) are read-only: schema-only scoring, no executor, no
contracts. A sample is "real" when source == REAL_SOURCE. This helper centralises
the routing so run_baseline / run_eval / run_sdpo_rollouts stay consistent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import evaluate_prediction
from src.evaluation.real_evaluator import evaluate_real_prediction
from src.executor.mock_telco_api import MockTelcoApi
from src.model.prompt_builder import build_prompt_messages
from src.registry.contract_registry import ContractRegistry
from src.registry.tool_registry import ToolRegistry

REAL_SOURCE = "real_tool_xlsx"


def is_real_sample(sample: dict[str, Any]) -> bool:
    return sample.get("source") == REAL_SOURCE


@dataclass
class RealAssets:
    registry: ToolRegistry
    references: dict[str, Any] | None


def load_real_assets(data_dir: str | Path) -> RealAssets | None:
    data_dir = Path(data_dir)
    if not (data_dir / "real_tools.json").exists():
        return None
    registry = ToolRegistry.from_file(data_dir / "real_tools.json")
    # Real KPI tools are read-only → no contracts (no preconditions/permissions).
    ref_path = data_dir / "real_reference_codes.json"
    references = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    return RealAssets(registry, references)


def build_sample_prompt(
    sample: dict[str, Any],
    tool_registry: ToolRegistry,
    contract_registry: ContractRegistry,
    real_assets: RealAssets | None,
) -> list[dict[str, str]]:
    extra = [sample["masked_tool"]] if sample.get("masked_tool") else None
    if is_real_sample(sample) and real_assets:
        # Real = read-only → no contract context; inject reference codes for name↔code mapping.
        return build_prompt_messages(sample, real_assets.registry, None, extra_tools=extra,
                                     references=real_assets.references)
    return build_prompt_messages(sample, tool_registry, contract_registry, extra_tools=extra)


def evaluate_sample(
    sample: dict[str, Any],
    prediction: dict[str, Any],
    tool_registry: ToolRegistry,
    contract_registry: ContractRegistry,
    data_dir: str | Path,
    real_assets: RealAssets | None,
) -> dict[str, Any]:
    if is_real_sample(sample) and real_assets:
        return evaluate_real_prediction(
            sample, prediction, real_assets.registry, references=real_assets.references
        )
    # Fresh executor per sample keeps write-call evaluation independent of mutation.
    state = MockTelcoApi.from_file(Path(data_dir) / "mock_telco_db.json")
    return evaluate_prediction(sample, prediction, tool_registry, state, contract_registry)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ContractRegistry:
    def __init__(self, contracts: list[dict[str, Any]]) -> None:
        self._contracts = {contract["tool_name"]: contract for contract in contracts}

    @classmethod
    def from_file(cls, path: str | Path) -> "ContractRegistry":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls(json.load(file))

    def get(self, tool_name: str) -> dict[str, Any] | None:
        return self._contracts.get(tool_name)

    def all(self) -> list[dict[str, Any]]:
        return list(self._contracts.values())

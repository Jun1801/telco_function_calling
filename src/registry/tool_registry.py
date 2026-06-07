from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ToolRegistry:
    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self._tools = {tool["name"]: tool for tool in tools}

    @classmethod
    def from_file(cls, path: str | Path) -> "ToolRegistry":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls(json.load(file))

    def get(self, name: str) -> dict[str, Any] | None:
        return self._tools.get(name)

    def require(self, name: str) -> dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[dict[str, Any]]:
        return list(self._tools.values())

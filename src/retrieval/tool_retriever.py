"""BM25-based tool retriever for zero-shot tool selection.

Replaces gold-seeded tool injection with blind retrieval:
  query → BM25 top-k tools → prompt context

This enables realistic deployment evaluation where gold tool names
are unknown at inference time.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.registry.tool_registry import ToolRegistry


def _tool_text(tool: dict[str, Any]) -> str:
    """Concatenate tool name, description, and parameter descriptions for indexing."""
    parts = [tool.get("name", ""), tool.get("description", "")]
    props = tool.get("parameters", {}).get("properties", {})
    for param, meta in props.items():
        parts.append(param)
        if isinstance(meta, dict) and meta.get("description"):
            parts.append(meta["description"])
    return " ".join(p for p in parts if p)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class ToolRetriever:
    """BM25 retriever over tool registry.

    Falls back to token-overlap scoring if rank-bm25 is not installed.
    """

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self._tools = tools
        self._corpus = [_tokenize(_tool_text(t)) for t in tools]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            self._bm25 = None

    @classmethod
    def from_registry(cls, registry: "ToolRegistry") -> "ToolRetriever":
        return cls(registry.all())

    def retrieve(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Return top-k tool dicts ranked by BM25 score against query."""
        tokens = _tokenize(query)
        if not tokens:
            return self._tools[:k]

        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokens)
        else:
            # Simple token overlap fallback
            query_set = set(tokens)
            scores = [len(query_set & set(doc)) for doc in self._corpus]

        ranked = sorted(range(len(self._tools)), key=lambda i: scores[i], reverse=True)
        return [self._tools[i] for i in ranked[:k]]

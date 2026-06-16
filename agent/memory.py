"""Layered memory: short-term context management + long-term vector recall.

* :class:`ShortTermMemory` keeps a sliding window of recent messages and falls
  back to summarization when the window would blow the token budget.
* :class:`LongTermMemory` is a zero-dependency in-memory vector store using NumPy
  cosine similarity, so it runs with no external database.

Both are intentionally simple; the README's design notes discuss what you'd
swap in to scale them (a real vector DB, persistent storage, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .llm import BaseLLM, estimate_tokens


class ShortTermMemory:
    """Sliding-window message memory with a summarization fallback.

    Keeps the system message (if any) plus the most recent ``window`` messages.
    When the retained transcript exceeds ``max_tokens``, older messages are
    compressed into a single summary message via the LLM.
    """

    def __init__(
        self,
        llm: BaseLLM,
        window: int = 12,
        max_tokens: int = 4000,
    ) -> None:
        self._llm = llm
        self.window = window
        self.max_tokens = max_tokens

    def manage(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a (possibly compressed) view of ``messages`` for the next call."""

        if self._token_count(messages) <= self.max_tokens and len(messages) <= self.window:
            return messages

        system = messages[0:1] if messages and messages[0].get("role") == "system" else []
        body = messages[len(system):]

        # Keep the freshest `window` messages verbatim; summarize the rest.
        recent = body[-self.window:]
        older = body[: len(body) - len(recent)]
        if not older:
            return system + recent

        summary = self._summarize(older)
        summary_msg = {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        }
        return system + [summary_msg] + recent

    def _summarize(self, messages: List[Dict[str, Any]]) -> str:
        transcript = "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in messages if m.get("content")
        )
        prompt = [
            {
                "role": "system",
                "content": "Summarize the conversation below concisely, "
                "preserving facts, decisions, and tool results.",
            },
            {"role": "user", "content": transcript},
        ]
        resp = self._llm.chat(prompt, tools=[])
        return (resp.content or transcript)[:2000]

    @staticmethod
    def _token_count(messages: List[Dict[str, Any]]) -> int:
        return sum(estimate_tokens(str(m.get("content") or "")) for m in messages)


@dataclass
class MemoryRecord:
    """A single long-term memory entry."""

    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class LongTermMemory:
    """In-memory vector store using cosine similarity.

    No external dependencies beyond NumPy, so it runs with zero setup. Swap in
    FAISS/Qdrant/pgvector behind this same ``add``/``search`` interface to scale.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm
        self._records: List[MemoryRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = np.asarray(self._llm.embed(text), dtype=np.float32)
        self._records.append(MemoryRecord(text=text, embedding=vec, metadata=metadata or {}))

    def search(self, query: str, k: int = 3) -> List[tuple[str, float]]:
        """Return up to ``k`` ``(text, score)`` pairs ranked by cosine similarity."""

        if not self._records:
            return []
        q = np.asarray(self._llm.embed(query), dtype=np.float32)
        q_norm = np.linalg.norm(q)
        scored: List[tuple[str, float]] = []
        for rec in self._records:
            denom = q_norm * np.linalg.norm(rec.embedding)
            score = float(np.dot(q, rec.embedding) / denom) if denom else 0.0
            scored.append((rec.text, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

"""Query-aware context compression.

Long-context agents spend most of their tokens re-sending tool outputs and
documents the model has already seen. Recent research -- e.g. *Latent Context
Language Models* (Chari et al., 2025), which compress the input sequence before
the decoder and report up to 16x compression, and *ACON*, which targets
long-horizon LLM agents specifically -- shows that aggressively shrinking the
input before it reaches the model preserves task accuracy while cutting compute
and latency.

This module implements a lightweight, **model-free approximation** of that idea,
suitable for a from-scratch harness: a *selective, query-aware extractive
compressor*. It splits context into units (sentences/lines), scores each unit
for relevance to the current query, and keeps only the highest-value units up to
a target compression ratio. It needs no trained encoder, runs with zero
dependencies beyond the stdlib, and is fully deterministic.

It is intentionally not the trained encoder-decoder of the papers; it captures
the same *interface and goal* -- "compress the input before the LLM call" -- so
the harness can demonstrate the technique and measure its effect on token usage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm import BaseLLM, estimate_tokens

# A tiny stopword set so query/unit overlap focuses on content words.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "for", "and", "or", "but", "with", "as", "at", "by",
    "from", "this", "that", "these", "those", "it", "its", "what", "which",
    "who", "how", "when", "where", "why", "do", "does", "did", "can", "will",
    "i", "you", "he", "she", "they", "we", "me", "my", "your", "about", "into",
}


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS]


def _split_units(text: str) -> List[str]:
    """Split context into sentence/line units, preserving non-empty pieces."""

    # Split on line breaks first, then on sentence punctuation.
    units: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        units.extend(p.strip() for p in parts if p.strip())
    return units or ([text.strip()] if text.strip() else [])


@dataclass
class CompressionResult:
    """The outcome of compressing one block of context."""

    text: str
    original_tokens: int
    compressed_tokens: int
    kept_units: int
    total_units: int

    @property
    def ratio(self) -> float:
        """Achieved compression ratio (e.g. 4.0 == 4x smaller)."""

        return self.original_tokens / max(1, self.compressed_tokens)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    def summary(self) -> str:
        return (
            f"{self.original_tokens}->{self.compressed_tokens} tokens "
            f"({self.ratio:.1f}x, kept {self.kept_units}/{self.total_units} units)"
        )


class ContextCompressor:
    """Selective, query-aware extractive context compressor.

    Args:
        target_ratio: Desired compression factor (e.g. 4.0 keeps ~1/4 of tokens).
        min_tokens: Skip compression for context below this size (not worth it).
        llm: Optional LLM; when provided with ``use_embeddings=True``, embedding
            cosine similarity augments the lexical relevance score.
        use_embeddings: Whether to blend in embedding similarity.
    """

    def __init__(
        self,
        target_ratio: float = 4.0,
        min_tokens: int = 80,
        llm: Optional[BaseLLM] = None,
        use_embeddings: bool = False,
    ) -> None:
        if target_ratio < 1.0:
            raise ValueError("target_ratio must be >= 1.0")
        self.target_ratio = target_ratio
        self.min_tokens = min_tokens
        self.llm = llm
        self.use_embeddings = use_embeddings and llm is not None

    def compress(
        self, context: str, query: str, target_ratio: Optional[float] = None
    ) -> CompressionResult:
        """Compress ``context`` toward ``target_ratio``, keeping query-relevant units."""

        ratio = target_ratio or self.target_ratio
        original_tokens = estimate_tokens(context)
        units = _split_units(context)

        # Too small or nothing to drop -> return unchanged.
        if original_tokens < self.min_tokens or len(units) <= 1:
            return CompressionResult(
                text=context,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                kept_units=len(units),
                total_units=len(units),
            )

        scored = self._score_units(units, query)
        budget = max(1, int(original_tokens / ratio))

        # Greedily take highest-scoring units until the token budget is hit,
        # always keeping at least one unit.
        order = sorted(range(len(units)), key=lambda i: scored[i], reverse=True)
        keep: set[int] = set()
        used = 0
        for idx in order:
            unit_tokens = estimate_tokens(units[idx])
            if keep and used + unit_tokens > budget:
                continue
            keep.add(idx)
            used += unit_tokens
            if used >= budget:
                break

        kept_in_order = [units[i] for i in sorted(keep)]
        compressed = " ".join(kept_in_order)
        return CompressionResult(
            text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=estimate_tokens(compressed),
            kept_units=len(kept_in_order),
            total_units=len(units),
        )

    def compress_messages(
        self, messages: List[Dict[str, Any]], query: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Compress large message bodies in a transcript before an LLM call.

        Returns a new message list plus the total tokens saved. The system
        message and the most recent user message are never compressed (they carry
        the instructions and the live request).
        """

        out: List[Dict[str, Any]] = []
        saved = 0
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=-1,
        )
        for i, msg in enumerate(messages):
            content = msg.get("content")
            if (
                msg.get("role") in ("system", "tool", "assistant")
                and i != last_user_idx
                and isinstance(content, str)
                and estimate_tokens(content) >= self.min_tokens
            ):
                result = self.compress(content, query)
                if result.tokens_saved > 0:
                    saved += result.tokens_saved
                    out.append({**msg, "content": result.text})
                    continue
            out.append(msg)
        return out, saved

    # -- scoring ----------------------------------------------------------
    def _score_units(self, units: List[str], query: str) -> List[float]:
        q_words = set(_content_words(query))
        embed_scores = self._embedding_scores(units, query) if self.use_embeddings else None

        scores: List[float] = []
        for i, unit in enumerate(units):
            u_words = _content_words(unit)
            if not u_words:
                scores.append(0.0)
                continue
            overlap = sum(1 for w in u_words if w in q_words)
            # Normalize by unit length so long units aren't unfairly favored.
            lexical = overlap / (len(u_words) ** 0.5)
            position_bonus = 0.15 if i == 0 else 0.0  # first unit often sets context
            score = lexical + position_bonus
            if embed_scores is not None:
                score += 0.5 * embed_scores[i]
            scores.append(score)
        return scores

    def _embedding_scores(self, units: List[str], query: str) -> List[float]:
        assert self.llm is not None
        import numpy as np

        q = np.asarray(self.llm.embed(query), dtype=np.float32)
        qn = np.linalg.norm(q)
        out: List[float] = []
        for unit in units:
            v = np.asarray(self.llm.embed(unit), dtype=np.float32)
            denom = qn * np.linalg.norm(v)
            out.append(float(np.dot(q, v) / denom) if denom else 0.0)
        return out

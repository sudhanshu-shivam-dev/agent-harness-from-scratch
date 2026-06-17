"""Tests for query-aware context compression."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ContextCompressor, MockLLM, ReActAgent, ToolRegistry, tool

DOC = (
    "Warehouse A processed 12,400 units in March. "
    "The cafeteria introduced a new vegetarian menu. "
    "Shipping delays in the northern corridor were caused by severe winter weather. "
    "The company picnic was rescheduled to August. "
    "Energy costs increased due to the unusually cold winter. "
    "Customer satisfaction rose to 4.6 out of 5. "
    "Marketing launched a campaign with the new logo. "
    "The IT team migrated the inventory database over the weekend."
)


def test_compression_reduces_tokens():
    c = ContextCompressor(target_ratio=4.0, min_tokens=20)
    r = c.compress(DOC, "Why were there shipping delays in winter?")
    assert r.compressed_tokens < r.original_tokens
    assert r.ratio > 1.5
    assert r.kept_units < r.total_units


def test_compression_keeps_relevant_unit():
    c = ContextCompressor(target_ratio=4.0, min_tokens=20)
    r = c.compress(DOC, "Why were there shipping delays in winter?")
    # The most relevant sentence should survive compression.
    assert "shipping delays" in r.text.lower() or "winter" in r.text.lower()


def test_small_context_is_untouched():
    c = ContextCompressor(target_ratio=4.0, min_tokens=80)
    text = "short text"
    r = c.compress(text, "query")
    assert r.text == text
    assert r.ratio == 1.0


def test_compress_messages_preserves_system_and_last_user():
    c = ContextCompressor(target_ratio=3.0, min_tokens=20)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": DOC},
        {"role": "user", "content": "Why were there shipping delays in winter?"},
    ]
    out, saved = c.compress_messages(messages, messages[-1]["content"])
    assert out[0] == messages[0]  # system untouched
    assert out[-1] == messages[-1]  # latest user untouched
    assert saved > 0  # the large tool message was compressed


def test_agent_runs_with_compressor():
    @tool
    def calculator(expression: str) -> str:
        """Evaluate arithmetic."""
        return str(eval(expression))  # noqa: S307 - test-only

    agent = ReActAgent(
        llm=MockLLM(),
        tools=ToolRegistry([calculator]),
        compressor=ContextCompressor(target_ratio=4.0, min_tokens=20),
        max_steps=4,
    )
    result = agent.run("What is 12 plus 30?")
    assert "42" in result.answer
    assert result.success

"""Tests for the agent harness. All run against the dependency-free MockLLM."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

from agent import (
    ExecutionContext,
    LongTermMemory,
    MockLLM,
    ReActAgent,
    ShortTermMemory,
    ToolRegistry,
    tool,
)
from agent.eval.harness import EvalHarness
from basic_tools import build_agent, build_registry, calculator, web_search


# ---------------------------------------------------------------------------
# Tools / schema generation
# ---------------------------------------------------------------------------
def test_tool_decorator_builds_schema():
    schema = calculator.to_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "calculator"
    assert "expression" in fn["parameters"]["properties"]
    assert fn["parameters"]["properties"]["expression"]["type"] == "string"
    assert "expression" in fn["parameters"]["required"]
    # Docstring summary becomes the description.
    assert "arithmetic" in fn["description"].lower()


def test_tool_with_default_is_not_required():
    @tool
    def greet(name: str, excited: bool = False) -> str:
        """Greet someone.

        Args:
            name: Who to greet.
            excited: Whether to add an exclamation mark.
        """
        return f"Hi {name}{'!' if excited else ''}"

    params = greet.to_schema()["function"]["parameters"]
    assert "name" in params["required"]
    assert "excited" not in params["required"]
    assert params["properties"]["excited"]["type"] == "boolean"


def test_registry_dispatch_and_unknown():
    registry = build_registry()
    assert "calculator" in registry
    assert registry.dispatch("calculator", {"expression": "2 + 2"}) == "4"
    with pytest.raises(KeyError):
        registry.dispatch("nope", {})


def test_calculator_tool_safe_eval():
    assert calculator.run(expression="(12 + 8) * 5") == "100"
    # Malformed input is reported, not raised.
    assert "Could not evaluate" in calculator.run(expression="import os")


def test_web_search_stub():
    assert "Paris" in web_search.run(query="capital of France")
    assert "No results" in web_search.run(query="something obscure")


# ---------------------------------------------------------------------------
# ExecutionContext / budget guardrail
# ---------------------------------------------------------------------------
def test_context_budget_guard():
    ctx = ExecutionContext(max_steps=2, max_tokens=1000)
    assert not ctx.over_budget()
    ctx.new_step()
    ctx.new_step()
    assert ctx.over_budget()
    assert "max_steps" in ctx.budget_reason()

    ctx2 = ExecutionContext(max_steps=100, max_tokens=10)
    ctx2.add_tokens(50)
    assert ctx2.over_budget()
    assert "max_tokens" in ctx2.budget_reason()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def test_agent_uses_tool_and_answers():
    agent = build_agent()
    result = agent.run("What is 23 times 17?")
    assert "391" in result.answer
    assert result.stop_reason == "finished"
    assert result.success
    # Trajectory recorded a calculator action.
    actions = [s["action"]["name"] for s in result.trajectory if s["action"]]
    assert "calculator" in actions


def test_agent_finishes_within_budget():
    agent = build_agent()
    result = agent.run("Search for the capital of France.")
    assert "Paris" in result.answer
    assert result.steps <= 6


def test_agent_handles_malformed_tool_call():
    """An LLM that requests a non-existent tool should be retried then fail cleanly."""

    from agent.llm import LLMResponse, ToolCall, Usage

    class BadLLM(MockLLM):
        def chat(self, messages, tools=None):
            # Always request a tool that does not exist.
            return LLMResponse(
                tool_calls=[ToolCall(id="x", name="ghost_tool", arguments={})],
                usage=Usage(1, 1),
            )

    agent = ReActAgent(llm=BadLLM(), tools=build_registry(), max_steps=4)
    result = agent.run("do something")
    # It should not crash; it stops via budget and returns a graceful message.
    assert result.answer
    assert "ERROR" in result.answer or "stopped" in result.answer.lower()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def test_long_term_memory_recall():
    mem = LongTermMemory(MockLLM())
    mem.add("The project mascot is a red panda.")
    mem.add("The capital of France is Paris.")
    results = mem.search("Tell me about France", k=1)
    assert results
    assert "Paris" in results[0][0]


def test_short_term_memory_summarizes_when_over_budget():
    stm = ShortTermMemory(MockLLM(), window=2, max_tokens=5)
    messages = [{"role": "system", "content": "sys"}]
    for i in range(8):
        messages.append({"role": "user", "content": f"message number {i} with text"})
    managed = stm.manage(messages)
    # Should be compressed below the original length.
    assert len(managed) < len(messages)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------
def test_eval_harness_scorecard():
    harness = EvalHarness(build_agent=build_agent)
    scorecard = harness.run()
    assert scorecard.total >= 10
    # The mock LLM should solve the large majority of deterministic tasks.
    assert scorecard.success_rate >= 0.8
    assert scorecard.avg_steps > 0
    rendered = scorecard.render()
    assert "EVAL SCORECARD" in rendered

"""Example: a ReAct agent with calculator, web-search-stub, and datetime tools.

Run it directly to watch the agent reason and call tools::

    python examples/basic_tools.py

By default it uses the dependency-free :class:`MockLLM`, so it works without an
API key. Set ``OPENAI_API_KEY`` (and ``USE_OPENAI=1``) to drive a real model.
"""

from __future__ import annotations

import ast
import operator
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Allow running this file directly from the repo root (`python examples/...`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import MockLLM, OpenAILLM, ReActAgent, ToolRegistry, tool

# A small, deterministic "knowledge base" backing the web-search stub. Keeping
# results canned makes demos and eval runs reproducible (and keeps the repo
# strictly clean-room -- generic public facts only).
_SEARCH_KB = {
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "tallest mountain": "Mount Everest is the tallest mountain on Earth at 8,849 m.",
    "speed of light": "The speed of light is approximately 299,792 km/s.",
    "creator of python": "Python was created by Guido van Rossum.",
    "python": "Python was created by Guido van Rossum.",
}

# Operators allowed by the safe expression evaluator.
_ALLOWED_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST without using ``eval``."""

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Args:
        expression: An arithmetic expression, e.g. '23 * 17' or '(12 + 8) * 5'.
    """

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception as exc:  # noqa: BLE001 - report the error back to the agent
        return f"Could not evaluate '{expression}': {exc}"
    # Render whole numbers without a trailing .0 so substring checks are natural.
    if result == int(result):
        return str(int(result))
    return str(result)


@tool
def web_search(query: str) -> str:
    """Look up a fact from a small canned knowledge base (offline stub).

    Args:
        query: The search query.
    """

    q = query.lower()
    for key, value in _SEARCH_KB.items():
        if key in q:
            return value
    return f"No results found for '{query}'."


@tool
def datetime_now() -> str:
    """Return the current UTC date and time in ISO-8601 format."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_registry() -> ToolRegistry:
    """Build the demo tool registry.

    The decorator names tools after their functions; we rename ``datetime_now``
    to ``datetime`` to match the names the MockLLM's heuristics look for.
    """

    registry = ToolRegistry([calculator, web_search])
    datetime_now.name = "datetime"
    registry.register(datetime_now)
    return registry


def build_agent() -> ReActAgent:
    """Construct a ReAct agent wired with the demo tools and an LLM."""

    if os.environ.get("USE_OPENAI") and os.environ.get("OPENAI_API_KEY"):
        llm = OpenAILLM()
    else:
        llm = MockLLM()
    return ReActAgent(llm=llm, tools=build_registry(), max_steps=6)


def main() -> None:
    agent = build_agent()
    demos = [
        "What is 23 times 17?",
        "Search for the capital of France.",
        "What is today's date?",
    ]
    for task in demos:
        print(f"\n>>> TASK: {task}")
        result = agent.run(task)
        for step in result.trajectory:
            if step["action"]:
                print(f"    [step {step['index']}] action={step['action']['name']} "
                      f"args={step['action']['arguments']} -> {step['observation']}")
        print(f"    ANSWER: {result.answer}")
        print(f"    (steps={result.steps}, tokens={result.tokens}, "
              f"stop={result.stop_reason})")


if __name__ == "__main__":
    main()

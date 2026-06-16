"""The upgrade backlog: an ordered list of genuine, single-file improvements.

Each entry is a dict:

* ``id``      -- stable identifier (used only for logging).
* ``path``    -- the one file the upgrade creates or updates.
* ``message`` -- the commit message.
* ``applied`` -- ``(root) -> bool``: True if the upgrade is already in the repo.
* ``render``  -- ``(root) -> str``: the full new contents for ``path``.

The runner applies the first entry whose ``applied`` is False, one per run. Add
your own entries to the bottom; keep each one scoped to a single file so the
commit stays Verified.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _exists(path: str):
    return lambda root: os.path.exists(os.path.join(root, path))


def _static(text: str):
    return lambda root: text


def _eval_applied(task_id: str):
    def check(root: str) -> bool:
        p = os.path.join(root, "agent/eval/tasks.json")
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return True  # nothing to append to; treat as done
        return any(t.get("id") == task_id for t in data)

    return check


def _eval_append(new_task: Dict[str, Any]):
    def render(root: str) -> str:
        p = os.path.join(root, "agent/eval/tasks.json")
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        if not any(t.get("id") == new_task["id"] for t in data):
            data.append(new_task)
        return json.dumps(data, indent=2) + "\n"

    return render


# ---------------------------------------------------------------------------
# Backlog content
# ---------------------------------------------------------------------------
_CONTRIBUTING = """# Contributing

Thanks for your interest in improving agent-harness-from-scratch!

## Setup

```bash
pip install -r requirements.txt
pytest -q
```

Everything runs on the dependency-free `MockLLM`, so you can develop and test
without an API key.

## Guidelines

- Keep the core (`agent/`) framework-free and well typed.
- Add a test for any new behavior (`tests/`).
- New example tools belong in `examples/`; new eval tasks in
  `agent/eval/tasks.json`.
- Run `pytest -q` before opening a PR.
"""

_EDITORCONFIG = """root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space

[*.py]
indent_size = 4
max_line_length = 100

[*.{json,yml,yaml}]
indent_size = 2
"""

_CHANGELOG = """# Changelog

All notable changes to this project are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/) loosely and
[Semantic Versioning](https://semver.org/).

## [0.1.0]

### Added
- ReAct agent with explicit `think()` / `act()` / `step()` / `run()` loop.
- `ExecutionContext` with step history and a token/step budget guard.
- `@tool` decorator with automatic JSON-schema generation + `ToolRegistry`.
- Layered memory: short-term window + summarization, long-term vector recall.
- Evaluation harness with rule-based and optional LLM-as-judge scoring.
- Dependency-free `MockLLM` so the repo runs without an API key.
"""

_PYPROJECT = """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-harness-from-scratch"
version = "0.1.0"
description = "A minimal, production-shaped ReAct agent framework in pure Python."
readme = "README.md"
requires-python = ">=3.9"
license = { text = "MIT" }
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
openai = ["openai>=1.0"]
dev = ["pytest>=7.0"]

[tool.setuptools]
packages = ["agent", "agent.eval"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
"""

_MAKEFILE = """.PHONY: install test example eval

install:
\tpip install -r requirements.txt

test:
\tpytest -q

example:
\tpython examples/basic_tools.py

eval:
\tpython examples/run_eval.py
"""

_ARCHITECTURE = """# Architecture

A deeper look at how the pieces fit together. See the README for the diagram.

## The loop

`ReActAgent.run(task)` creates one `ExecutionContext` and iterates:

1. **think** -- `llm.chat(messages, tools)` after short-term memory management.
   Returns either tool calls or a final answer.
2. **act** -- dispatch tool calls through the `ToolRegistry`, append observations
   back into the transcript. Malformed calls are retried once, then fail cleanly.
3. **budget check** -- before every step, `ExecutionContext.over_budget()` stops
   the loop at the step or token ceiling.

## Why a single context object

All mutable run state lives in `ExecutionContext`: the message transcript,
scratch `state`, the recorded `steps` trajectory, and the budget counters. One
owner means one place to serialize for logging, one place to enforce the budget,
and a clean reset between tasks (the eval harness builds a fresh agent per task).

## Memory layers

- **Short-term**: a sliding window over recent messages with a summarization
  fallback when the window would exceed the token budget.
- **Long-term**: an in-memory vector store (NumPy cosine similarity) that recalls
  relevant facts from past runs and primes the system prompt.

## Extending

- New tools: subclass `BaseTool` or decorate a function with `@tool`.
- New LLM providers: implement `BaseLLM.chat` and `BaseLLM.embed`.
- New eval tasks: add to `agent/eval/tasks.json`.
"""

_SECURITY = """# Security Policy

## Reporting a vulnerability

This is an educational project. If you find a security issue, please open an
issue describing it (omit any sensitive exploit details in public) or contact
the maintainer directly.

## Notes

- The example `calculator` tool uses a restricted AST evaluator, **not** Python's
  `eval`, to avoid arbitrary code execution.
- The `web_search` tool is an offline stub backed by a canned dictionary; it
  makes no network calls.
- Never commit real API keys. Use `.env` (git-ignored) for `OPENAI_API_KEY`.
"""

_CODE_OF_CONDUCT = """# Code of Conduct

This project adopts the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/).

Be respectful, assume good intent, and keep discussion focused on the work.
Harassment of any kind is not tolerated. Maintainers may remove comments or
contributions that violate these principles.
"""

_MEMORY_DEMO = '''"""Example: long-term vector memory recall across tasks.

    python examples/memory_demo.py

Shows how facts stored from earlier runs are recalled to prime later prompts.
Runs on the dependency-free MockLLM.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import LongTermMemory, MockLLM


def main() -> None:
    mem = LongTermMemory(MockLLM())
    mem.add("The project mascot is a red panda named Pip.")
    mem.add("The capital of France is Paris.")
    mem.add("The speed of light is about 299,792 km/s.")

    for query in ["Tell me about France", "What is the mascot?", "How fast is light?"]:
        top = mem.search(query, k=1)
        text, score = top[0]
        print(f"query={query!r}\\n  -> {text}  (score={score:.3f})\\n")


if __name__ == "__main__":
    main()
'''

_TEST_TOOLS = '''"""Extra tests for the tool abstraction and schema generation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ToolRegistry, tool


def test_optional_param_not_required():
    @tool
    def fetch(url: str, retries: int = 3) -> str:
        """Fetch a URL.

        Args:
            url: The URL to fetch.
            retries: How many times to retry.
        """
        return f"{url}:{retries}"

    params = fetch.to_schema()["function"]["parameters"]
    assert "url" in params["required"]
    assert "retries" not in params["required"]
    assert params["properties"]["retries"]["type"] == "integer"


def test_registry_names_and_len():
    @tool
    def a(x: str) -> str:
        """A."""
        return x

    reg = ToolRegistry([a])
    assert len(reg) == 1
    assert reg.names() == ["a"]
'''


# ---------------------------------------------------------------------------
# The ordered backlog
# ---------------------------------------------------------------------------
UPGRADES: List[Dict[str, Any]] = [
    {
        "id": "contributing",
        "path": "CONTRIBUTING.md",
        "message": "Add contributing guide",
        "applied": _exists("CONTRIBUTING.md"),
        "render": _static(_CONTRIBUTING),
    },
    {
        "id": "editorconfig",
        "path": ".editorconfig",
        "message": "Add editorconfig for consistent formatting",
        "applied": _exists(".editorconfig"),
        "render": _static(_EDITORCONFIG),
    },
    {
        "id": "changelog",
        "path": "CHANGELOG.md",
        "message": "Add changelog",
        "applied": _exists("CHANGELOG.md"),
        "render": _static(_CHANGELOG),
    },
    {
        "id": "pyproject",
        "path": "pyproject.toml",
        "message": "Add pyproject.toml for packaging and pytest config",
        "applied": _exists("pyproject.toml"),
        "render": _static(_PYPROJECT),
    },
    {
        "id": "makefile",
        "path": "Makefile",
        "message": "Add Makefile with common dev targets",
        "applied": _exists("Makefile"),
        "render": _static(_MAKEFILE),
    },
    {
        "id": "architecture-doc",
        "path": "docs/ARCHITECTURE.md",
        "message": "Add architecture deep-dive doc",
        "applied": _exists("docs/ARCHITECTURE.md"),
        "render": _static(_ARCHITECTURE),
    },
    {
        "id": "memory-demo",
        "path": "examples/memory_demo.py",
        "message": "Add long-term memory recall example",
        "applied": _exists("examples/memory_demo.py"),
        "render": _static(_MEMORY_DEMO),
    },
    {
        "id": "extra-tool-tests",
        "path": "tests/test_tools.py",
        "message": "Add extra tests for tool schema generation",
        "applied": _exists("tests/test_tools.py"),
        "render": _static(_TEST_TOOLS),
    },
    {
        "id": "security-policy",
        "path": "SECURITY.md",
        "message": "Add security policy",
        "applied": _exists("SECURITY.md"),
        "render": _static(_SECURITY),
    },
    {
        "id": "code-of-conduct",
        "path": "CODE_OF_CONDUCT.md",
        "message": "Add code of conduct",
        "applied": _exists("CODE_OF_CONDUCT.md"),
        "render": _static(_CODE_OF_CONDUCT),
    },
    {
        "id": "eval-task-square",
        "path": "agent/eval/tasks.json",
        "message": "Add multiplication eval task",
        "applied": _eval_applied("calc-square"),
        "render": _eval_append(
            {
                "id": "calc-square",
                "prompt": "What is 15 times 15?",
                "expect_substrings": ["225"],
                "expect_tool": "calculator",
            }
        ),
    },
    {
        "id": "eval-task-japan-search",
        "path": "agent/eval/tasks.json",
        "message": "Add Japan capital eval task",
        "applied": _eval_applied("search-japan-2"),
        "render": _eval_append(
            {
                "id": "search-japan-2",
                "prompt": "Search for the capital of Japan.",
                "expect_substrings": ["Tokyo"],
                "expect_tool": "web_search",
            }
        ),
    },
]

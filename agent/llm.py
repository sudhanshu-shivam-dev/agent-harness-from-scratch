"""Thin LLM client wrappers.

This module defines a small, provider-agnostic interface (:class:`BaseLLM`) for
chat completions with tool-calling, plus two concrete implementations:

* :class:`MockLLM` -- a deterministic, dependency-free "LLM" that is good enough
  to drive the ReAct loop over the sample eval tasks. It lets the whole repo run
  (and CI pass) **without an API key**.
* :class:`OpenAILLM` -- a real client that talks to the OpenAI chat-completions
  API when ``OPENAI_API_KEY`` is set and the ``openai`` package is installed.

Keeping the surface area tiny is deliberate: the point of this project is to show
the agent *internals*, not to wrap every provider feature.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Usage:
    """Token accounting for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Normalized response returned by every :class:`BaseLLM`."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token).

    Good enough for the budget guard; avoids pulling in ``tiktoken``.
    """

    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseLLM(ABC):
    """Provider-agnostic chat interface used by the agent."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Run one chat completion, optionally exposing ``tools``."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for ``text`` (used by long-term memory)."""


# ---------------------------------------------------------------------------
# Mock implementation (zero-dependency, deterministic)
# ---------------------------------------------------------------------------
# Canned "knowledge base" so web-search-style tasks are deterministic.
_MOCK_SEARCH_KB: Dict[str, str] = {
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "tallest mountain": "Mount Everest is the tallest mountain on Earth at 8,849 m.",
    "speed of light": "The speed of light is approximately 299,792 km/s.",
    "python creator": "Python was created by Guido van Rossum.",
}

_WORD_OPS = {
    "plus": "+",
    "add": "+",
    "added to": "+",
    "minus": "-",
    "subtract": "-",
    "times": "*",
    "multiplied by": "*",
    "multiply": "*",
    "divided by": "/",
    "divide": "/",
}


class MockLLM(BaseLLM):
    """A deterministic stand-in for a real LLM.

    The behaviour is intentionally simple but it is enough to exercise the full
    think -> act -> observe loop:

    1. If the latest turn contains a tool result (role ``tool``), synthesize a
       final answer that embeds the observation(s).
    2. Otherwise, inspect the user's request and decide which registered tool to
       call, extracting arguments heuristically.
    3. If no tool fits, answer directly.

    It never relies on randomness, so eval runs are reproducible.
    """

    def __init__(self, **_: Any) -> None:
        # Kept for API parity with OpenAILLM (e.g. model/temperature kwargs).
        pass

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        tools = tools or []
        tool_names = {t["function"]["name"] for t in tools}
        prompt_tokens = estimate_tokens(json.dumps(messages))

        # (0) Grader mode: if asked to act as an LLM-as-judge, return PASS/FAIL.
        if self._is_grader_prompt(messages):
            verdict = self._grade(messages)
            return LLMResponse(content=verdict, usage=Usage(prompt_tokens, 1))

        # (1) Do we already have observations to answer from?
        observations = [m for m in messages if m.get("role") == "tool"]
        if observations:
            answer = self._synthesize_answer(messages, observations)
            return LLMResponse(
                content=answer,
                usage=Usage(prompt_tokens, estimate_tokens(answer)),
            )

        # (2) Decide on a tool based on the most recent user message.
        user_text = self._last_user_text(messages)
        decision = self._decide_tool(user_text, tool_names)
        if decision is not None:
            name, args = decision
            call = ToolCall(id=f"call_{name}", name=name, arguments=args)
            return LLMResponse(
                tool_calls=[call],
                usage=Usage(prompt_tokens, 8),
            )

        # (3) Nothing fits -> answer directly.
        answer = f"I don't have a tool to handle: {user_text!r}."
        return LLMResponse(
            content=answer,
            usage=Usage(prompt_tokens, estimate_tokens(answer)),
        )

    def embed(self, text: str) -> List[float]:
        """Deterministic hashing bag-of-words embedding (256 dims).

        Not semantically rich, but stable and dependency-free, which is all the
        in-memory vector store needs for a runnable demo.
        """

        dims = 256
        vec = [0.0] * dims
        for token in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _is_grader_prompt(messages: List[Dict[str, Any]]) -> bool:
        for m in messages:
            if m.get("role") == "system":
                content = str(m.get("content") or "").lower()
                if "grader" in content or "'pass' or" in content:
                    return True
        return False

    @staticmethod
    def _grade(messages: List[Dict[str, Any]]) -> str:
        """Heuristic judge: FAIL on empty/error answers, PASS otherwise."""

        answer = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                answer = str(m.get("content") or "").lower()
                break
        if not answer or "error" in answer or "stopped" in answer or "don't have a tool" in answer:
            return "FAIL"
        return "PASS"

    @staticmethod
    def _last_user_text(messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content") or "")
        return ""

    def _decide_tool(
        self, text: str, tool_names: set
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        lower = text.lower()

        if "calculator" in tool_names:
            expr = self._extract_expression(text)
            if expr:
                return "calculator", {"expression": expr}

        if "datetime" in tool_names and any(
            kw in lower for kw in ("time", "date", "day", "today", "now")
        ):
            return "datetime", {}

        if "web_search" in tool_names and any(
            kw in lower
            for kw in ("search", "who", "what is", "capital", "tallest", "speed of")
        ):
            return "web_search", {"query": text}

        return None

    @staticmethod
    def _extract_expression(text: str) -> Optional[str]:
        """Pull an arithmetic expression out of free text."""

        lowered = text.lower()
        for word, op in _WORD_OPS.items():
            lowered = lowered.replace(word, op)
        # Keep only math-ish characters.
        candidate = re.sub(r"[^0-9\.\+\-\*\/\(\)% ]", " ", lowered)
        candidate = candidate.strip()
        # Require at least one operator and one digit to count as an expression.
        if re.search(r"\d", candidate) and re.search(r"[\+\-\*\/%]", candidate):
            return re.sub(r"\s+", " ", candidate).strip()
        return None

    @staticmethod
    def _synthesize_answer(
        messages: List[Dict[str, Any]], observations: List[Dict[str, Any]]
    ) -> str:
        latest = str(observations[-1].get("content") or "").strip()
        return f"Based on the tool result, the answer is: {latest}"


# ---------------------------------------------------------------------------
# Real OpenAI implementation
# ---------------------------------------------------------------------------
class OpenAILLM(BaseLLM):
    """Wraps the OpenAI chat-completions API.

    Imported lazily so the repo stays importable without the ``openai`` package.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        embed_model: str = "text-embedding-3-small",
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only w/o openai
            raise RuntimeError(
                "The 'openai' package is required for OpenAILLM. "
                "Install it or use MockLLM."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; use MockLLM instead.")

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.embed_model = embed_model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message

        tool_calls: List[ToolCall] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )

        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return LLMResponse(content=choice.content, tool_calls=tool_calls, usage=usage)

    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self.embed_model, input=text)
        return list(resp.data[0].embedding)

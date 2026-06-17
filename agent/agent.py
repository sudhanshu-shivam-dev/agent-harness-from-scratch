"""The ReAct agent: an explicit think -> act -> observe loop with guardrails.

The loop is split into small, testable methods on purpose:

* :meth:`ReActAgent.think` -- one LLM call that yields either tool calls or a
  final answer.
* :meth:`ReActAgent.act` -- dispatch the requested tool(s) and capture
  observations (with safe handling of malformed calls).
* :meth:`ReActAgent.step` -- one full iteration (think then act/finish).
* :meth:`ReActAgent.run` -- drive ``step`` until the agent finishes or the
  :class:`~agent.context.ExecutionContext` budget guard trips.

Guardrails included: a hard step/token budget, a finish check, and
retry-once-then-fail handling of malformed tool calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .compression import ContextCompressor
from .context import ExecutionContext, Step
from .llm import BaseLLM, LLMResponse, ToolCall
from .memory import LongTermMemory, ShortTermMemory
from .tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful ReAct agent. Reason step by step. Use the provided tools "
    "when they help answer the user's request. When you have enough information, "
    "respond with a final answer and do not call any more tools."
)


@dataclass
class AgentResult:
    """The outcome of an agent run."""

    answer: str
    success: bool
    steps: int
    tokens: int
    stop_reason: str
    trajectory: List[Dict[str, Any]]


class ReActAgent:
    """A minimal but production-shaped ReAct agent."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        max_tokens: int = 100_000,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        compressor: Optional[ContextCompressor] = None,
        max_tool_retries: int = 1,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.short_term = short_term or ShortTermMemory(llm)
        self.long_term = long_term
        self.compressor = compressor
        self.max_tool_retries = max_tool_retries

    # -- public API -------------------------------------------------------
    def run(self, task: str) -> AgentResult:
        """Run the agent to completion on ``task`` and return an :class:`AgentResult`."""

        ctx = ExecutionContext(max_steps=self.max_steps, max_tokens=self.max_tokens)
        ctx.add_message("system", self.system_prompt)

        # Optionally prime the prompt with relevant long-term memories.
        if self.long_term is not None and len(self.long_term):
            recalled = self.long_term.search(task, k=3)
            if recalled:
                snippet = "\n".join(f"- {text}" for text, _ in recalled)
                ctx.add_message("system", f"Relevant prior knowledge:\n{snippet}")

        ctx.add_message("user", task)

        stop_reason = "finished"
        answer: Optional[str] = None
        while True:
            if ctx.over_budget():
                stop_reason = f"budget: {ctx.budget_reason()}"
                break
            answer = self.step(ctx)
            if answer is not None:
                break

        if answer is None:
            answer = self._force_finish(ctx)
            if stop_reason == "finished":
                stop_reason = "no_answer"

        # Persist the resolved task to long-term memory for future runs.
        if self.long_term is not None:
            self.long_term.add(f"Task: {task}\nAnswer: {answer}", {"type": "task"})

        return AgentResult(
            answer=answer,
            success=stop_reason in ("finished",),
            steps=len(ctx.steps),
            tokens=ctx.tokens_used,
            stop_reason=stop_reason,
            trajectory=ctx.trajectory(),
        )

    # -- one iteration ----------------------------------------------------
    def step(self, ctx: ExecutionContext) -> Optional[str]:
        """Run one loop iteration. Return a final answer string, or ``None``."""

        step = ctx.new_step()
        response = self.think(ctx)

        if response.wants_tool:
            step.thought = response.content or "(calling tool)"
            self.act(ctx, step, response.tool_calls)
            return None

        # No tool requested -> treat the content as the final answer.
        answer = (response.content or "").strip()
        step.thought = answer
        step.observation = "final_answer"
        ctx.add_message("assistant", answer)
        return answer or self._force_finish(ctx)

    # -- think ------------------------------------------------------------
    def think(self, ctx: ExecutionContext) -> LLMResponse:
        """One LLM call.

        Applies short-term memory management, optional query-aware context
        compression (shrinking large message bodies before they reach the model),
        and token budgeting.
        """

        managed = self.short_term.manage(ctx.messages)
        if self.compressor is not None:
            query = self._latest_user(ctx)
            managed, saved = self.compressor.compress_messages(managed, query)
            ctx.state["tokens_saved_by_compression"] = (
                ctx.state.get("tokens_saved_by_compression", 0) + saved
            )
        response = self.llm.chat(managed, tools=self.tools.schemas())
        ctx.add_tokens(response.usage.total_tokens)
        return response

    @staticmethod
    def _latest_user(ctx: ExecutionContext) -> str:
        for msg in reversed(ctx.messages):
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    # -- act --------------------------------------------------------------
    def act(self, ctx: ExecutionContext, step: Step, tool_calls: List[ToolCall]) -> None:
        """Dispatch tool calls and record observations on ``step``/``ctx``.

        Malformed calls (unknown tool, bad arguments, tool exceptions) are
        retried once via an error observation fed back to the model; if the model
        keeps failing, the run ends cleanly through the budget/force-finish path.
        """

        # Record the assistant's tool-call turn in the transcript.
        ctx.add_message(
            "assistant",
            step.thought,
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ],
        )

        observations: List[str] = []
        for tc in tool_calls:
            observation = self._dispatch_one(ctx, tc)
            observations.append(observation)
            ctx.add_message("tool", observation, tool_call_id=tc.id, name=tc.name)

        step.action = {
            "name": tool_calls[0].name,
            "arguments": tool_calls[0].arguments,
        }
        step.observation = "\n".join(observations)

    def _dispatch_one(self, ctx: ExecutionContext, tc: ToolCall) -> str:
        """Execute a single tool call with retry-once-then-fail semantics."""

        retry_key = f"retries::{tc.name}"
        retries = ctx.state.get(retry_key, 0)

        if tc.name not in self.tools:
            ctx.state[retry_key] = retries + 1
            if retries < self.max_tool_retries:
                return (
                    f"ERROR: unknown tool '{tc.name}'. "
                    f"Available tools: {', '.join(self.tools.names())}. Please retry."
                )
            return f"ERROR: tool '{tc.name}' is unavailable after retry; giving up."

        if not isinstance(tc.arguments, dict):
            ctx.state[retry_key] = retries + 1
            if retries < self.max_tool_retries:
                return f"ERROR: arguments for '{tc.name}' must be a JSON object. Please retry."
            return f"ERROR: malformed arguments for '{tc.name}' after retry; giving up."

        try:
            return self.tools.dispatch(tc.name, tc.arguments)
        except TypeError as exc:
            ctx.state[retry_key] = retries + 1
            if retries < self.max_tool_retries:
                return f"ERROR calling '{tc.name}': {exc}. Check the arguments and retry."
            return f"ERROR calling '{tc.name}': {exc}. Giving up after retry."
        except Exception as exc:  # noqa: BLE001 - tools are user code; isolate failures
            return f"ERROR: tool '{tc.name}' raised: {exc}"

    # -- finish helpers ---------------------------------------------------
    def _force_finish(self, ctx: ExecutionContext) -> str:
        """Best-effort final answer when the loop ends without a clean finish."""

        for step in reversed(ctx.steps):
            if step.observation and step.observation != "final_answer":
                return f"(stopped) Last observation: {step.observation}"
        return "(stopped) No answer was produced within the configured budget."

"""Execution context: the mutable state threaded through an agent run.

The :class:`ExecutionContext` is deliberately the *only* object that owns
mutable run state. Everything the loop needs to make a decision -- the message
transcript, scratch state, the step-by-step trajectory, and the token/step
budget -- lives here. Keeping it in one place makes runs easy to log, replay,
and reason about, and it is where the budget guardrail is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """One iteration of the ReAct loop: thought -> action -> observation."""

    index: int
    thought: str = ""
    action: Optional[Dict[str, Any]] = None  # {"name": str, "arguments": dict}
    observation: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "error": self.error,
        }


@dataclass
class ExecutionContext:
    """Owns all mutable state for a single agent run.

    Attributes:
        messages: The running chat transcript (OpenAI message dicts).
        state: Free-form scratch space for tools/agents to stash data.
        steps: The recorded trajectory, one :class:`Step` per loop iteration.
        max_steps: Hard cap on loop iterations (anti-runaway guardrail).
        max_tokens: Hard cap on cumulative tokens (anti-runaway guardrail).
        tokens_used: Running total of tokens consumed across LLM calls.
    """

    messages: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    steps: List[Step] = field(default_factory=list)
    max_steps: int = 10
    max_tokens: int = 100_000
    tokens_used: int = 0

    # -- message helpers --------------------------------------------------
    def add_message(self, role: str, content: Any, **extra: Any) -> Dict[str, Any]:
        """Append a message to the transcript and return it."""

        message: Dict[str, Any] = {"role": role, "content": content}
        message.update(extra)
        self.messages.append(message)
        return message

    # -- step / trajectory helpers ---------------------------------------
    def new_step(self) -> Step:
        """Create, register, and return the next :class:`Step`."""

        step = Step(index=len(self.steps))
        self.steps.append(step)
        return step

    @property
    def current_step(self) -> Optional[Step]:
        return self.steps[-1] if self.steps else None

    # -- budget guardrail -------------------------------------------------
    def add_tokens(self, count: int) -> None:
        self.tokens_used += max(0, count)

    def over_budget(self) -> bool:
        """True if either the step or token budget has been exhausted."""

        return len(self.steps) >= self.max_steps or self.tokens_used >= self.max_tokens

    def budget_reason(self) -> Optional[str]:
        """Human-readable reason the budget tripped, or ``None``."""

        if len(self.steps) >= self.max_steps:
            return f"max_steps ({self.max_steps}) reached"
        if self.tokens_used >= self.max_tokens:
            return f"max_tokens ({self.max_tokens}) reached"
        return None

    # -- serialization ----------------------------------------------------
    def trajectory(self) -> List[Dict[str, Any]]:
        """Return the recorded steps as plain dicts (for logging/eval)."""

        return [s.to_dict() for s in self.steps]

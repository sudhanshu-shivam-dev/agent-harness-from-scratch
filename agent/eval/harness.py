"""Evaluation harness.

Loads a task set, runs an agent on each task, records the full trajectory, and
scores results two ways:

* **Rule-based** -- checks expected substrings appear in the answer and (if
  specified) that the expected tool was actually used.
* **LLM-as-judge** (optional) -- asks an LLM to grade the answer against the
  task; only runs when ``judge_llm`` is provided.

The harness is provider/tool agnostic: callers pass a ``build_agent`` factory so
each task runs with a fresh agent (clean memory and context).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..agent import AgentResult, ReActAgent
from ..llm import BaseLLM

# Default task set shipped with the repo.
DEFAULT_TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks.json")


@dataclass
class TaskResult:
    """Per-task evaluation record, including the full trajectory."""

    task_id: str
    prompt: str
    answer: str
    rule_pass: bool
    judge_pass: Optional[bool]
    used_expected_tool: Optional[bool]
    steps: int
    tokens: int
    stop_reason: str
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Scorecard:
    """Aggregate metrics across a task run."""

    total: int
    successes: int
    avg_steps: float
    avg_tokens: float
    judge_successes: Optional[int]
    results: List[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def render(self) -> str:
        """Return a printable scorecard string."""

        lines = [
            "=" * 52,
            "EVAL SCORECARD",
            "=" * 52,
            f"{'Task':<22}{'rule':<7}{'judge':<7}{'steps':<7}{'tokens':<7}",
            "-" * 52,
        ]
        for r in self.results:
            judge = "-" if r.judge_pass is None else ("ok" if r.judge_pass else "x")
            rule = "ok" if r.rule_pass else "x"
            lines.append(
                f"{r.task_id:<22}{rule:<7}{judge:<7}{r.steps:<7}{r.tokens:<7}"
            )
        lines.append("-" * 52)
        lines.append(f"Success rate (rule): {self.success_rate:.0%} "
                     f"({self.successes}/{self.total})")
        if self.judge_successes is not None:
            jr = self.judge_successes / self.total if self.total else 0.0
            lines.append(f"Success rate (judge): {jr:.0%} "
                         f"({self.judge_successes}/{self.total})")
        lines.append(f"Avg steps/task:  {self.avg_steps:.2f}")
        lines.append(f"Avg tokens/task: {self.avg_tokens:.1f}")
        lines.append("=" * 52)
        return "\n".join(lines)


class EvalHarness:
    """Runs an agent over a task set and produces a :class:`Scorecard`."""

    def __init__(
        self,
        build_agent: Callable[[], ReActAgent],
        tasks_path: str = DEFAULT_TASKS_PATH,
        judge_llm: Optional[BaseLLM] = None,
    ) -> None:
        self.build_agent = build_agent
        self.tasks_path = tasks_path
        self.judge_llm = judge_llm

    def load_tasks(self) -> List[Dict[str, Any]]:
        with open(self.tasks_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def run(self) -> Scorecard:
        tasks = self.load_tasks()
        results: List[TaskResult] = []

        for task in tasks:
            agent = self.build_agent()  # fresh agent => clean context/memory
            outcome: AgentResult = agent.run(task["prompt"])

            rule_pass = self._rule_score(task, outcome)
            used_tool = self._used_expected_tool(task, outcome)
            judge_pass = self._judge_score(task, outcome) if self.judge_llm else None

            results.append(
                TaskResult(
                    task_id=task["id"],
                    prompt=task["prompt"],
                    answer=outcome.answer,
                    rule_pass=rule_pass,
                    judge_pass=judge_pass,
                    used_expected_tool=used_tool,
                    steps=outcome.steps,
                    tokens=outcome.tokens,
                    stop_reason=outcome.stop_reason,
                    trajectory=outcome.trajectory,
                )
            )

        return self._aggregate(results)

    # -- scoring ----------------------------------------------------------
    @staticmethod
    def _rule_score(task: Dict[str, Any], outcome: AgentResult) -> bool:
        answer = outcome.answer.lower()
        substrings = task.get("expect_substrings", [])
        substrings_ok = all(s.lower() in answer for s in substrings)
        # The agent must also have actually finished, not just been force-stopped.
        return substrings_ok and outcome.stop_reason == "finished"

    @staticmethod
    def _used_expected_tool(
        task: Dict[str, Any], outcome: AgentResult
    ) -> Optional[bool]:
        expected = task.get("expect_tool")
        if not expected:
            return None
        for step in outcome.trajectory:
            action = step.get("action")
            if action and action.get("name") == expected:
                return True
        return False

    def _judge_score(self, task: Dict[str, Any], outcome: AgentResult) -> bool:
        assert self.judge_llm is not None
        prompt = [
            {
                "role": "system",
                "content": "You are a strict grader. Reply with only 'PASS' or "
                "'FAIL'. PASS only if the answer correctly addresses the task.",
            },
            {
                "role": "user",
                "content": f"Task: {task['prompt']}\nAnswer: {outcome.answer}",
            },
        ]
        resp = self.judge_llm.chat(prompt, tools=[])
        return "pass" in (resp.content or "").strip().lower()

    @staticmethod
    def _aggregate(results: List[TaskResult]) -> Scorecard:
        total = len(results)
        successes = sum(1 for r in results if r.rule_pass)
        avg_steps = sum(r.steps for r in results) / total if total else 0.0
        avg_tokens = sum(r.tokens for r in results) / total if total else 0.0
        judged = [r for r in results if r.judge_pass is not None]
        judge_successes = (
            sum(1 for r in judged if r.judge_pass) if judged else None
        )
        return Scorecard(
            total=total,
            successes=successes,
            avg_steps=avg_steps,
            avg_tokens=avg_tokens,
            judge_successes=judge_successes,
            results=results,
        )


def dump_results(scorecard: Scorecard, path: str) -> None:
    """Write the full scorecard (including trajectories) to a JSON file."""

    payload = {
        "summary": {
            "total": scorecard.total,
            "successes": scorecard.successes,
            "success_rate": scorecard.success_rate,
            "avg_steps": scorecard.avg_steps,
            "avg_tokens": scorecard.avg_tokens,
            "judge_successes": scorecard.judge_successes,
        },
        "results": [asdict(r) for r in scorecard.results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

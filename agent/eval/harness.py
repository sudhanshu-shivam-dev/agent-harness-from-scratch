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
import time
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
    # τ-bench pass^k reliability: one bool per trial run.
    # pass^k = 1.0 iff *every* trial passed; captures consistency, not just peak.
    trial_passes: List[bool] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def pass_k(self) -> float:
        """1.0 if all trials passed; 0.0 if any trial failed (τ-bench pass^k)."""
        if not self.trial_passes:
            return 1.0 if self.rule_pass else 0.0
        return 1.0 if all(self.trial_passes) else 0.0


@dataclass
class Scorecard:
    """Aggregate metrics across a task run.

    Includes the CLASSic harness dimensions (Cost/Latency/Accuracy/Stability)
    from "Beyond Accuracy" (2024) and the τ-bench pass^k reliability metric.
    """

    total: int
    successes: int
    avg_steps: float
    avg_tokens: float
    judge_successes: Optional[int]
    results: List[TaskResult] = field(default_factory=list)
    # τ-bench reliability (2024): fraction of tasks where *every* trial passed.
    pass_k: float = 0.0
    k_trials: int = 1
    # CLASSic: latency and cost dimensions.
    avg_latency_s: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def render(self) -> str:
        """Return a printable scorecard string."""

        lines = [
            "=" * 62,
            "EVAL SCORECARD",
            "=" * 62,
            f"{'Task':<22}{'rule':<7}{'judge':<7}{'pass^k':<8}{'steps':<7}{'lat(s)':<8}",
            "-" * 62,
        ]
        for r in self.results:
            judge = "-" if r.judge_pass is None else ("ok" if r.judge_pass else "x")
            rule = "ok" if r.rule_pass else "x"
            pk = f"{r.pass_k:.1f}"
            lines.append(
                f"{r.task_id:<22}{rule:<7}{judge:<7}{pk:<8}{r.steps:<7}{r.latency_s:<8.2f}"
            )
        lines.append("-" * 62)
        lines.append(f"Accuracy  (rule):  {self.success_rate:.0%} ({self.successes}/{self.total})")
        if self.judge_successes is not None:
            jr = self.judge_successes / self.total if self.total else 0.0
            lines.append(f"Accuracy  (judge): {jr:.0%} ({self.judge_successes}/{self.total})")
        lines.append(f"Stability (pass^{self.k_trials}): {self.pass_k:.0%}  "
                     f"[all {self.k_trials} trial(s) pass per task]")
        lines.append(f"Latency   (avg):   {self.avg_latency_s:.2f}s/task")
        lines.append(f"Cost      (avg):   {self.avg_tokens:.0f} tokens/task")
        lines.append(f"Steps     (avg):   {self.avg_steps:.2f}/task")
        lines.append("=" * 62)
        return "\n".join(lines)


class EvalHarness:
    """Runs an agent over a task set and produces a :class:`Scorecard`.

    Args:
        build_agent: Factory that returns a fresh :class:`ReActAgent` for each run.
        tasks_path: Path to a JSON task list.
        judge_llm: Optional LLM used for open-ended grading.
        k_trials: Number of independent trials per task for the τ-bench pass^k
            reliability metric.  pass^k = 1 iff all k trials pass.
    """

    def __init__(
        self,
        build_agent: Callable[[], ReActAgent],
        tasks_path: str = DEFAULT_TASKS_PATH,
        judge_llm: Optional[BaseLLM] = None,
        k_trials: int = 1,
    ) -> None:
        self.build_agent = build_agent
        self.tasks_path = tasks_path
        self.judge_llm = judge_llm
        self.k_trials = max(1, k_trials)

    def load_tasks(self) -> List[Dict[str, Any]]:
        with open(self.tasks_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def run(self) -> Scorecard:
        tasks = self.load_tasks()
        results: List[TaskResult] = []

        for task in tasks:
            trial_passes: List[bool] = []
            last_outcome: Optional[AgentResult] = None
            t0 = time.monotonic()

            for _ in range(self.k_trials):
                agent = self.build_agent()  # fresh agent per trial => clean context
                outcome: AgentResult = agent.run(task["prompt"])
                trial_passes.append(self._rule_score(task, outcome))
                last_outcome = outcome

            latency_s = time.monotonic() - t0
            assert last_outcome is not None

            rule_pass = trial_passes[-1]
            used_tool = self._used_expected_tool(task, last_outcome)
            judge_pass = self._judge_score(task, last_outcome) if self.judge_llm else None

            results.append(
                TaskResult(
                    task_id=task["id"],
                    prompt=task["prompt"],
                    answer=last_outcome.answer,
                    rule_pass=rule_pass,
                    judge_pass=judge_pass,
                    used_expected_tool=used_tool,
                    steps=last_outcome.steps,
                    tokens=last_outcome.tokens,
                    stop_reason=last_outcome.stop_reason,
                    trajectory=last_outcome.trajectory,
                    trial_passes=trial_passes,
                    latency_s=latency_s,
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

    def _aggregate(self, results: List[TaskResult]) -> Scorecard:
        total = len(results)
        successes = sum(1 for r in results if r.rule_pass)
        avg_steps = sum(r.steps for r in results) / total if total else 0.0
        avg_tokens = sum(r.tokens for r in results) / total if total else 0.0
        avg_latency_s = sum(r.latency_s for r in results) / total if total else 0.0
        # τ-bench pass^k: fraction of tasks where every trial passed.
        pass_k = sum(r.pass_k for r in results) / total if total else 0.0
        judged = [r for r in results if r.judge_pass is not None]
        judge_successes = (
            sum(1 for r in judged if r.judge_pass) if judged else None
        )
        return Scorecard(
            total=total,
            successes=successes,
            avg_steps=avg_steps,
            avg_tokens=avg_tokens,
            avg_latency_s=avg_latency_s,
            pass_k=pass_k,
            k_trials=self.k_trials,
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
            "pass_k": scorecard.pass_k,
            "k_trials": scorecard.k_trials,
            "avg_steps": scorecard.avg_steps,
            "avg_tokens": scorecard.avg_tokens,
            "avg_latency_s": scorecard.avg_latency_s,
            "judge_successes": scorecard.judge_successes,
        },
        "results": [{**asdict(r), "pass_k": r.pass_k} for r in scorecard.results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

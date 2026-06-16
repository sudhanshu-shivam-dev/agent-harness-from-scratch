"""Example: run the eval harness over the sample tasks and print a scorecard.

    python examples/run_eval.py

Uses the MockLLM by default (no API key needed). Pass ``--judge`` to enable the
optional LLM-as-judge scoring pass, and ``--dump PATH`` to write full
trajectories to a JSON file.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running this file directly from the repo root (`python examples/...`).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`
sys.path.insert(0, _HERE)  # examples dir -> `import basic_tools`

from agent import MockLLM, OpenAILLM, ReActAgent
from agent.eval.harness import EvalHarness, dump_results

from basic_tools import build_registry  # noqa: E402  (local example import)


def _make_llm():
    if os.environ.get("USE_OPENAI") and os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM()
    return MockLLM()


def build_agent() -> ReActAgent:
    """Factory the harness calls per task to get a fresh agent."""

    return ReActAgent(llm=_make_llm(), tools=build_registry(), max_steps=6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent eval harness.")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge.")
    parser.add_argument("--dump", metavar="PATH", help="Write results JSON to PATH.")
    args = parser.parse_args()

    judge_llm = _make_llm() if args.judge else None
    harness = EvalHarness(build_agent=build_agent, judge_llm=judge_llm)
    scorecard = harness.run()
    print(scorecard.render())

    if args.dump:
        dump_results(scorecard, args.dump)
        print(f"\nWrote full results to {args.dump}")


if __name__ == "__main__":
    main()

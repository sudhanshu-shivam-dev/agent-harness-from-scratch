"""Example: long-term vector memory recall.

    python examples/memory_demo.py

Shows how facts stored in long-term memory are recalled by semantic similarity to
prime later prompts. Runs on the dependency-free MockLLM.
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
        text, score = mem.search(query, k=1)[0]
        print(f"query={query!r}\n  -> {text}  (score={score:.3f})\n")


if __name__ == "__main__":
    main()

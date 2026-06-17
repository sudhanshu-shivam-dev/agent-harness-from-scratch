"""Example: query-aware context compression.

    python examples/context_compression.py

Demonstrates shrinking a long context before it reaches the LLM, keeping only
the units relevant to the query. Inspired by research on context compression for
long-context / long-horizon agents (e.g. Latent Context Language Models, ACON).
Runs on the dependency-free MockLLM.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ContextCompressor, MockLLM

# A long, mostly-irrelevant document; only a couple of lines answer the query.
DOCUMENT = """
The quarterly logistics report covers warehouse throughput across all regions.
Warehouse A processed 12,400 units in March, slightly below its monthly target.
The cafeteria introduced a new vegetarian menu that proved popular with staff.
Shipping delays in the northern corridor were caused by severe winter weather.
The company picnic has been rescheduled to the second weekend of August.
Return rates held steady at 3.2 percent, consistent with the prior quarter.
The new forklift fleet reduced loading times by an estimated eleven percent.
Marketing launched a campaign featuring the updated brand colors and logo.
Customer satisfaction scores rose to 4.6 out of 5 in the latest survey.
The IT team migrated the inventory system to a new database over the weekend.
Parking lot resurfacing will begin next month and last roughly two weeks.
Energy costs increased due to the unusually cold winter across the region.
""".strip()


def main() -> None:
    compressor = ContextCompressor(target_ratio=4.0, min_tokens=40)
    query = "Why were there shipping delays this winter?"

    result = compressor.compress(DOCUMENT, query)
    print(f"Query: {query}\n")
    print(f"Compression: {result.summary()}\n")
    print("Kept context:")
    print(result.text)

    # The same compressor blended with embeddings (mock embeddings here).
    embed = ContextCompressor(target_ratio=4.0, min_tokens=40, llm=MockLLM(), use_embeddings=True)
    r2 = embed.compress(DOCUMENT, query)
    print(f"\nWith embedding blend: {r2.summary()}")


if __name__ == "__main__":
    main()

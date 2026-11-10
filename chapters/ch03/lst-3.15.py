# --8<-- [start:lst-3.15]
# src/ch10_weight_context_continuum/routing_example.py
"""
Worked example: routing a mathematical proof query through L1, L2, and L3.

Demonstrates the quality/cost/latency trade-off empirically.
In production, the gating classifier in Section 10.3 routes this
automatically based on a complexity score.
"""

import asyncio
import enum
from types import SimpleNamespace


class ReasoningStrategy(enum.Enum):
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHTS = "tot"
    BEST_OF_N = "bon"


def ReasoningBudget(**kwargs):
    return SimpleNamespace(**kwargs)


class LLMClient:
    """Stub client; replace with a real async API wrapper."""


class ReasoningEngine:
    def __init__(self, budget, client):
        self.budget, self.client = budget, client

    async def reason(self, query):
        node = SimpleNamespace(status=SimpleNamespace(value="expanded"))
        trace = SimpleNamespace(nodes=[node], wall_time_s=0.0, summary=lambda: "stub")
        return SimpleNamespace(trace=trace, confidence=0.0, timed_out=False)


# Query requiring multi-step proof (justifies L2/L3)
PROOF_QUERY = (
    "Prove that for any prime p > 2, the sum of all integers from 1 to (p-1) "
    "is divisible by p. Show all intermediate steps."
)


async def run_all_levels(client: LLMClient) -> None:
    """Execute the proof query at L1, L2, and L3; log comparative results."""

    configs = [
        ("L1 – Chain-of-Thought", ReasoningBudget(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            primary_model="reasoner-model-id",  # your provider's IDs
            verifier_model="processor-model-id",
            max_thoughts=16,
            timeout_s=60.0,
        )),
        ("L2 – Tree-of-Thoughts", ReasoningBudget(
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            primary_model="reasoner-model-id",
            verifier_model="processor-model-id",
            tree_branching_factor=3,
            tree_depth=3,
            min_score_threshold=0.45,
            max_thoughts=32,
            timeout_s=120.0,
        )),
        ("L3 – Best-of-N (N=8)", ReasoningBudget(
            strategy=ReasoningStrategy.BEST_OF_N,
            primary_model="reasoner-model-id",  # your provider's IDs
            verifier_model="processor-model-id",
            best_of_n=8,
            hedge_factor=1.25,  # Launch 10 requests, keep first 8 to finish
            max_thoughts=64,
            timeout_s=300.0,
        )),
    ]

    print(f"\n{'='*70}")
    print(f"Query: {PROOF_QUERY[:80]}...")
    print(f"{'='*70}\n")

    results = []
    for label, budget in configs:
        engine = ReasoningEngine(budget=budget, client=client)
        result = await engine.reason(PROOF_QUERY)
        results.append((label, result))

        nodes = result.trace.nodes
        expanded = sum(1 for n in nodes if n.status.value == "expanded")
        pruned   = sum(1 for n in nodes if n.status.value == "pruned")

        print(f"[{label}]")
        print(f"  Confidence:    {result.confidence:.3f}")
        print(f"  Wall time:     {result.trace.wall_time_s:.1f}s")
        print(f"  Nodes total:   {len(nodes)}")
        print(f"  Nodes pruned:  {pruned}  ({pruned/max(len(nodes),1)*100:.0f}%)")
        print(f"  Timed out:     {result.timed_out}")
        print(f"  Trace:         {result.trace.summary()}")
        print()

    # Print comparative summary as Table 10.2
    print(f"\n{'='*70}")
    print("Table 10.2: Comparative Results — Three Reasoning Levels")
    print(f"{'Level':<30} {'Confidence':>12} {'Time (s)':>10} {'Pruned Nodes':>14}")
    print("-" * 70)
    for label, result in results:
        pruned = sum(1 for n in result.trace.nodes if n.status.value == "pruned")
        print(
            f"{label:<30} "
            f"{result.confidence:>12.3f} "
            f"{result.trace.wall_time_s:>10.1f} "
            f"{pruned:>14}"
        )


if __name__ == "__main__":
    # Production: replace StubClient with openai.AsyncOpenAI wrapper
    StubClient = LLMClient
    asyncio.run(run_all_levels(StubClient()))
# --8<-- [end:lst-3.15]

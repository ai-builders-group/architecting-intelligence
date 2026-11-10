# --8<-- [start:lst-3.14]
# src/ch10_weight_context_continuum/reasoning_engine.py
"""
Reasoning Engine: Executes CoT, ToT, and Best-of-N strategies.

Key invariants:
  - _verify() never raises on verifier malfunction; it returns VerificationResult.structural_penalty().
  - All tree node expansions are gated by min_score_threshold (semantic pruning).
  - Hedge requests use asyncio.wait_for to enforce timeout at the OS level.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from enum import Enum
from typing import AsyncIterator

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning types (minimal definitions used by the engine)
# ---------------------------------------------------------------------------

class ReasoningStrategy(str, Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    BEST_OF_N = "best_of_n"


class NodeStatus(str, Enum):
    EXPANDED = "expanded"
    PRUNED = "pruned"
    TERMINAL = "terminal"


class ReasoningBudget(BaseModel):
    strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    primary_model: str
    verifier_model: str
    timeout_s: float = 60.0
    verifier_timeout_s: float = 20.0
    max_thoughts: int = 4
    max_tokens_per_call: int = 4000
    stream_thoughts: bool = False
    tree_branching_factor: int = 3
    tree_depth: int = 3
    min_score_threshold: float = 0.4
    best_of_n: int = 4
    hedge_factor: float = 1.5


class TraceNode(BaseModel):
    node_id: str
    depth: int
    content_hash: str
    score: float
    status: NodeStatus
    parent_id: str | None = None


class ReasoningTrace(BaseModel):
    trace_id: str
    strategy: ReasoningStrategy
    query_hash: str
    nodes: list[TraceNode] = Field(default_factory=list)
    wall_time_s: float = 0.0
    final_score: float = 0.0
    hedge_requests_launched: int = 0
    hedge_requests_cancelled: int = 0


class ReasoningResult(BaseModel):
    answer: str
    confidence: float
    strategy_used: ReasoningStrategy
    trace: ReasoningTrace
    timed_out: bool = False


class VerificationResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    is_valid: bool
    reasoning: str = ""

    @classmethod
    def structural_penalty(cls, reason: str) -> "VerificationResult":
        return cls(score=0.1, is_valid=False, reasoning=reason[:100])


# ---------------------------------------------------------------------------
# Simulated LLM client interface (replace with openai.AsyncOpenAI in prod)
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin async wrapper around a provider API.
    Production: replace stub with openai.AsyncOpenAI or anthropic.AsyncAnthropic.
    """

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> str:
        """Single-turn generation. Returns the assistant message content."""
        raise NotImplementedError

    async def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Server-Sent Events stream of token deltas."""
        raise NotImplementedError
        yield  # make this a generator


# ---------------------------------------------------------------------------
# Reasoning Engine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """
    Executes inference-time reasoning under a configurable ReasoningBudget.

    Three strategies are supported:
      L1 — Chain-of-Thought: Linear, single-path scratchpad reasoning.
      L2 — Tree-of-Thoughts: Shallow DAG with semantic pruning at each level.
      L3 — Best-of-N: Parallel sampling with verifier-based rejection.

    For hedge-request tail-latency mitigation, L3 launches
    int(best_of_n * hedge_factor) concurrent requests, cancelling
    surplus tasks when the first best_of_n verified candidates return.
    """

    def __init__(self, budget: ReasoningBudget, client: LLMClient) -> None:
        self._budget = budget
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reason(self, query: str) -> ReasoningResult:
        """Entry point. Dispatches to the configured strategy."""
        trace_id = str(uuid.uuid4())
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        trace = ReasoningTrace(
            trace_id=trace_id,
            strategy=self._budget.strategy,
            query_hash=query_hash,
        )
        t0 = time.monotonic()

        try:
            result = await asyncio.wait_for(
                self._dispatch(query, trace),
                timeout=self._budget.timeout_s,
            )
            result.trace.wall_time_s = time.monotonic() - t0
            return result
        except asyncio.TimeoutError:
            trace.wall_time_s = self._budget.timeout_s
            return ReasoningResult(
                answer="[TIMEOUT] Reasoning budget exhausted. Best partial result unavailable.",
                confidence=0.0,
                strategy_used=self._budget.strategy,
                trace=trace,
                timed_out=True,
            )

    async def stream_thoughts(self, query: str) -> AsyncIterator[str]:
        """
        Yield intermediate reasoning tokens for UI progress indication.
        Only valid when budget.stream_thoughts is True and the primary
        model supports extended thinking (e.g., o1, o3-mini, deepseek-r1).

        Consumers should render these tokens in a collapsible "Thinking..." UI
        component and suppress them from the final answer surface.
        """
        if not self._budget.stream_thoughts:
            raise RuntimeError("stream_thoughts=False in ReasoningBudget.")

        cot_prompt = _build_cot_prompt(query)
        async for token in self._client.stream(
            model=self._budget.primary_model,
            prompt=cot_prompt,
            max_tokens=self._budget.max_tokens_per_call,
        ):
            yield token

    # ------------------------------------------------------------------
    # Strategy Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        query: str,
        trace: ReasoningTrace,
    ) -> ReasoningResult:
        match self._budget.strategy:
            case ReasoningStrategy.CHAIN_OF_THOUGHT:
                return await self._chain_of_thought(query, trace)
            case ReasoningStrategy.TREE_OF_THOUGHTS:
                return await self._tree_of_thoughts(query, trace)
            case ReasoningStrategy.BEST_OF_N:
                return await self._best_of_n(query, trace)
            case _:
                raise ValueError(f"Unknown strategy: {self._budget.strategy}")

    # ------------------------------------------------------------------
    # L1: Chain-of-Thought
    # ------------------------------------------------------------------

    async def _chain_of_thought(
        self,
        query: str,
        trace: ReasoningTrace,
    ) -> ReasoningResult:
        """
        Linear scratchpad reasoning.
        Generates one extended thinking path, then extracts the final answer.
        """
        cot_prompt = _build_cot_prompt(query)
        raw_response = await self._client.generate(
            model=self._budget.primary_model,
            prompt=cot_prompt,
            max_tokens=self._budget.max_tokens_per_call,
        )
        final_answer = _extract_final_answer(raw_response)
        verification = await self._verify(query, final_answer)

        node = TraceNode(
            node_id=str(uuid.uuid4()),
            depth=0,
            content_hash=_content_hash(final_answer),
            score=verification.score,
            status=NodeStatus.TERMINAL,
        )
        trace.nodes.append(node)
        trace.final_score = verification.score

        return ReasoningResult(
            answer=final_answer,
            confidence=verification.score,
            strategy_used=ReasoningStrategy.CHAIN_OF_THOUGHT,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # L2: Tree-of-Thoughts with Semantic Pruning
    # ------------------------------------------------------------------

    async def _tree_of_thoughts(
        self,
        query: str,
        trace: ReasoningTrace,
    ) -> ReasoningResult:
        """
        Shallow DAG search with verifier-gated pruning.

        At each depth level:
          1. Expand B children from each active node.
          2. Score all children concurrently via _verify.
          3. Prune nodes with score < min_score_threshold (semantic pruning).
          4. Continue expansion on surviving nodes.

        The pruned nodes are logged to TraceNode with status=PRUNED for
        post-hoc debugging. They are NOT expanded, saving API round-trips.
        """
        # Root: generate B initial partial solutions
        frontier: list[tuple[str, str | None]] = []  # (content, parent_id)
        root_thoughts = await self._expand_node(
            query=query,
            partial_solution="",
            n=self._budget.tree_branching_factor,
        )
        for thought in root_thoughts:
            frontier.append((thought, None))

        best_answer: str = root_thoughts[0] if root_thoughts else ""
        best_score: float = 0.0

        for depth in range(self._budget.tree_depth):
            if not frontier:
                break

            # Score all frontier nodes concurrently
            score_tasks = [
                self._verify(query, content) for content, _ in frontier
            ]
            verifications: list[VerificationResult] = await asyncio.gather(
                *score_tasks, return_exceptions=False
            )

            next_frontier: list[tuple[str, str | None]] = []

            for (content, parent_id), verification in zip(frontier, verifications):
                node_id = str(uuid.uuid4())
                is_final_depth = (depth == self._budget.tree_depth - 1)

                if verification.score >= self._budget.min_score_threshold:
                    status = NodeStatus.TERMINAL if is_final_depth else NodeStatus.EXPANDED
                    trace.nodes.append(TraceNode(
                        node_id=node_id,
                        depth=depth,
                        content_hash=_content_hash(content),
                        score=verification.score,
                        status=status,
                        parent_id=parent_id,
                    ))

                    if verification.score > best_score:
                        best_score = verification.score
                        best_answer = content

                    if not is_final_depth:
                        children = await self._expand_node(
                            query=query,
                            partial_solution=content,
                            n=self._budget.tree_branching_factor,
                        )
                        for child in children:
                            next_frontier.append((child, node_id))

                else:
                    # === SEMANTIC PRUNING: node discarded, not expanded ===
                    trace.nodes.append(TraceNode(
                        node_id=node_id,
                        depth=depth,
                        content_hash=_content_hash(content),
                        score=verification.score,
                        status=NodeStatus.PRUNED,
                        parent_id=parent_id,
                    ))
                    pruned_count = sum(
                        1 for n in trace.nodes if n.status == NodeStatus.PRUNED
                    )
                    logger.debug(
                        "Pruned node at depth=%d score=%.3f (threshold=%.3f). "
                        "Total pruned: %d.",
                        depth, verification.score,
                        self._budget.min_score_threshold, pruned_count,
                    )
                    # Node is NOT added to next_frontier. Subtree is abandoned.

            frontier = next_frontier

        trace.final_score = best_score
        return ReasoningResult(
            answer=_extract_final_answer(best_answer),
            confidence=best_score,
            strategy_used=ReasoningStrategy.TREE_OF_THOUGHTS,
            trace=trace,
        )

    async def _expand_node(
        self,
        query: str,
        partial_solution: str,
        n: int,
    ) -> list[str]:
        """Generate N continuation candidates from a partial solution."""
        expansion_prompt = _build_expansion_prompt(query, partial_solution, n)
        tasks = [
            self._client.generate(
                model=self._budget.primary_model,
                prompt=expansion_prompt,
                max_tokens=512,
            )
            for _ in range(n)
        ]
        responses = await asyncio.gather(*tasks)
        return list(responses)

    # ------------------------------------------------------------------
    # L3: Best-of-N with Hedge Requests
    # ------------------------------------------------------------------

    async def _best_of_n(
        self,
        query: str,
        trace: ReasoningTrace,
    ) -> ReasoningResult:
        """
        Parallel sampling with verifier-based rejection.

        Launches int(best_of_n * hedge_factor) concurrent generation requests
        to mitigate P99 tail-latency. As soon as best_of_n candidates have
        returned verified scores, surplus (hedge) tasks are cancelled.

        This is inference-time rejection sampling. Contrast with Section 10.6
        (training-time rejection sampling), where the same statistical
        primitive operates on a dataset over hours rather than seconds.
        """
        target_n = self._budget.best_of_n
        hedge_n  = int(target_n * self._budget.hedge_factor)
        trace.hedge_requests_launched = hedge_n

        cot_prompt = _build_cot_prompt(query)

        # Launch all hedge tasks
        generation_tasks = [
            asyncio.create_task(
                self._client.generate(
                    model=self._budget.primary_model,
                    prompt=cot_prompt,
                    max_tokens=self._budget.max_tokens_per_call,
                )
            )
            for _ in range(hedge_n)
        ]

        candidates: list[tuple[str, VerificationResult]] = []

        # Collect results as they complete; stop at target_n
        pending = set(generation_tasks)
        while pending and len(candidates) < target_n:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=self._budget.timeout_s,
            )
            for task in done:
                try:
                    raw = task.result()
                    answer = _extract_final_answer(raw)
                    verification = await self._verify(query, answer)
                    candidates.append((answer, verification))
                except Exception as exc:
                    logger.warning("Generation task failed: %s", exc)

        # Cancel surplus hedge tasks
        cancelled = 0
        for task in pending:
            task.cancel()
            cancelled += 1
        trace.hedge_requests_cancelled = cancelled

        candidates = [c for c in candidates if c[1].is_valid]
        if not candidates:
            trace.final_score = 0.0
            return ReasoningResult(
                answer="[ERROR] No valid candidate.",
                confidence=0.0,
                strategy_used=ReasoningStrategy.BEST_OF_N,
                trace=trace,
            )

        # Select best candidate
        best_answer, best_verification = max(candidates, key=lambda x: x[1].score)

        # Log all candidates to trace
        for idx, (content, verification) in enumerate(candidates):
            trace.nodes.append(TraceNode(
                node_id=str(uuid.uuid4()),
                depth=0,
                content_hash=_content_hash(content),
                score=verification.score,
                status=(
                    NodeStatus.TERMINAL
                    if content == best_answer
                    else NodeStatus.PRUNED
                ),
            ))

        trace.final_score = best_verification.score
        return ReasoningResult(
            answer=best_answer,
            confidence=best_verification.score,
            strategy_used=ReasoningStrategy.BEST_OF_N,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Verification: Parse-Don't-Validate
    # ------------------------------------------------------------------

    async def _verify(
        self,
        query: str,
        candidate_answer: str,
    ) -> VerificationResult:
        """
        Verify a candidate answer using the configured verifier model.

        Implements the Parse-Don't-Validate pattern (Chapter 1, ADR 002):
          1. Call the verifier with a JSON-schema-constrained prompt.
          2. Parse the response into VerificationResult via Pydantic.
          3. On ANY failure (malformed JSON, ValidationError, network error,
             verifier hallucination), return
             VerificationResult.structural_penalty().
             This ensures the engine never propagates raw verifier output
             upstream.

        The structural_penalty score (0.1) is intentionally non-zero. A score
        of 0.0 would make a verifier failure look like a provably wrong answer;
        is_valid=False marks it, and selection ignores invalid candidates.
        """
        verify_prompt = _build_verify_prompt(query, candidate_answer)
        t0 = time.monotonic()

        try:
            raw_response = await asyncio.wait_for(self._client.generate(
                model=self._budget.verifier_model,
                prompt=verify_prompt,
                max_tokens=256,
                response_format={"type": "json_object"},
            ), timeout=self._budget.verifier_timeout_s)
        except Exception as exc:
            return VerificationResult.structural_penalty(
                f"Verifier API call failed: {type(exc).__name__}: {exc}"
            )

        try:
            parsed_json = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return VerificationResult.structural_penalty(
                f"JSONDecodeError at position {exc.pos}: {exc.msg}"
            )

        try:
            result = VerificationResult.model_validate(parsed_json)
        except ValidationError as exc:
            # Collect all field-level validation errors for diagnostics
            error_summary = "; ".join(
                f"{e['loc']}: {e['msg']}" for e in exc.errors()
            )
            return VerificationResult.structural_penalty(
                f"Pydantic ValidationError: {error_summary}"
            )

        latency_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "Verification completed: score=%.3f valid=%s latency=%.1fms",
            result.score, result.is_valid, latency_ms,
        )
        return result


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def _build_cot_prompt(query: str) -> str:
    return f"""Reason step-by-step. Show all intermediate steps explicitly.
After completing your reasoning, write 'FINAL ANSWER:' followed by your answer.

Question: {query}"""


def _build_expansion_prompt(query: str, partial: str, n: int) -> str:
    context = f"\nCurrent approach:\n{partial}" if partial else ""
    return f"""Generate one continuation of a reasoning path for the following problem.
Be concise. Produce exactly one distinct approach in 2-3 sentences.{context}

Problem: {query}"""


def _build_verify_prompt(query: str, candidate: str) -> str:
    return f"""Evaluate the following answer to the question.
Return a JSON object with exactly these fields:
  "score": float in [0.0, 1.0] — quality/correctness score
  "is_valid": boolean — true if the answer is correct and complete
  "reasoning": string — brief explanation (max 100 chars)

Question: {query}
Candidate Answer: {candidate}

Respond ONLY with the JSON object."""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_final_answer(raw: str) -> str:
    """Extract the content after 'FINAL ANSWER:' marker, or return full text."""
    marker = "FINAL ANSWER:"
    if marker in raw:
        return raw.split(marker, 1)[1].strip()
    return raw.strip()


def _content_hash(content: str) -> str:
    """SHA-256 of the first 256 characters. Avoids logging raw PII in traces."""
    return hashlib.sha256(content[:256].encode()).hexdigest()[:12]
# --8<-- [end:lst-3.14]

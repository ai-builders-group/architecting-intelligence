# --8<-- [start:lst-3.13]
# reasoning_budget.py
"""
Reasoning Budget Pattern: Configurable inference-time compute allocation.

This module defines the schema and execution engine for the three levels of
inference-time reasoning: Chain-of-Thought (L1), Tree-of-Thoughts (L2),
and Best-of-N Rejection Sampling (L3).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, UTC
from enum import Enum
from typing import AsyncIterator, Literal

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ReasoningStrategy(str, Enum):
    """Inference-time search algorithm selection."""
    CHAIN_OF_THOUGHT = "chain_of_thought"   # L1: Linear, single path
    TREE_OF_THOUGHTS = "tree_of_thoughts"   # L2: Shallow branching DAG
    BEST_OF_N        = "best_of_n"          # L3: Parallel sampling + rejection


class NodeStatus(str, Enum):
    """State of a node in the reasoning tree."""
    EXPANDED = "expanded"
    PRUNED   = "pruned"
    TERMINAL = "terminal"


# ---------------------------------------------------------------------------
# Core Schemas
# ---------------------------------------------------------------------------

class ReasoningBudget(BaseModel):
    """
    Configurable resource contract for inference-time compute.

    Operators tune this schema per query class (e.g., code generation
    vs. FAQ retrieval) without model redeployment. See also the
    'Reasoning Budget Pattern' in the glossary.
    """

    strategy: ReasoningStrategy = Field(
        default=ReasoningStrategy.CHAIN_OF_THOUGHT,
        description="Search algorithm: linear (L1), branching (L2), or parallel (L3).",
    )
    max_thoughts: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Maximum number of requests one budget may launch.",
    )
    max_tokens_per_call: int = Field(
        default=4000,
        ge=1,
        description="Token cap on each Reasoner call.",
    )
    best_of_n: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Number of independent candidates for Best-of-N sampling (L3 only).",
    )
    tree_branching_factor: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Children per node in Tree-of-Thoughts (L2 only). "
                    "Empirical B_LLM ≈ 1.93; set to 3 for a conservative margin.",
    )
    tree_depth: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Maximum depth of Tree-of-Thoughts expansion. "
                    "At B=3, depth=4 yields 81 leaf nodes before pruning.",
    )
    min_score_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum verifier score for a node to be expanded. "
                    "Nodes scoring below this threshold are pruned and logged.",
    )
    verifier_model: str = Field(
        default="claude-3-5-haiku",
        description="Model used for intermediate verification and scoring. "
                    "Must support structured JSON output.",
    )
    primary_model: str = Field(
        default="o3-mini",
        description="Primary reasoning model. Use 'o1' for maximum quality, "
                    "'o3-mini' for cost-latency balance.",
    )
    timeout_s: float = Field(
        default=120.0,
        gt=0.0,
        le=600.0,
        description="Hard wall-clock timeout. Connections exceeding this are "
                    "cancelled; best available result is returned.",
    )
    hedge_factor: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Multiplier applied to best_of_n to launch hedge requests. "
                    "Mitigates P99 tail latency from slow API replicas.",
    )
    stream_thoughts: bool = Field(
        default=False,
        description="If True, emit intermediate reasoning tokens to the caller "
                    "for UI progress indication. Requires SSE-capable transport.",
    )

    @field_validator("primary_model")
    @classmethod
    def _validate_reasoning_model(cls, v: str) -> str:
        known_reasoning_models = {"o1", "o3-mini", "o3", "deepseek-r1"}
        if v not in known_reasoning_models:
            logger.warning(
                "primary_model '%s' is not in the known reasoning model registry %s. "
                "Ensure the model supports extended thinking tokens.",
                v, known_reasoning_models,
            )
        return v

    @model_validator(mode="after")
    def _validate_hedge_budget(self) -> "ReasoningBudget":
        """Hedge requests must not exceed max_thoughts budget."""
        effective_n = int(self.best_of_n * self.hedge_factor)
        if effective_n > self.max_thoughts:
            raise ValueError(
                f"Effective hedge requests ({effective_n}) exceed max_thoughts "
                f"({self.max_thoughts}). Reduce best_of_n or hedge_factor."
            )
        return self


class VerificationResult(BaseModel):
    """
    Structured output of the verifier model.

    Uses the Parse-Don't-Validate pattern: malformed verifier output
    is caught at the schema boundary and penalized with a structural
    failure score, never propagated as raw text.
    """
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality score in [0, 1]. 0 = incorrect/irrelevant; "
                    "1 = correct, complete, well-reasoned.",
    )
    is_valid: bool = Field(
        description="Hard correctness verdict. False if the answer is factually "
                    "wrong or structurally incomplete."
    )
    reasoning: str = Field(
        description="Brief natural language explanation of the score."
    )
    # Structural failure sentinel: set to True when the verifier itself fails.
    structural_failure: bool = Field(
        default=False,
        description="True if this result was produced by the fallback handler "
                    "due to a verifier call, timeout or parse failure.",
    )

    @classmethod
    def structural_penalty(cls, error_context: str) -> "VerificationResult":
        """
        Fallback factory for verifier hallucination or parse failure.

        Returns a penalized-but-non-zero score (0.1) to avoid NaN propagation
        in aggregation. A score of 0.0 would collapse Best-of-N selection
        to a uniform distribution over failures.
        """
        return cls(
            score=0.1,
            is_valid=False,
            reasoning=f"Verifier structural failure: {error_context}",
            structural_failure=True,
        )


class TraceNode(BaseModel):
    """
    Audit record for a single reasoning tree node.
    Required for operational visibility (anti-pattern: Operational Blindness).
    """
    node_id: str
    depth: int
    content_hash: str  # SHA-256 of first 256 chars; avoid logging raw PII
    score: float
    status: NodeStatus
    parent_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verifier_latency_ms: float = 0.0


class ReasoningTrace(BaseModel):
    """
    Complete audit log for a single reasoning execution.
    Serialized to append-only storage for post-hoc debugging.
    """
    trace_id: str
    strategy: ReasoningStrategy
    query_hash: str
    nodes: list[TraceNode] = Field(default_factory=list)
    final_score: float = 0.0
    total_tokens_used: int = 0
    wall_time_s: float = 0.0
    hedge_requests_launched: int = 0
    hedge_requests_cancelled: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def summary(self) -> str:
        expanded = sum(1 for n in self.nodes if n.status == NodeStatus.EXPANDED)
        pruned   = sum(1 for n in self.nodes if n.status == NodeStatus.PRUNED)
        return (
            f"trace_id={self.trace_id} strategy={self.strategy.value} "
            f"nodes={len(self.nodes)} expanded={expanded} pruned={pruned} "
            f"final_score={self.final_score:.3f} wall_time={self.wall_time_s:.1f}s"
        )


class ReasoningResult(BaseModel):
    """
    Deterministic container for a stochastic reasoning output.
    Follows the StochasticResult[T] protocol from Chapter 1.
    """
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    strategy_used: ReasoningStrategy
    trace: ReasoningTrace
    timed_out: bool = False
    tokens_used: int = 0
# --8<-- [end:lst-3.13]

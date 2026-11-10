# --8<-- [start:lst-9.36]
# src/08_retrieval_topology/cost_model.py
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, UTC
from enum import Enum

class PricingTier(BaseModel):
    """Token pricing for a specific model tier."""
    input_cost_per_1m: float = Field(description="USD per 1M input tokens")
    cached_cost_per_1m: float = Field(
        description="USD per 1M cached-read tokens (provider-specific)")
    output_cost_per_1m: float = Field(description="USD per 1M output tokens")
    model_name: str = Field(
        description="Canonical model identifier (e.g., 'gpt-4o', 'llama-3-70b')")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Pricing snapshot timestamp")

class RetrievalCost(BaseModel):
    """Fixed costs for the retrieval pipeline."""
    embedding_cost_per_query: float = Field(default=0.0001, description="Cost to embed the query")
    vector_search_cost_per_query: float = Field(default=0.01, description="Amortized DB query cost")
    reranking_cost_per_query: float = Field(default=0.02, description="Cross-encoder reranking cost")

class ConsistencyLevel(str, Enum):
    """Serve a stale entry during refresh, or wait for a fresh one."""
    EVENTUAL = "eventual"  # Serve stale during refresh
    STRONG = "strong"      # Wait (bounded) for a fresh cache

class ContextStrategy(BaseModel):
    """Decision metadata for routing."""
    strategy: Literal["rag_only", "full_context", "hybrid"]
    reasoning: str = Field(description="Explanation of the decision")
    expected_cost_usd: float = Field(description="Projected cost per query")
    expected_latency_ms: int = Field(description="Projected p95 latency (TTFT)")
    cache_reuse_assumption: int = Field(
        default=1, description="Queries per cache refresh")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.9, description="Confidence in this decision")
    fallback_strategy: Literal["rag_only", "reject"] = Field(
        default="rag_only", description="Action if budget exceeded")
    consistency_level: ConsistencyLevel = Field(
        default=ConsistencyLevel.EVENTUAL, description="Stale tolerance")

# Example: GPT-4o pricing (2026)
GPT4O_PRICING = PricingTier(
    model_name="gpt-4o",
    input_cost_per_1m=15.0,
    cached_cost_per_1m=1.5,
    output_cost_per_1m=60.0,
)

# Example: Llama-3-70B self-hosted (amortized)
LLAMA3_70B_PRICING = PricingTier(
    model_name="llama-3-70b",
    input_cost_per_1m=2.0,  # Self-hosted GPU amortization
    cached_cost_per_1m=0.2,
    output_cost_per_1m=5.0,
)
# --8<-- [end:lst-9.36]

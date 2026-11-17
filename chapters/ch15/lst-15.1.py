# --8<-- [start:lst-15.1]
# substrate_router.py
from enum import Enum
from typing import Protocol
from pydantic import BaseModel, Field, computed_field
from datetime import datetime, UTC

class KnowledgeSubstrate(str, Enum):
    """The three physical layers where knowledge can be stored."""
    WEIGHTS = "weights"      # Model parameters (ROM)
    CONTEXT = "context"      # Prompt/RAG (RAM)
    COMPUTE = "compute"      # Inference-time reasoning (CPU)

class KnowledgeCharacteristics(BaseModel):
    """Empirical properties of a knowledge domain."""
    
    access_frequency: float = Field(
        description="Expected queries per day",
        gt=0
    )
    update_frequency: float = Field(
        description="Expected updates per week",
        ge=0
    )
    is_user_specific: bool = Field(
        description="Whether knowledge is private (PII, session state)"
    )
    average_size_tokens: int = Field(
        description="Typical context footprint in tokens",
        gt=0
    )
    requires_reasoning: bool = Field(
        default=False,
        description="Whether domain requires multi-hop inference"
    )
    must_be_exact: bool = Field(
        default=False,
        description="Whether a wrong answer has a direct cost (a billing code), so facts must be recalled exactly"
    )
    
    @computed_field
    @property
    def volatility_score(self) -> float:
        """Normalized measure of how often knowledge changes."""
        # Weekly updates normalized to [0, 1] scale
        return min(self.update_frequency / 7.0, 1.0)

class CostProfile(BaseModel):
    """Economic model for a substrate choice."""
    
    write_cost_usd: float = Field(
        description="One-time cost to encode knowledge (training, indexing, etc.)"
    )
    read_cost_per_query_usd: float = Field(
        description="Marginal cost per access (tokens, retrieval, compute)"
    )
    latency_ms: float = Field(
        description="p95 latency for knowledge access"
    )
    
    def total_cost(
        self, 
        queries: int, 
        latency_penalty_usd_per_ms: float = 0.0
    ) -> float:
        """Calculate total cost over expected lifetime."""
        return (
            self.write_cost_usd 
            + (queries * self.read_cost_per_query_usd)
            + (queries * self.latency_ms * latency_penalty_usd_per_ms)
        )

class OptimizationStrategy(BaseModel):
    """The routing decision with economic justification."""
    
    recommended_substrate: KnowledgeSubstrate
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cost_breakdown: dict[KnowledgeSubstrate, float]
    
    class Config:
        use_enum_values = True

# Cost constants (empirical baselines as of 2024)
WEIGHTS_WRITE_COST = 500.0  # Fine-tuning a 7B model (~$500 for 1M tokens)
CONTEXT_WRITE_COST = 0.0    # Prompt engineering is "free" (developer time excluded)
COMPUTE_WRITE_COST = 0.0    # No training required

# Per-query costs (USD)
WEIGHTS_READ_COST = 0.00001   # Marginal compute per query
WEIGHTS_HOSTING_USD_PER_DAY = 72.0  # One reserved H100 at $3/hour; this is most of the cost
CONTEXT_READ_COST_PER_1K = 0.0025  # $2.50/1M tokens (OpenAI input pricing)
COMPUTE_READ_COST_PER_1K = 0.015   # $15/1M tokens (o1 input pricing)

def route_knowledge(
    characteristics: KnowledgeCharacteristics,
    expected_lifetime_days: int = 365,
    latency_sla_ms: float = 1000.0,
    latency_penalty_usd_per_ms: float = 0.001
) -> OptimizationStrategy:
    """
    Determine optimal substrate using economic decision model.
    
    Decision Logic:
    1. Privacy constraints force CONTEXT or COMPUTE.
    2. High volatility prohibits WEIGHTS (training is too slow).
    3. Facts that must be recalled exactly go to CONTEXT.
    4. Reasoning requirements favor COMPUTE.
    5. Otherwise, minimize total cost over expected lifetime, hosting included.
    """
    
    total_queries = characteristics.access_frequency * expected_lifetime_days
    
    # Calculate cost profiles for each substrate
    cost_weights = CostProfile(
        write_cost_usd=WEIGHTS_WRITE_COST + WEIGHTS_HOSTING_USD_PER_DAY * expected_lifetime_days,
        read_cost_per_query_usd=WEIGHTS_READ_COST,
        latency_ms=50.0  # Single forward pass
    )
    
    cost_context = CostProfile(
        write_cost_usd=CONTEXT_WRITE_COST,
        read_cost_per_query_usd=(
            characteristics.average_size_tokens / 1000.0 * CONTEXT_READ_COST_PER_1K
        ),
        latency_ms=200.0  # Retrieval + prefill
    )
    
    cost_compute = CostProfile(
        write_cost_usd=COMPUTE_WRITE_COST,
        read_cost_per_query_usd=(
            characteristics.average_size_tokens / 1000.0 * COMPUTE_READ_COST_PER_1K * 5.0  # 5x token multiplier for reasoning
        ),
        latency_ms=15000.0  # 15 seconds for deep reasoning
    )
    
    # Rule-based gates
    if characteristics.is_user_specific:
        # Privacy constraint: WEIGHTS are shared across users
        if characteristics.requires_reasoning:
            return OptimizationStrategy(
                recommended_substrate=KnowledgeSubstrate.COMPUTE,
                confidence=0.95,
                reasoning="User-specific knowledge with reasoning requirements mandates COMPUTE substrate.",
                cost_breakdown={
                    KnowledgeSubstrate.WEIGHTS: float('inf'),
                    KnowledgeSubstrate.CONTEXT: cost_context.total_cost(total_queries, latency_penalty_usd_per_ms),
                    KnowledgeSubstrate.COMPUTE: cost_compute.total_cost(total_queries, latency_penalty_usd_per_ms)
                }
            )
        else:
            return OptimizationStrategy(
                recommended_substrate=KnowledgeSubstrate.CONTEXT,
                confidence=0.95,
                reasoning="User-specific knowledge without reasoning requirements mandates CONTEXT substrate (RAG).",
                cost_breakdown={
                    KnowledgeSubstrate.WEIGHTS: float('inf'),
                    KnowledgeSubstrate.CONTEXT: cost_context.total_cost(total_queries, latency_penalty_usd_per_ms),
                    KnowledgeSubstrate.COMPUTE: cost_compute.total_cost(total_queries, latency_penalty_usd_per_ms)
                }
            )
    
    if characteristics.volatility_score > 0.3:
        # High-churn knowledge: training cannot keep pace
        return OptimizationStrategy(
            recommended_substrate=KnowledgeSubstrate.CONTEXT,
            confidence=0.85,
            reasoning=f"Volatility score {characteristics.volatility_score:.2f} exceeds threshold (0.3). WEIGHTS training latency (days) cannot match update frequency.",
            cost_breakdown={
                KnowledgeSubstrate.WEIGHTS: float('inf'),
                KnowledgeSubstrate.CONTEXT: cost_context.total_cost(total_queries, latency_penalty_usd_per_ms),
                KnowledgeSubstrate.COMPUTE: cost_compute.total_cost(total_queries, latency_penalty_usd_per_ms)
            }
        )
    
    if characteristics.must_be_exact:
        # Exact-recall gate: fine-tuning stores facts unreliably, and a wrong code
        # is a billing error
        return OptimizationStrategy(
            recommended_substrate=KnowledgeSubstrate.CONTEXT,
            confidence=0.85,
            reasoning="Facts that must be recalled exactly belong in CONTEXT (a lookup you can check), whatever the cost minimum says.",
            cost_breakdown={
                KnowledgeSubstrate.WEIGHTS: cost_weights.total_cost(total_queries, latency_penalty_usd_per_ms),
                KnowledgeSubstrate.CONTEXT: cost_context.total_cost(total_queries, latency_penalty_usd_per_ms),
                KnowledgeSubstrate.COMPUTE: cost_compute.total_cost(total_queries, latency_penalty_usd_per_ms)
            }
        )
    
    if (
        characteristics.requires_reasoning
        and cost_compute.latency_ms <= latency_sla_ms
    ):
        # Reasoning gate: Use COMPUTE if latency budget allows
        return OptimizationStrategy(
            recommended_substrate=KnowledgeSubstrate.COMPUTE,
            confidence=0.80,
            reasoning="Domain requires multi-hop reasoning and latency SLA permits inference-time compute.",
            cost_breakdown={
                KnowledgeSubstrate.WEIGHTS: cost_weights.total_cost(total_queries, latency_penalty_usd_per_ms),
                KnowledgeSubstrate.CONTEXT: cost_context.total_cost(total_queries, latency_penalty_usd_per_ms),
                KnowledgeSubstrate.COMPUTE: cost_compute.total_cost(total_queries, latency_penalty_usd_per_ms)
            }
        )
    
    # Economic optimization: Choose substrate with minimum total cost
    costs = {
        KnowledgeSubstrate.WEIGHTS: cost_weights.total_cost(total_queries, latency_penalty_usd_per_ms),
        KnowledgeSubstrate.CONTEXT: cost_context.total_cost(total_queries, latency_penalty_usd_per_ms),
        KnowledgeSubstrate.COMPUTE: cost_compute.total_cost(total_queries, latency_penalty_usd_per_ms)
    }
    
    optimal = min(costs, key=costs.get)
    
    # Confidence based on cost gap
    sorted_costs = sorted(costs.values())
    cost_gap = (sorted_costs[1] - sorted_costs[0]) / sorted_costs[0]
    confidence = min(0.95, 0.6 + (cost_gap * 0.5))  # Higher gap = higher confidence
    
    return OptimizationStrategy(
        recommended_substrate=optimal,
        confidence=confidence,
        reasoning=f"Economic optimization over {expected_lifetime_days} days ({int(total_queries)} queries). Cost advantage: {cost_gap:.1%}.",
        cost_breakdown=costs
    )
# --8<-- [end:lst-15.1]

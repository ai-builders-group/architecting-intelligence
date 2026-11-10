# --8<-- [start:lst-9.40]
# src/08_retrieval_topology/strategy_selector.py
import math
from typing import Optional
import structlog

logger = structlog.get_logger()

async def select_context_strategy(
    corpus_size_tokens: int,
    corpus_volatility_per_hour: float,  # Updates/hour
    max_context_window: int,
    pricing: PricingTier,
    retrieval_cost: RetrievalCost,
    metrics_store: MetricsStore,
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
    max_daily_budget_usd: Optional[float] = None,
    top_k: int = 10,
    avg_doc_size_tokens: int = 2048,
    output_tokens: int = 512,
) -> ContextStrategy:
    """
    Determine the optimal context strategy with dynamic metrics and budget guardrails.
    
    Changes from Iteration 2:
    1. Uses metrics_store.get_queries_per_hour() instead of static parameter
    2. Models cache invalidation frequency (volatility)
    3. Enforces max_daily_budget with fallback to RAG
    4. Respects consistency_level (strong consistency disables stale-while-revalidate)
    5. Logs decision as structured metric for observability
    """
    
    # --- Dynamic Rate Estimation ---
    query_frequency_per_hour = await metrics_store.get_queries_per_hour()
    
    # --- Physical Constraint: Does the corpus fit in the context window? ---
    if corpus_size_tokens > max_context_window:
        strategy = ContextStrategy(
            strategy="rag_only",
            reasoning=f"Corpus ({corpus_size_tokens:,} tokens) exceeds max context ({max_context_window:,} tokens).",
            expected_cost_usd=_calculate_rag_cost(
                top_k, avg_doc_size_tokens, output_tokens, pricing, retrieval_cost
            ),
            expected_latency_ms=400,  # Retrieval (100ms) + Generation (300ms)
            consistency_level=consistency_level,
        )
        await _log_decision(strategy, query_frequency_per_hour)
        return strategy
    
    # --- Economic Constraint: Calculate cache reuse rate ---
    # Assume cache TTL of 30 minutes (conservative for most providers)
    cache_ttl_hours = 0.5
    queries_per_cache_cycle = query_frequency_per_hour * cache_ttl_hours
    
    # Account for cache invalidations due to corpus updates
    invalidations_per_cycle = corpus_volatility_per_hour * cache_ttl_hours
    effective_cache_reuse = max(
        1, queries_per_cache_cycle - invalidations_per_cycle)
    
    # --- Consistency Constraint: Strong consistency disables stale-while-revalidate ---
    if consistency_level == ConsistencyLevel.STRONG and corpus_volatility_per_hour > 0:
        # Strong consistency requires blocking on cache miss (no stale serving)
        # This makes full_context less attractive unless volatility is near-zero
        effective_cache_reuse *= 0.5  # Penalize due to blocking latency
    
    # --- Cost Calculation: RAG vs. Full Context ---
    cost_rag = _calculate_rag_cost(
        top_k, avg_doc_size_tokens, output_tokens, pricing, retrieval_cost)
    cost_full_context = _calculate_full_context_cost_with_invalidation(
        corpus_size_tokens, output_tokens, pricing,
        effective_cache_reuse, invalidations_per_cycle
    )
    
    # --- Budget Guardrail ---
    if max_daily_budget_usd is not None:
        projected_daily_cost_full = cost_full_context * query_frequency_per_hour * 24
        
        if projected_daily_cost_full > max_daily_budget_usd:
            strategy = ContextStrategy(
                strategy="rag_only",
                reasoning=f"Full context would breach daily budget: ${projected_daily_cost_full:.2f} > ${max_daily_budget_usd:.2f}. Falling back to RAG.",
                expected_cost_usd=cost_rag,
                expected_latency_ms=400,
                fallback_strategy="rag_only",
                consistency_level=consistency_level,
            )
            await _log_decision(strategy, query_frequency_per_hour, budget_breach=True)
            return strategy
    
    # --- Decision Logic ---
    if (
        cost_full_context < cost_rag * 0.7 and effective_cache_reuse > 50
    ):  # 30% margin + reuse threshold
        strategy = ContextStrategy(
            strategy="full_context",
            reasoning=f"Effective cache reuse ({effective_cache_reuse:.1f} queries/cycle) makes full context ${cost_full_context:.4f}/query vs RAG ${cost_rag:.4f}/query. Consistency: {consistency_level.value}.",
            expected_cost_usd=cost_full_context,
            expected_latency_ms=1200 if consistency_level == ConsistencyLevel.STRONG else 800,  # Blocking adds latency
            cache_reuse_assumption=int(effective_cache_reuse),
            consistency_level=consistency_level,
        )
    elif corpus_size_tokens > 50_000:  # Hybrid zone
        cost_hybrid = _calculate_hybrid_cost(
            top_k=50, avg_doc_size_tokens=avg_doc_size_tokens, output_tokens=output_tokens,
            pricing=pricing, retrieval_cost=retrieval_cost, cache_reuse_rate=min(10, effective_cache_reuse)
        )
        strategy = ContextStrategy(
            strategy="hybrid",
            reasoning=f"Corpus is large ({corpus_size_tokens:,} tokens). Hybrid (RAG→cache) costs ${cost_hybrid:.4f}/query.",
            expected_cost_usd=cost_hybrid,
            expected_latency_ms=1200,  # Retrieval (400ms) + Cached prefill (500ms) + Decode (300ms)
            cache_reuse_assumption=min(10, int(effective_cache_reuse)),
            consistency_level=consistency_level,
        )
    else:
        strategy = ContextStrategy(
            strategy="rag_only",
            reasoning=f"Low effective cache reuse ({effective_cache_reuse:.1f}) and/or high volatility favor RAG at ${cost_rag:.4f}/query.",
            expected_cost_usd=cost_rag,
            expected_latency_ms=400,
            consistency_level=consistency_level,
        )
    
    await _log_decision(strategy, query_frequency_per_hour)
    return strategy

# --- Helper Functions ---

def _calculate_rag_cost(
    top_k: int,
    avg_doc_size_tokens: int,
    output_tokens: int,
    pricing: PricingTier,
    retrieval_cost: RetrievalCost,
) -> float:
    """Cost for RAG: retrieval overhead + generation on top-k docs."""
    input_tokens = top_k * avg_doc_size_tokens
    generation_cost = (
        (input_tokens / 1_000_000) * pricing.input_cost_per_1m +
        (output_tokens / 1_000_000) * pricing.output_cost_per_1m
    )
    return (
        retrieval_cost.embedding_cost_per_query +
        retrieval_cost.vector_search_cost_per_query +
        retrieval_cost.reranking_cost_per_query +
        generation_cost
    )

def _calculate_full_context_cost_with_invalidation(
    corpus_size_tokens: int,
    output_tokens: int,
    pricing: PricingTier,
    effective_cache_reuse: float,
    invalidations_per_cycle: float,
) -> float:
    """
    Cost for full context with cache invalidation modeling.
    
    Formula:
    - First query (or after invalidation): Full cost (cache miss)
    - Subsequent queries: Cached cost (cache hit)
    - Amortize over effective reuse period
    """
    cache_miss_cost = (corpus_size_tokens / 1_000_000) * pricing.input_cost_per_1m
    cache_hit_cost = (corpus_size_tokens / 1_000_000) * pricing.cached_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m
    
    # Amortize: (1 miss + N hits) / (N + 1)
    # But add penalty for invalidations (extra misses)
    avg_input_cost = (
        (cache_miss_cost * (1 + invalidations_per_cycle)) +
        (cache_hit_cost * max(0, effective_cache_reuse - 1))
    ) / max(1, effective_cache_reuse)
    
    return avg_input_cost + output_cost

def _calculate_hybrid_cost(
    top_k: int,
    avg_doc_size_tokens: int,
    output_tokens: int,
    pricing: PricingTier,
    retrieval_cost: RetrievalCost,
    cache_reuse_rate: float,
) -> float:
    """Cost for hybrid: retrieval to filter, then cache the filtered set."""
    filtered_tokens = top_k * avg_doc_size_tokens
    
    retrieval_overhead = (
        retrieval_cost.embedding_cost_per_query +
        retrieval_cost.vector_search_cost_per_query +
        retrieval_cost.reranking_cost_per_query
    )
    
    cache_miss_cost = (filtered_tokens / 1_000_000) * pricing.input_cost_per_1m
    cache_hit_cost = (filtered_tokens / 1_000_000) * pricing.cached_cost_per_1m
    avg_cached_cost = (cache_miss_cost + (cache_reuse_rate - 1) * cache_hit_cost) / cache_reuse_rate
    
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m
    
    return retrieval_overhead + avg_cached_cost + output_cost

async def _log_decision(
    strategy: ContextStrategy,
    query_frequency: float,
    budget_breach: bool = False,
) -> None:
    """
    Log the strategy selection as a structured metric.
    
    This enables observability: CTOs can track strategy distribution
    and cost impact in Datadog/Grafana.
    """
    logger.info(
        "context_strategy_decision",
        strategy=strategy.strategy,
        expected_cost_usd=strategy.expected_cost_usd,
        expected_latency_ms=strategy.expected_latency_ms,
        cache_reuse=strategy.cache_reuse_assumption,
        consistency_level=strategy.consistency_level.value,
        query_frequency_per_hour=query_frequency,
        budget_breach=budget_breach,
        reasoning=strategy.reasoning,
    )
# --8<-- [end:lst-9.40]

# --8<-- [start:lst-9.42]
# src/08_retrieval_topology/cache_telemetry.py
from pydantic import BaseModel
from typing import Literal

class CacheInspectionResult(BaseModel):
    """Telemetry extracted from provider response headers."""
    status: Literal["hit", "miss", "partial", "unknown"]
    cached_tokens: int = 0
    uncached_tokens: int = 0
    eviction_detected: bool = False  # True if expected hit but got miss

async def inspect_cache_status(
    response_headers: dict[str, str],
    expected_cached_tokens: int,
) -> CacheInspectionResult:
    """
    Parse provider headers to detect unexpected cache evictions.
    
    If we expected a cache hit (based on our Merkle versioning) but got
    a miss, this indicates provider-side eviction due to memory pressure.
    """
    cache_status = response_headers.get("x-cache-status", "unknown").lower()
    cached_tokens = int(response_headers.get("x-cached-tokens", 0))
    uncached_tokens = int(response_headers.get("x-uncached-tokens", 0))
    
    eviction_detected = False
    
    if cache_status == "miss" and expected_cached_tokens > 0:
        # We expected a hit but got a miss: likely eviction
        eviction_detected = True
    
    elif (
        cache_status == "partial"
        and cached_tokens < expected_cached_tokens * 0.9
    ):
        # Partial hit with <90% of expected tokens: potential eviction
        eviction_detected = True
    
    return CacheInspectionResult(
        status=cache_status,
        cached_tokens=cached_tokens,
        uncached_tokens=uncached_tokens,
        eviction_detected=eviction_detected,
    )

async def handle_unexpected_eviction(
    cache_result: CacheInspectionResult,
    shard_id: str,
    retriever: HybridLongContextRetriever,
) -> None:
    """
    React to provider-side cache eviction.
    
    1. Log telemetry for cost analysis
    2. Invalidate our local cache entry (it's out of sync)
    3. Trigger proactive re-warming for high-traffic shards
    """
    if cache_result.eviction_detected:
        logger.warning(
            "provider_cache_eviction_detected",
            shard_id=shard_id,
            expected_cached=cache_result.cached_tokens,
            actual_cached=cache_result.cached_tokens,
            uncached=cache_result.uncached_tokens,
        )
        
        # Invalidate our local cache (force refresh on next query)
        await retriever.invalidate_shard(shard_id)
# --8<-- [end:lst-9.42]

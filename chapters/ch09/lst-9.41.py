# --8<-- [start:lst-9.41]
# src/08_retrieval_topology/hybrid_long_context.py
from typing import Optional, Tuple
import hashlib
import asyncio
import uuid
from datetime import datetime, UTC, timedelta

class CacheEntry(BaseModel):
    """Metadata for a cached context."""
    content: str
    shard_id: str
    version_hash: str  # Merkle shard hash
    created_at: datetime
    is_stale: bool = False

class HybridLongContextRetriever:
    """
    A retriever that uses RAG to filter, then caches the filtered set with
    stale-while-revalidate to prevent thundering herd.
    
    CRITICAL CHANGE (Iteration 3):
    - Uses DistributedLock (Redis) instead of asyncio.Lock for multi-pod coordination
    - Supports ConsistencyLevel to choose between eventual (stale) and strong (blocking)
    """
    
    def __init__(
        self,
        base_retriever: "DenseRetriever",
        distributed_lock: DistributedLock,  # CHANGED: Redis-backed lock
        cache_ttl_seconds: int = 3600,
        stale_grace_period_seconds: int = 60,
    ):
        self.base_retriever = base_retriever
        self.distributed_lock = distributed_lock  # Shared across all pods
        self.cache: dict[str, CacheEntry] = {}
        self.cache_ttl = cache_ttl_seconds
        self.stale_grace_period = stale_grace_period_seconds
    
    async def retrieve_with_cache(
        self,
        query: str,
        merkle_version: MerkleCorpusVersion,
        shard_id: str,
        tenant_id: str,
        acl_fp: str,
        top_k: int = 50,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
    ) -> Tuple[str, bool]:
        """
        Retrieve top-k documents, concatenate, and cache the result.
        
        Implements stale-while-revalidate (EVENTUAL) or blocking refresh (STRONG).
        
        Returns:
            (context_string, was_cached)
        """
        cache_key = self._cache_key(shard_id, top_k, tenant_id, acl_fp)
        shard_hash = merkle_version.shard_hashes.get(shard_id)
        
        # Check cache
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            age = (datetime.now(UTC) - entry.created_at).total_seconds()
            
            # Fresh cache hit
            if not entry.is_stale and entry.version_hash == shard_hash and age < self.cache_ttl:
                return entry.content, True
            
            # Stale cache hit
            if entry.is_stale and age < (self.cache_ttl + self.stale_grace_period):
                if consistency_level == ConsistencyLevel.EVENTUAL:
                    # Serve stale + trigger background refresh (non-blocking)
                    asyncio.create_task(
                        self._refresh_cache(
                            query, shard_id, shard_hash, top_k, tenant_id, acl_fp
                        )
                    )
                    return entry.content, True  # Return stale (marked in logs)
                else:
                    # STRONG consistency: Block until fresh
                    return await self._refresh_cache(
                        query, shard_id, shard_hash, top_k, tenant_id, acl_fp
                    )
        
        # Cache miss or expired: perform synchronous refresh
        return await self._refresh_cache(
            query, shard_id, shard_hash, top_k, tenant_id, acl_fp
        )
    
    async def _refresh_cache(
        self,

        query: str,
        shard_id: str,
        shard_hash: str,
        top_k: int,
        tenant_id: str,
        acl_fp: str,
    ) -> Tuple[str, bool]:
        """
        Refresh the cache with thundering herd protection.
        
        CRITICAL CHANGE (Iteration 3):
        Uses a DISTRIBUTED lock (Redis) to ensure only ONE worker across
        ALL pods performs the refresh. Other concurrent requests wait for
        the lock and receive the fresh result.
        """
        scope = f"{tenant_id}:{acl_fp}"  # tenant plus ACL fingerprint
        cache_key = f"{scope}:{shard_id}:{top_k}"
        lock_key = f"cache_refresh:{cache_key}"
        lock_token = uuid.uuid4(
            ).hex  # Owner token: only this call may release the lock
        
        # Try to acquire distributed lock (30-second TTL for long prefill
        # operations)
        acquired = await self.distributed_lock.acquire(
            lock_key, token=lock_token, ttl=timedelta(seconds=30)
        )
        
        if acquired:
            try:
                # We hold the lock: perform the refresh
                
                # Double-check: another worker may have refreshed while we waited
                # for lock
                if cache_key in self.cache:
                    entry = self.cache[cache_key]
                    if entry.version_hash == shard_hash and not entry.is_stale:
                        return entry.content, True
                
                # Perform retrieval
                context_obj = await self.base_retriever.retrieve_with_parent_expansion(
                    query=query,
                    top_k=top_k,
                )
                
                # Concatenate documents
                context_string = "\n\n---\n\n".join([
                    f"[Document {i+1}]\n{doc.content}"
                    for i, doc in enumerate(context_obj.documents)
                ])
                
                # Cache the result
                self.cache[cache_key] = CacheEntry(
                    content=context_string,
                    shard_id=shard_id,
                    version_hash=shard_hash,
                    created_at=datetime.now(UTC),
                    is_stale=False,
                )
                
                return context_string, False
            
            finally:
                # Always release the lock (only if our token still owns it)
                await self.distributed_lock.release(lock_key, token=lock_token)
        
        else:
            # Another worker holds the lock: wait for it to finish
            # Exponential backoff: 100ms, 200ms, 400ms, 800ms
            for attempt in range(4):
                await asyncio.sleep(0.1 * (2 ** attempt))
                
                # Check if cache was updated
                if cache_key in self.cache:
                    entry = self.cache[cache_key]
                    if entry.version_hash == shard_hash and not entry.is_stale:
                        return entry.content, True
            
            # Timeout: fallback to uncached retrieval
            context_obj = await self.base_retriever.retrieve_with_parent_expansion(
                query=query,
                top_k=top_k,
            )
            
            context_string = "\n\n---\n\n".join([
                f"[Document {i+1}]\n{doc.content}"
                for i, doc in enumerate(context_obj.documents)
            ])
            
            return context_string, False
    
    async def invalidate_shard(self, shard_id: str) -> None:
        """
        Mark all cache entries for a shard as stale (soft invalidation).
        
        Entries remain in cache for stale_grace_period to serve stale-while-revalidate.
        """
        for key, entry in self.cache.items():
            if entry.shard_id == shard_id:
                entry.is_stale = True
    
    def _cache_key(
        self, shard_id: str, top_k: int, tenant_id: str, acl_fp: str
    ) -> str:
        """Generate a deterministic, tenant- and ACL-scoped cache key."""
        return f"{tenant_id}:{acl_fp}:{shard_id}:{top_k}"
# --8<-- [end:lst-9.41]

# --8<-- [start:lst-11.13]
# src/08_retrieval_topology/hot_buffer.py
from __future__ import annotations
import faiss
import numpy as np
from datetime import datetime, UTC, timedelta
from collections import deque
from typing import Deque
import threading
import hashlib
import json

class HotBuffer:
    """In-memory vector index with automatic LRU eviction."""
    
    def __init__(
        self,
        dimension: int = 1536,
        max_size: int = 100_000,
        ttl_hours: int = 24,
    ):
        self.dimension = dimension
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        
        # HNSW index (Hierarchical Navigable Small World)
        self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
        
        # Metadata storage (parallel to index)
        self.events: Deque[EventSignal] = deque(maxlen=max_size)
        
        # Deduplication cache (content hash -> event_id)
        self.seen_hashes: dict[str, str] = {}
        
        # Thread safety
        self.lock = threading.RLock()
    
    def add(self, event: EventSignal) -> bool:
        """
        Add a new event to the hot buffer.
        
        Returns:
            True if added, False if duplicate (idempotent)
        """
        if event.embedding is None:
            raise ValueError("Event must have a pre-computed embedding for hot buffer insertion.")
        
        with self.lock:
            # Deduplication check
            content_hash = event.content_hash()
            if content_hash in self.seen_hashes:
                return False  # Duplicate, skip
            
            # Evict stale events (older than TTL)
            self._evict_stale()
            
            # Add to HNSW index
            vector = np.array([event.embedding], dtype=np.float32)
            self.index.add(vector)
            
            # Store metadata
            self.events.append(event)
            self.seen_hashes[content_hash] = event.event_id
            
            return True
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        time_window_minutes: int | None = None,
    ) -> list[tuple[EventSignal, float]]:
        """Retrieve top-k events by similarity, optionally filtered by time."""
        with self.lock:
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # HNSW search (returns distances and indices)
            distances, indices = self.index.search(query_vector, min(top_k * 2, len(self.events)))
            
            # Post-filter by time window if specified
            results = []
            cutoff_time = (
                datetime.now(UTC) - timedelta(minutes=time_window_minutes)
                if time_window_minutes
                else None
            )
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Faiss returns -1 for empty slots
                    continue
                
                event = self.events[idx]
                
                # Time filter
                if cutoff_time and event.timestamp < cutoff_time:
                    continue
                
                # Convert L2 distance to similarity score (inverse)
                similarity = 1 / (1 + dist)
                results.append((event, similarity))
                
                if len(results) >= top_k:
                    break
            
            return results
    
    def _evict_stale(self) -> None:
        """Remove events older than TTL."""
        cutoff = datetime.now(UTC) - self.ttl
        
        while self.events and self.events[0].timestamp < cutoff:
            evicted = self.events.popleft()
            # Remove from dedup cache
            content_hash = evicted.content_hash()
            self.seen_hashes.pop(content_hash, None)
            # Note: Faiss does not support deletion; we rely on max_size eviction
# --8<-- [end:lst-11.13]

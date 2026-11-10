# --8<-- [start:lst-9.39]
# src/08_retrieval_topology/metrics_store.py
from collections import deque
from datetime import datetime, UTC, timedelta
from typing import Deque
import asyncio

class MetricsStore:
    """
    Thread-safe in-memory store for trailing metrics.
    
    Uses a ring buffer (deque) to maintain the last N events.
    """
    
    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.query_timestamps: Deque[datetime] = deque(maxlen=10000)
        self._lock = asyncio.Lock()
    
    async def record_query(self) -> None:
        """Log a query event (non-blocking)."""
        async with self._lock:
            self.query_timestamps.append(datetime.now(UTC))
    
    async def get_queries_per_hour(self) -> float:
        """
        Calculate trailing queries/hour using a sliding window.
        
        Returns 0 if no queries in the window (prevents division by zero).
        """
        async with self._lock:
            cutoff = datetime.now(UTC) - timedelta(minutes=self.window_minutes)
            
            # Remove stale entries (outside window)
            while self.query_timestamps and self.query_timestamps[0] < cutoff:
                self.query_timestamps.popleft()
            
            count = len(self.query_timestamps)
            
            if count == 0:
                return 0.0
            
            # Extrapolate to queries/hour
            return (count / self.window_minutes) * 60
# --8<-- [end:lst-9.39]

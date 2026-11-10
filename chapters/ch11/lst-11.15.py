# --8<-- [start:lst-11.15]
# src/08_retrieval_topology/stream_retriever.py
import asyncio
import asyncpg
from typing import Protocol
from datetime import datetime, UTC, timedelta
import numpy as np

class StreamRetriever:
    """Unified retrieval across hot (real-time) and cold (historical) layers."""
    
    def __init__(
        self,
        hot_buffer: HotBuffer,
        cold_pool: asyncpg.Pool,
        alpha: float = 0.6,  # Semantic vs. temporal weight
        decay_lambda: float = 0.01,  # Exponential decay rate (per minute)
    ):
        self.hot = hot_buffer
        self.cold = cold_pool
        self.alpha = alpha
        self.decay_lambda = decay_lambda
    
    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        time_window_minutes: int = 60,
    ) -> list[tuple[EventSignal | SignalBatch, float]]:
        """Parallel retrieval with time-weighted fusion."""
        
        # Parallel fetch
        hot_task = asyncio.to_thread(
            self.hot.search, query_embedding, top_k=top_k,
            time_window_minutes=time_window_minutes
        )
        cold_task = self._fetch_cold(query_embedding, top_k=top_k,
                                     time_window_minutes=time_window_minutes)
        
        hot_results, cold_results = await asyncio.gather(hot_task, cold_task)
        
        # Compute time-weighted scores
        now = datetime.now(UTC)
        scored_results = []
        
        for item, sim_score in hot_results + cold_results:
            timestamp = (item.timestamp if isinstance(item, EventSignal)
                         else item.end_time)
            time_delta_minutes = (now - timestamp).total_seconds() / 60
            
            # Exponential decay
            time_score = np.exp(-self.decay_lambda * time_delta_minutes)
            
            # Weighted blend
            final_score = self.alpha * sim_score + (1 - self.alpha) * time_score
            
            scored_results.append((item, final_score))
        
        # Sort and deduplicate
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
    
    async def _fetch_cold(
        self,
        query_embedding: list[float],
        top_k: int,
        time_window_minutes: int,
    ) -> list[tuple[SignalBatch, float]]:
        """Retrieve from cold archive (PostgreSQL)."""
        cutoff = datetime.now(UTC) - timedelta(minutes=time_window_minutes)
        
        async with self.cold.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    batch_id, start_time, end_time, event_count, summary, 
                    representative_embedding, sample_events,
                    1 - (representative_embedding <=> $1::vector) AS similarity
                FROM signal_batches
                WHERE end_time >= $2
                ORDER BY similarity DESC
                LIMIT $3
                """,
                query_embedding,
                cutoff,
                top_k,
            )
        
        return [
            (
                SignalBatch(
                    batch_id=row["batch_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    event_count=row["event_count"],
                    summary=row["summary"],
                    representative_embedding=row["representative_embedding"],
                    sample_events=row["sample_events"],
                ),
                row["similarity"],
            )
            for row in rows
        ]
# --8<-- [end:lst-11.15]

# --8<-- [start:lst-11.14]
# src/08_retrieval_topology/batch_aggregator.py
from typing import Protocol
from datetime import datetime, UTC
import numpy as np

class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str: ...

async def aggregate_events(
    events: list[EventSignal],
    llm: LLMClient,
    batch_window_minutes: int = 60,
) -> SignalBatch:
    """Compress N events into a single summarized batch."""
    
    # Sort by timestamp
    events = sorted(events, key=lambda e: e.timestamp)
    
    # Generate summary
    event_texts = [e.content for e in events[:50]]  # Cap at 50 for prompt size
    prompt = f"""Summarize the following {len(events)} log events into a \
single insight.\nFocus on patterns, anomalies, and actionable information.

Events:
{chr(10).join(event_texts)}

Summary (max 200 words):"""
    
    summary = await llm.generate(prompt)
    
    # Compute centroid embedding (average of all event embeddings)
    embeddings = np.array([e.embedding for e in events if e.embedding])
    centroid = embeddings.mean(axis=0).tolist() if len(embeddings) else []
    
    return SignalBatch(
        batch_id=f"batch_{events[0].event_id}",
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
        event_count=len(events),
        summary=summary,
        representative_embedding=centroid,
        sample_events=[e.event_id for e in events[:5]],
    )
# --8<-- [end:lst-11.14]

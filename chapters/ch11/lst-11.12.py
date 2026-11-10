# --8<-- [start:lst-11.12]
# src/08_retrieval_topology/signal_schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
import hashlib
import json

class EventSignal(BaseModel):
    """The atomic unit of a temporal event stream."""
    event_id: str = Field(description="UUIDv7 (time-sortable)")
    timestamp: datetime = Field(description="Event emission time (ISO 8601, UTC)")
    source: str = Field(
        description="Origin system (e.g., 'k8s-cluster-1', 'trade-feed')")
    event_type: Literal["log", "metric", "trace", "alert", "transaction"]
    content: str = Field(
        description="Raw event payload (text or JSON-serialized)")
    embedding: list[float] | None = Field(
        default=None, description="Optional pre-computed vector")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Filterable dimensions")
    severity: Literal["debug", "info", "warn", "error", "critical"] | None = None
    
    def content_hash(self) -> str:
        """
        Generate deterministic hash for deduplication.
        
        CRITICAL: Uses sort_keys=True to ensure identical objects
        produce identical hashes regardless of dict insertion order.
        """
        payload = {
            "source": self.source,
            "event_type": self.event_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

class SignalBatch(BaseModel):
    """A compressed representation of N similar events."""
    batch_id: str
    start_time: datetime
    end_time: datetime
    event_count: int
    summary: str = Field(description="LLM-generated abstract of the batch")
    representative_embedding: list[float] = Field(description="Centroid of batch vectors")
    sample_events: list[str] = Field(max_length=5, description="Example event IDs for drill-down")
# --8<-- [end:lst-11.12]

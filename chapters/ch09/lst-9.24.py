# --8<-- [start:lst-9.24]
# path/to/metrics.py
from pydantic import BaseModel
from datetime import datetime, UTC
from typing import Protocol

class MetricsBackend(Protocol):
    def increment(self, metric: str, value: float, tags: dict[str, str]) -> None: ...
    def gauge(self, metric: str, value: float, tags: dict[str, str]) -> None: ...

class ProcessorMetrics:
    """Observability wrapper for stream processor."""
    
    def __init__(self, processor: KappaStreamProcessor, backend: MetricsBackend):
        self.processor = processor
        self.metrics = backend
    
    def process_event_with_metrics(self, event: IngestionEvent) -> None:
        """Process event and emit metrics."""
        start = datetime.now(UTC)
        
        try:
            self.processor.process_event(event)
            
            # Calculate lag
            lag_seconds = (start - event.timestamp).total_seconds()
            self.metrics.gauge("event_lag_seconds", lag_seconds, {
                "tenant_id": event.tenant_id,
                "event_type": event.event_type
            })
            
            # Track success
            self.metrics.increment("events_processed", 1.0, {
                "tenant_id": event.tenant_id,
                "status": "success"
            })
        except Exception as e:
            self.metrics.increment("events_processed", 1.0, {
                "tenant_id": event.tenant_id,
                "status": "failed",
                "error": type(e).__name__
            })
            raise
# --8<-- [end:lst-9.24]

# --8<-- [start:lst-9.21]
# path/to/kappa_events.py
from pydantic import BaseModel, Field
from datetime import datetime, UTC
from typing import Literal
from enum import StrEnum

class EventType(StrEnum):
    DOCUMENT_ADDED = "document.added"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    EMBEDDING_GENERATED = "embedding.generated"

class IngestionEvent(BaseModel):
    """Immutable event log entry for document lifecycle."""
    
    event_id: str = Field(description="UUID for idempotency")
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    # Document Identity
    document_id: str
    document_uri: str
    
    # Payload (event-specific)
    content: str | None = None  # Full text for ADDED/UPDATED
    metadata: dict[str, str | int | float] = Field(default_factory=dict)
    
    # Provenance
    source_system: str = Field(description="api | batch_upload | webhook")
    tenant_id: str = Field(description="Multi-tenancy isolation")
    
    # Processing Fingerprint
    schema_version: int = Field(default=1, description="Event schema version")
    
    class Config:
        frozen = True  # Immutability enforcement
# --8<-- [end:lst-9.21]

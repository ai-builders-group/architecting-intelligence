# --8<-- [start:lst-6.1]
# architecting-intelligence/src/06_durable/checkpoint_with_offload.py
from pydantic import BaseModel, Field
from typing import Optional
import hashlib
import json

class EventPayload(BaseModel):
    """Compact event representation with external blob reference."""
    event_id: str
    event_type: str  # "activity_completed", "llm_invoked"
    blob_uri: Optional[str] = Field(
        None, 
        description="S3 URI for large payloads (>10KB)"
    )
    inline_data: Optional[dict] = Field(
        None,
        description="Small payloads (<10KB) stored inline"
    )
    checksum: str = Field(
        description="SHA256 hash for integrity verification"
    )

def store_event(tenant_id: str, workflow_id: str, payload: dict) -> EventPayload:
    """Store event with automatic blob offloading."""
    data = json.dumps(payload, sort_keys=True).encode('utf-8')
    payload_size = len(data)
    
    if payload_size > 10_000:  # illustrative threshold
        checksum = hashlib.sha256(
            data
        ).hexdigest()
        key = f"{tenant_id}/{workflow_id}/{checksum}"
        blob_uri = upload_to_blob_storage(key, data)
        
        return EventPayload(
            event_id=generate_event_id(),
            event_type="llm_invoked",
            blob_uri=blob_uri,
            checksum=checksum
        )
    else:
        return EventPayload(
            event_id=generate_event_id(),
            event_type="llm_invoked",
            inline_data=payload,
            checksum=hashlib.sha256(
                data
            ).hexdigest()
        )
# --8<-- [end:lst-6.1]

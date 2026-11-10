# --8<-- [start:lst-9.46]
# chapter_07/memory_trace.py
from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime, UTC

class MemorySourceTrace(BaseModel):
    """Provenance for a chunk of context."""
    source_layer: Literal["L1_working", "L2_summary", "L3_archived"]
    source_id: str  # Message ID or summary ID
    token_range: tuple[int, int]  # Start and end positions in assembled context
    confidence: float = Field(ge=0.0, le=1.0)
    retrieval_score: float | None = None  # If from L3
    is_pinned: bool = Field(default=False)  # Eviction immunity status
    
class MemoryTraceMap(BaseModel):
    """
    Complete memory map for an inference request.
    
    Stores the provenance of every token in the assembled context,
    enabling post-mortem debugging of memory failures.
    """
    request_id: str
    conversation_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    # Token sources
    sources: List[MemorySourceTrace] = Field(default_factory=list)
    
    # Statistics
    total_tokens: int = 0
    l1_tokens: int = 0
    l2_tokens: int = 0
    l3_tokens: int = 0
    pinned_tokens: int = 0
    
    # Eviction audit
    evicted_message_ids: List[str] = Field(default_factory=list)
    eviction_reason: str | None = None
    
    # Retrieval audit
    l3_query_embedding: List[float] | None = None
    l3_candidates_count: int = 0
    l3_returned_count: int = 0
    
    # Pinning audit
    pinned_message_ids: List[str] = Field(default_factory=list)
    pin_prevented_evictions: int = 0
    
    def add_source(self, source: MemorySourceTrace):
        """Add a source trace and update statistics."""
        self.sources.append(source)
        token_count = source.token_range[1] - source.token_range[0]
        
        if source.source_layer == "L1_working":
            self.l1_tokens += token_count
        elif source.source_layer == "L2_summary":
            self.l2_tokens += token_count
        elif source.source_layer == "L3_archived":
            self.l3_tokens += token_count
        
        if source.is_pinned:
            self.pinned_tokens += token_count
    
    def get_layer_utilization(self) -> dict[str, float]:
        """Calculate percentage utilization by layer."""
        if self.total_tokens == 0:
            return {"L1": 0.0, "L2": 0.0, "L3": 0.0, "Pinned": 0.0}
        
        return {
            "L1": self.l1_tokens / self.total_tokens,
            "L2": self.l2_tokens / self.total_tokens,
            "L3": self.l3_tokens / self.total_tokens,
            "Pinned": self.pinned_tokens / self.total_tokens
        }

class MemoryTraceCollector:
    """
    Collects memory provenance during context assembly.
    
    Integrates with SemanticLRU, Summarizer, and VirtualMemory
    to track every token's origin and pinning status.
    """
    
    def __init__(self, request_id: str, conversation_id: str):
        self.trace = MemoryTraceMap(
            request_id=request_id,
            conversation_id=conversation_id
        )
        self.current_position = 0
    
    def add_working_memory(self, messages: List[Message]):
        """Track L1 working memory sources."""
        for msg in messages:
            end_pos = self.current_position + msg.token_count
            self.trace.add_source(MemorySourceTrace(
                source_layer="L1_working",
                source_id=msg.id,
                token_range=(self.current_position, end_pos),
                confidence=1.0,
                is_pinned=msg.is_pinned
            ))
            self.current_position = end_pos
            
            if msg.is_pinned:
                self.trace.pinned_message_ids.append(msg.id)
    
    def add_summary(self, summary_id: str, token_count: int, confidence: float):
        """Track L2 summary sources."""
        end_pos = self.current_position + token_count
        self.trace.add_source(MemorySourceTrace(
            source_layer="L2_summary",
            source_id=summary_id,
            token_range=(self.current_position, end_pos),
            confidence=confidence
        ))
        self.current_position = end_pos
    
    def add_archived_retrieval(self, envelope: TemporalEnvelope):
        """Track L3 archived retrieval sources."""
        end_pos = self.current_position + envelope.total_tokens
        self.trace.add_source(MemorySourceTrace(
            source_layer="L3_archived",
            source_id=envelope.anchor_message.id,
            token_range=(self.current_position, end_pos),
            confidence=envelope.confidence_score,
            retrieval_score=envelope.confidence_score,
            is_pinned=envelope.is_pinned
        ))
        self.current_position = end_pos
        
        if envelope.is_pinned:
            self.trace.pinned_message_ids.append(envelope.anchor_message.id)
    
    def record_eviction(
        self, evicted: List[Message], reason: str, pin_prevented: int = 0):
        """Record eviction event for debugging."""
        self.trace.evicted_message_ids = [msg.id for msg in evicted]
        self.trace.eviction_reason = reason
        self.trace.pin_prevented_evictions = pin_prevented
    
    def finalize(self) -> MemoryTraceMap:
        """Finalize the trace and return immutable map."""
        self.trace.total_tokens = self.current_position
        return self.trace
# --8<-- [end:lst-9.46]

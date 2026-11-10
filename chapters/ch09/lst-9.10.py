# --8<-- [start:lst-9.10]
# path/to/chunking_strategy.py
from enum import Enum
from pydantic import BaseModel

class ChunkingStrategy(str, Enum):
    CHARACTER = "character"
    RECURSIVE_SEMANTIC = "recursive_semantic"
    EMBEDDING_BASED = "embedding_based"
    CONTEXTUAL_PREPEND = "contextual_prepend"
    LATE_CHUNKING = "late_chunking"  # Local weights only

class DocumentProfile(BaseModel):
    """Document characteristics that determine optimal chunking strategy."""
    avg_token_count: int
    has_cross_references: bool
    domain: str  # "legal" | "technical" | "conversational" | "numerical"
    importance: str  # "critical" | "standard" | "ephemeral"
    late_chunking_available: bool = False  # Self-hosted weights or a hosted late-chunking endpoint?

def select_chunking_strategy(profile: DocumentProfile) -> ChunkingStrategy:
    """Decision tree for chunking strategy selection."""
    
    # Never use Late Chunking for numerical/tabular data (hallucination risk)
    if profile.domain == "numerical":
        return ChunkingStrategy.RECURSIVE_SEMANTIC  # Fallback to deterministic
    
    # Critical documents with cross-references: Preserve context
    if profile.importance == "critical" and profile.has_cross_references:
        if profile.avg_token_count < 8192:
            # Late chunking, self-hosted or hosted, if available
            if profile.late_chunking_available:
                return ChunkingStrategy.LATE_CHUNKING
            else:
                # No late chunking available: Contextual Prepend as proxy
                return ChunkingStrategy.CONTEXTUAL_PREPEND
        else:
            # Document too large for Late Chunking VRAM limits
            return ChunkingStrategy.EMBEDDING_BASED
    
    # Technical/Legal: Respect semantic boundaries
    if profile.domain in ["legal", "technical"]:
        return ChunkingStrategy.EMBEDDING_BASED
    
    # Short conversational: Fast and simple
    if profile.avg_token_count < 1000:
        return ChunkingStrategy.RECURSIVE_SEMANTIC
    
    # Default: Balanced approach
    return ChunkingStrategy.RECURSIVE_SEMANTIC
# --8<-- [end:lst-9.10]

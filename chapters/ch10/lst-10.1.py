# --8<-- [start:lst-10.1]
# src/08_retrieval_topology/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ChildChunk(BaseModel):
    """The indexed unit: optimized for retrieval precision."""
    chunk_id: str = Field(description="Unique identifier (UUIDv7)")
    parent_id: str = Field(description="Foreign key to ParentDocument")
    content: str = Field(description="128-512 tokens of text")
    embedding: list[float] = Field(description="Dense vector (e.g., 1536-dim)")
    position: int = Field(
        description="Ordinal position within parent (0-indexed)")
    metadata: dict[
        str, str] = Field(default_factory=dict, description="Filterable tags")

class ParentDocument(BaseModel):
    """The retrieved unit: optimized for generation context."""
    document_id: str
    content: str = Field(description="Full text (1024-2048 tokens)")
    summary: Optional[
        str] = Field(default=None, description="LLM-generated abstract")
    source_uri: str = Field(description="Provenance (e.g., s3://bucket/key)")
    ingested_at: datetime = Field(description="ISO 8601 timestamp")
    chunk_ids: list[str] = Field(description="Child chunk references")

class RetrievalContext(BaseModel):
    """The assembled input for the generator."""
    query: str
    documents: list[ParentDocument]
    total_tokens: int = Field(description="Sum of document token counts")
    retrieval_latency_ms: int
    strategy: str = Field(description="e.g., 'hybrid_rrf', 'parent_child'")
# --8<-- [end:lst-10.1]

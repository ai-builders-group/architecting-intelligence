# --8<-- [start:lst-10.14]
# src/08_retrieval_topology/schemas.py (extension)
from pydantic import BaseModel, Field
from enum import Enum

class RetrievalSource(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

class RetrievalConfig(BaseModel):
    """Versioned configuration for retrieval thresholds."""
    config_version: str = Field(description="e.g., 'v2024.02.06'")
    dense_min_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for dense retrieval"
    )
    sparse_min_score: float = Field(
        default=0.05, ge=0.0, description="Minimum full-text rank")
    dense_timeout_ms: int = Field(default=500, ge=100, le=5000)
    sparse_timeout_ms: int = Field(default=200, ge=50, le=2000)
    rrf_k: int = Field(default=60, ge=10, le=200)
    enable_score_gating: bool = Field(default=True)
    enable_fallback: bool = Field(
        default=True,
        description="Allow sparse-only results if dense fails"
    )

class RankedResult(BaseModel):
    """A document with its rank and score from a single retrieval system."""
    chunk_id: str
    rank: int = Field(ge=1, description="1-indexed position in result list")
    score: float = Field(description="Raw score (cosine or BM25)")
    source: RetrievalSource
    parent_id: str | None = None

class FusedResult(BaseModel):
    """A document after RRF fusion."""
    chunk_id: str
    rrf_score: float = Field(description="Reciprocal Rank Fusion score")
    dense_rank: int | None = Field(default=None, description="Rank in dense retrieval")
    sparse_rank: int | None = Field(default=None, description="Rank in sparse retrieval")
    dense_score: float | None = Field(default=None, description="Raw cosine score")
    sparse_score: float | None = Field(default=None, description="Raw BM25 score")
    parent_id: str
    is_partial: bool = Field(
        default=False,
        description="True if result comes from only one retriever (fallback)"
    )
# --8<-- [end:lst-10.14]

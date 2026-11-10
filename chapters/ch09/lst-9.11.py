# --8<-- [start:lst-9.11]
# path/to/chunk_metadata.py
from pydantic import BaseModel, Field
from datetime import datetime, UTC
from typing import Optional
import importlib.metadata

class ChunkMetadata(BaseModel):
    """Immutable provenance with versioning for zero-downtime migrations."""
    
    chunk_id: str
    source_uri: str
    
    # Versioning: Critical for model migrations
    chunking_strategy: str = Field(description="recursive_semantic | embedding_based | contextual_prepend | late_chunking")
    chunking_version: str = Field(description="v1.2.3 - Semantic versioning")
    embedding_model: str = Field(description="text-embedding-3-large")
    embedding_dim: int = 3072
    context_summary_model: Optional[str] = Field(default=None, description="gpt-4o (for Contextual Prepend)")
    
    # Parser Versioning (prevents Blue/Green drift)
    parser_lib: str = Field(description="pdfplumber | pymupdf | unstructured")
    parser_lib_version: str = Field(
        description="Library version for boundary parity")
    
    # Temporal versioning
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    # Allows strategy-specific invalidation during upgrades
    def is_compatible_with(
        self, 
        target_embedding: str, 
        target_strategy: str,
        target_parser_version: str
    ) -> bool:
        """Check if chunk can be reused or must be re-indexed.
        
        Enforces parser version parity to prevent boundary drift.
        """
        return (
            self.embedding_model == target_embedding and
            self.chunking_strategy == target_strategy and
            self.parser_lib_version == target_parser_version != "unknown"
        )
    
    @classmethod
    def from_current_environment(cls, chunk_id: str, source_uri: str, **kwargs):
        """Auto-populate parser version from installed library."""
        parser_lib = kwargs.pop("parser_lib", "pdfplumber")
        try:
            parser_version = importlib.metadata.version(parser_lib)
        except importlib.metadata.PackageNotFoundError:
            parser_version = "unknown"
        
        return cls(
            chunk_id=chunk_id,
            source_uri=source_uri,
            parser_lib=parser_lib,
            parser_lib_version=parser_version,
            **kwargs
        )
# --8<-- [end:lst-9.11]

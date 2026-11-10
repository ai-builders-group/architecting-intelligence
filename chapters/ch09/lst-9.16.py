# --8<-- [start:lst-9.16]
# path/to/metadata_schema.py
from pydantic import BaseModel, Field
from datetime import datetime, UTC
from typing import Optional, Literal

class MetadataSchemaVersion(BaseModel):
    """
    Schema versioning header to enable zero-downtime metadata evolution.
    """
    version: str = Field(description="Semantic version (e.g., '2.1.0')")
    effective_date: datetime
    deprecated_date: Optional[datetime] = None
    migration_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Maps old enum values to new ones"
    )

# Current schema (v2.0.0)
class DepartmentV2(str):
    """
    Free-form department string with canonical mappings.
    Replaces strict enum to enable organizational evolution.
    """
    pass

class SourceType(str):
    """Fixed taxonomy - stable across schema versions."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    API = "api"
    EMAIL = "email"

class FilterableMetadata(BaseModel):
    """
    Structured metadata optimized for pre-retrieval filtering.
    Includes schema version for migration compatibility.
    """
    # Schema versioning
    schema_version: str = "2.0.0"
    
    # Identifiers
    chunk_id: str = Field(description="SHA256 hash of content")
    document_id: str = Field(description="Parent document UUID")
    
    # Temporal Filters
    document_date: datetime = Field(description="When the source was authored")
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_modified: Optional[datetime] = None
    
    # Structural Filters
    source_type: SourceType
    source_uri: str
    section_path: list[str] = Field(
        default_factory=list,
        description="Hierarchical path: ['Chapter 3', 'Section 3.2']"
    )
    page_number: Optional[int] = None
    
    # Authority Filters (with canonical mapping support)
    department: str = Field(
        description="Department name (free-form with canonical mapping)")
    department_canonical: Optional[str] = Field(
        default=None,
        description="Mapped to current taxonomy before the filter runs"
    )
    author: Optional[str] = None
    confidence_score: float = Field(
        ge=0.0, 
        le=1.0, 
        default=1.0,
        description="Extraction quality (OCR/VLM/LLM)"
    )
    
    # Embedding Fingerprint (for migration)
    embedding_model: str = "text-embedding-3-large"
    embedding_version: str = "2024-01"
    
    # Business Logic
    is_authoritative: bool = Field(
        default=False,
        description="Manually flagged as source of truth"
    )
    security_classification: Literal["public", "internal", "confidential", "restricted"] = "internal"

class DepartmentMapper:
    """
    Virtual mapping layer for department taxonomy evolution.
    Translates legacy department names to current taxonomy at query time.
    """
    def __init__(self):
        self.mappings = {
            "2.0.0": {
                # Direct mappings (current)
                "platform": "platform",
                "product": "product",
                "finance": "finance",
                "legal": "legal",
                "hr": "hr",
                "sales": "sales",
                "marketing": "marketing",
            },
            "1.0.0": {
                # Legacy mappings (v1 -> v2)
                "engineering": "platform",  # Default mapping
                "finance": "finance",
                "legal": "legal",
                "hr": "hr",
                "sales": "sales",
                "unknown": None,  # Unmapped
            }
        }
    
    def map_to_canonical(
        self, department: str, schema_version: str) -> Optional[str]:
        """
        Map a department value to current canonical taxonomy.
        Returns None if unmapped (triggers soft filter).
        """
        version_map = self.mappings.get(schema_version, self.mappings["2.0.0"])
        return version_map.get(department.lower())
    
    def apply_virtual_mapping(
        self, metadata: FilterableMetadata) -> FilterableMetadata:
        """
        Apply canonical mapping at query time (non-destructive).
        """
        canonical = self.map_to_canonical(
            metadata.department, metadata.schema_version)
        metadata.department_canonical = canonical
        return metadata
# --8<-- [end:lst-9.16]

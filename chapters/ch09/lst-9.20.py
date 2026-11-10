# --8<-- [start:lst-9.20]
# path/to/metadata_debugger.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, UTC

class MetadataAudit(BaseModel):
    """
    Immutable audit trail for every metadata extraction.
    Stored alongside the chunk in the vector DB.
    """
    chunk_id: str
    extraction_timestamp: datetime
    extractor_model: str
    extraction_confidence: float
    
    # Original input
    raw_text_sample: str = Field(description="First 500 chars of source text")
    source_page: Optional[int] = None
    
    # Extracted metadata (versioned)
    extracted_metadata: dict
    schema_version: str
    
    # Extraction reasoning (for debugging)
    extraction_reasoning: Optional[str] = None
    
    # Human corrections (if any)
    corrections: list[dict] = Field(default_factory=list)
    last_corrected_at: Optional[datetime] = None

class MetadataDebugger:
    """
    Tools for diagnosing and correcting metadata-induced retrieval failures.
    """
    def __init__(self, vector_db, audit_store):
        self.vector_db = vector_db
        self.audit_store = audit_store
    
    def bypass_search(
        self,
        query: str,
        caller: str,
        tenant_id: str,
        applied_filters: dict,
        bypass_metadata_filters: bool = True,
        min_confidence_override: Optional[float] = None
    ) -> list[str]:
        """
        Diagnostic search that bypasses potentially corrupted metadata filters.
        Used to find "shadow banned" documents.
        """
        # Drop extracted filters only; caller and tenant_id always go through
        results = self.vector_db.search(
            query=query,
            top_k=50,
            caller=caller,
            tenant_id=tenant_id,
            filters={} if bypass_metadata_filters else applied_filters
        )
        
        # If min_confidence_override, include low-confidence chunks
        if min_confidence_override is not None:
            results = [r for r in results if r.metadata.get("confidence_score", 1.0) >= min_confidence_override]
        
        return results
    
    def diagnose_null_result(
        self,
        query: str,
        expected_document_id: str,
        applied_filters: dict,
        caller: str,
        tenant_id: str
    ) -> dict:
        """
        Diagnose why a known-relevant document was not retrieved.
        Returns root cause analysis.
        """
        # Step 1: Fetch the document's metadata
        doc_metadata = self.vector_db.get_metadata(expected_document_id, tenant_id)
        
        if doc_metadata is None:
            return {"error": "Document not in index"}
        
        # Step 2: Fetch audit trail
        audit = self.audit_store.get(expected_document_id)
        
        # Step 3: Check which filter failed
        failed_filters = []
        for filter_field, filter_value in applied_filters.items():
            doc_value = doc_metadata.get(filter_field)
            
            if doc_value != filter_value:
                failed_filters.append({
                    "field": filter_field,
                    "expected": filter_value,
                    "actual": doc_value,
                    "extraction_confidence": audit.extraction_confidence if audit else None,
                    "extraction_reasoning": audit.extraction_reasoning if audit else None
                })
        
        # Step 4: Check semantic similarity (would it have been retrieved without filters?)
        bypass_rank = None
        bypass_results = self.bypass_search(
            query, caller, tenant_id, applied_filters,
            bypass_metadata_filters=True
        )
        for rank, result in enumerate(bypass_results, start=1):
            if result.chunk_id == expected_document_id:
                bypass_rank = rank
                break
        
        return {
            "diagnosis": "metadata_shadow_ban" if failed_filters else "semantic_mismatch",
            "failed_filters": failed_filters,
            "bypass_rank": bypass_rank,
            "extraction_model": audit.extractor_model if audit else None,
            "extraction_confidence": audit.extraction_confidence if audit else None,
            "correction_api_available": True
        }
    
    def submit_correction(
        self,
        chunk_id: str,
        tenant_id: str,
        corrected_metadata: dict,
        corrector_id: str
    ):
        """
        Human-in-the-loop correction of misclassified metadata.
        Triggers re-indexing and audit log update.
        """
        # Capture the metadata before it is overwritten
        old_metadata = self.vector_db.get_metadata(chunk_id, tenant_id)
        
        # Log correction in audit trail
        correction_entry = {
            "corrected_at": datetime.now(UTC),
            "corrector_id": corrector_id,
            "old_metadata": old_metadata,
            "new_metadata": corrected_metadata
        }
        
        self.audit_store.append_correction(chunk_id, correction_entry)
        
        # Update vector DB metadata
        self.vector_db.update_metadata(chunk_id, tenant_id, corrected_metadata)
        
        # Trigger re-indexing for this chunk
        self.vector_db.mark_for_reindex(chunk_id)
# --8<-- [end:lst-9.20]

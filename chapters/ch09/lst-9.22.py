# --8<-- [start:lst-9.22]
# path/to/stream_processor.py
from __future__ import annotations
from enum import StrEnum
from typing import Protocol
from openai import OpenAI
from pydantic import BaseModel, Field
import hashlib
from collections import Counter

class EventType(StrEnum):
    DOCUMENT_ADDED = "document.added"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    EMBEDDING_GENERATED = "embedding.generated"

class VectorStore(Protocol):
    """Abstract interface for vector storage backends."""
    def upsert(self, chunk_id: str, embedding: list[float], metadata: dict) -> None: ...
    def delete(self, chunk_id: str) -> None: ...
    def delete_by_filter(self, filters: dict) -> None: ...
    def delete_except(self, filters: dict, keep_ids: list[str]) -> None: ...

class PayloadReader(Protocol):
    """Reads a document's text from the event or its snapshot; raises if the snapshot is gone."""
    def read(self, event: IngestionEvent) -> str: ...

class KeyStore(Protocol):
    """Holds each document's encryption key; a destroyed key is crypto-shredding."""
    def is_destroyed(self, tenant_id: str, document_id: str) -> bool: ...

class KeyDestroyed(Exception):
    """The payload's key was destroyed or its snapshot is gone: the document is erased."""

class ProcessedChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, str | int | float]

class KappaStreamProcessor:
    """Idempotent event processor for RAG ingestion."""
    
    def __init__(self, vector_store: VectorStore, payloads: PayloadReader, keys: KeyStore,
                 embedding_model: str = "text-embedding-3-large"):
        self.client = OpenAI()
        self.vector_store = vector_store
        self.payloads = payloads
        self.keys = keys
        self.embedding_model = embedding_model
    
    def process_event(self, event: IngestionEvent) -> None:
        """Process a single event idempotently."""
        
        match event.event_type:
            case EventType.DOCUMENT_ADDED | EventType.DOCUMENT_UPDATED:
                self._handle_upsert(event)
            case EventType.DOCUMENT_DELETED:
                self._handle_deletion(event)
            case _:
                raise ValueError(f"Unknown event type: {event.event_type}")
    
    def _handle_upsert(self, event: IngestionEvent) -> None:
        """Generate embeddings and upsert to vector store."""
        
        # Chunk the document (simplified - production uses semantic splitter)
        chunks = self._chunk_document(self._read_payload(event))
        
        seen, new_ids = Counter(), []
        for chunk_text in chunks:
            # Generate deterministic chunk ID (idempotency key)
            chunk_id = self._generate_chunk_id(
                event.tenant_id, event.document_id, chunk_text, seen)
            new_ids.append(chunk_id)
            
            # Generate embedding
            embedding = self._embed(chunk_text)
            
            # Upsert to vector store (idempotent operation)
            self.vector_store.upsert(
                chunk_id=chunk_id,
                embedding=embedding,
                metadata={"document_id": event.document_id,
                          **event.metadata,
                          "document_uri": event.document_uri,
                          "ingested_at": event.timestamp.isoformat(),
                          "embedding_model": self.embedding_model,
                          "tenant_id": event.tenant_id},
            )
        
        # Delete the document's chunks whose IDs are not in the new version
        self._delete_stale(event, new_ids)
    
    def _read_payload(self, event: IngestionEvent) -> str:
        """The document text, decrypted from the event or read from its snapshot.
        Raises KeyDestroyed only when the key store confirms the key is destroyed;
        a missing snapshot with a live key is an ordinary failure."""
        if self.keys.is_destroyed(event.tenant_id, event.document_id):
            raise KeyDestroyed(event.event_id)
        return self.payloads.read(event)  # raises if the snapshot is gone
    
    def _delete_stale(self, event: IngestionEvent, new_ids: list[str]) -> None:
        """After the last upsert, remove chunks of older versions of this document."""
        self.vector_store.delete_except(
            {"document_id": event.document_id, "tenant_id": event.tenant_id},
            keep_ids=new_ids
        )
    
    def _handle_deletion(self, event: IngestionEvent) -> None:
        """Remove all chunks for a deleted document."""
        # This requires a metadata query to find all chunks with document_id
        # Most vector stores support delete-by-metadata filter
        self.vector_store.delete_by_filter(
            {"document_id": event.document_id, "tenant_id": event.tenant_id})
    
    def _generate_chunk_id(self, tenant_id: str, document_id: str,
                           content: str, seen: Counter) -> str:
        """Deterministic chunk ID; a per-document counter disambiguates repeated content."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        seen[content_hash] += 1
        return f"{tenant_id}:{document_id}:{content_hash}:{seen[content_hash]}"
    
    def delete_key(self, key: str) -> None:
        """Apply a Kafka tombstone: delete everything stored under tenant_id:document_id."""
        tenant_id, document_id = key.split(":", 1)
        self.vector_store.delete_by_filter(
            {"document_id": document_id, "tenant_id": tenant_id}
        )
    
    def _chunk_document(self, text: str, chunk_size: int = 500) -> list[str]:
        """Simplified chunking (production uses semantic splitter from §7.1)."""
        words = text.split()
        return [
            " ".join(words[i:i+chunk_size]) 
            for i in range(0, len(words), chunk_size)
        ]
    
    def _embed(self, text: str) -> list[float]:
        """Generate embedding (with retry logic in production)."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _generate_chunk_id(self, tenant_id: str, document_id: str,
                           content: str, seen: Counter) -> str:
        """Generate a deterministic, tenant-scoped chunk ID (no chunk index)."""
        # Tenant IDs are platform-assigned and contain no colon
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        seen[content_hash] += 1
        return f"{tenant_id}:{document_id}:{content_hash}:{seen[content_hash]}"
# --8<-- [end:lst-9.22]

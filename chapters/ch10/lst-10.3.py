# --8<-- [start:lst-10.3]
# src/08_retrieval_topology/dense_retriever.py
import asyncio
from typing import Any
import asyncpg
from datetime import datetime, UTC

class DenseRetriever:
    """Baseline: Pure vector similarity using pgvector."""
    
    def __init__(self, pool: asyncpg.Pool, embedding_model: str = "text-embedding-3-small"):
        self.pool = pool
        self.embedding_model = embedding_model
    
    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[ChildChunk]:
        """Cosine similarity search with optional metadata filtering."""
        
        # Generate query embedding (stub—replace with actual API call)
        query_embedding = await self._embed(query)
        
        # Build SQL with optional filters
        filter_clause = ""
        params: list[Any] = [query_embedding, top_k, tenant_id]
        
        if filters:
            conditions = []
            for key, value in filters.items():
                params.append(key)
                key_idx = len(params)
                params.append(value)
                conditions.append(f"metadata->>${key_idx}::text = ${len(params)}")
            filter_clause = "AND " + " AND ".join(conditions)
        
        sql = f"""
            SELECT 
                chunk_id, parent_id, content, embedding, position, metadata
            FROM child_chunks
            WHERE tenant_id = $3 {filter_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        return [
            ChildChunk(
                chunk_id=row["chunk_id"],
                parent_id=row["parent_id"],
                content=row["content"],
                embedding=row["embedding"],
                position=row["position"],
                metadata=row["metadata"],
            )
            for row in rows
        ]
    
    async def retrieve_with_parent_expansion(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
    ) -> RetrievalContext:
        """Two-stage retrieval: chunks → parents."""
        start = datetime.now(UTC)
        
        # Stage 1: Retrieve child chunks
        chunks = await self.retrieve(query, tenant_id=tenant_id, top_k=top_k)
        
        # Stage 2: Fetch parent documents
        parent_ids = list({chunk.parent_id for chunk in chunks})
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT document_id, content, summary, source_uri, ingested_at, chunk_ids
                FROM parent_documents
                WHERE document_id = ANY($1::text[]) AND tenant_id = $2
                """,
                parent_ids,
                tenant_id,
            )
        
        parents = [
            ParentDocument(
                document_id=row["document_id"],
                content=row["content"],
                summary=row["summary"],
                source_uri=row["source_uri"],
                ingested_at=row["ingested_at"],
                chunk_ids=row["chunk_ids"],
            )
            for row in rows
        ]
        
        latency_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
        total_tokens = sum(len(p.content.split()) for p in parents)  # Rough estimate
        
        return RetrievalContext(
            query=query,
            documents=parents,
            total_tokens=total_tokens,
            retrieval_latency_ms=latency_ms,
            strategy="dense_parent_child",
        )
    
    async def _embed(self, text: str) -> list[float]:
        # In 2026, we use Matryoshka-capable embeddings (e.g. text-embedding-3-large)
        # allowing us to truncate 3072 dims to 1536 or 768 for cost/performance.
        return [0.0] * 1536
# --8<-- [end:lst-10.3]

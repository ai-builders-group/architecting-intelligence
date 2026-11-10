# --8<-- [start:lst-10.15]
# src/08_retrieval_topology/hybrid_retriever.py
import asyncio
import re
from typing import Any
import asyncpg
from datetime import datetime, UTC
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval via RRF.
    
    Resilience Features:
    - Independent timeouts for dense and sparse retrievers
    - Graceful degradation to partial results on failure
    - Config-driven thresholds (no hardcoded magic numbers)
    - Circuit breaker telemetry
    """
    
    def __init__(
        self,
        pool: asyncpg.Pool,
        config: RetrievalConfig,
        embedding_model: str = "text-embedding-3-small",
        enable_bulkhead: bool = False,
    ):
        """
        Args:
            pool: Shared connection pool (or dense-specific pool if bulkhead enabled)
            config: Retrieval configuration object
            embedding_model: Model identifier for embedding API
            enable_bulkhead: If True, expects a second pool for sparse queries
        """
        self.pool = pool
        self.config = config
        self.embedding_model = embedding_model
        self.enable_bulkhead = enable_bulkhead
        
        # TODO: In production, inject a separate sparse_pool here
        # self.sparse_pool = sparse_pool if enable_bulkhead else pool
    
    async def retrieve_hybrid(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 20,
        filters: dict[str, str] | None = None,
    ) -> list[FusedResult]:
        """
        Execute parallel dense + sparse retrieval with circuit breaking.
        
        Returns partial results if one retriever fails (graceful degradation).
        """
        # Parallel execution with independent timeouts
        # The tenant is server-side: a caller cannot drop or override it
        filters = {**(filters or {}), "tenant_id": tenant_id}
        dense_task = asyncio.create_task(
            self._retrieve_dense_with_timeout(query, top_k, filters))
        sparse_task = asyncio.create_task(
            self._retrieve_sparse_with_timeout(query, top_k, filters))
        
        # Wait for both (return_exceptions=True prevents one failure from killing the other)
        results = await asyncio.gather(
            dense_task, sparse_task, return_exceptions=True)
        
        # Handle Dense Result
        if isinstance(results[0], BaseException):
            logger.warning(
                f"Dense retrieval failed: {results[0]}. "
                "Falling back to sparse-only.",
                extra={"query": query, "error": str(results[0])}
            )
            dense_results, dense_failed = [], True
        else:
            dense_results, dense_failed = results[0], False
        
        # Handle Sparse Result
        if isinstance(results[1], BaseException):
            logger.error(
                f"Sparse retrieval failed: {results[1]}. This indicates DB-level failure.",
                extra={"query": query, "error": str(results[1])}
            )
            sparse_results, sparse_failed = [], True
        else:
            sparse_results, sparse_failed = results[1], False
        
        # Failsafe: If both failed, return empty
        if not dense_results and not sparse_results:
            logger.error(
                "Total retrieval failure: both dense and sparse returned no results.",
                extra={"query": query}
            )
            return []
        
        # Fuse rankings
        fused = self.reciprocal_rank_fusion(
            rankings=[dense_results, sparse_results],
            k=self.config.rrf_k,
            is_partial=(dense_failed or sparse_failed),
        )
        
        return fused[:top_k]
    
    async def _retrieve_dense_with_timeout(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None,
    ) -> list[RankedResult]:
        """Dense retrieval with timeout circuit breaker."""
        try:
            return await asyncio.wait_for(
                self._retrieve_dense(query, top_k, filters),
                timeout=self.config.dense_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Dense retrieval timed out after {self.config.dense_timeout_ms}ms"
            )
            return []
    
    async def _retrieve_sparse_with_timeout(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None,
    ) -> list[RankedResult]:
        """Sparse retrieval with timeout circuit breaker."""
        try:
            return await asyncio.wait_for(
                self._retrieve_sparse(query, top_k, filters),
                timeout=self.config.sparse_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Sparse retrieval timed out after {self.config.sparse_timeout_ms}ms. "
                "This may indicate Shared Buffer thrashing."
            )
            return []
    
    async def _retrieve_dense(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None,
    ) -> list[RankedResult]:
        """Dense retrieval with Score Gating."""
        query_embedding = await self._embed(query)
        
        filter_clause = self._build_filter_clause(filters)
        params: list[Any] = [query_embedding, top_k]
        
        # Apply score gating only if enabled in config
        score_filter = ""
        if self.config.enable_score_gating:
            score_filter = f"AND 1 - (embedding <=> $1::vector) > {self.config.dense_min_score}"
        
        if filters:
            for value in filters.values():
                params.append(value)
        
        sql = f"""
            SELECT 
                chunk_id, parent_id,
                1 - (embedding <=> $1::vector) AS cosine_score
            FROM child_chunks
            WHERE 1=1
            {score_filter}
            {filter_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        return [
            RankedResult(
                chunk_id=row["chunk_id"],
                rank=idx + 1,
                score=row["cosine_score"],
                source=RetrievalSource.DENSE,
                parent_id=row["parent_id"],
            )
            for idx, row in enumerate(rows)
        ]
    
    async def _retrieve_sparse(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str] | None,
    ) -> list[RankedResult]:
        """
        Sparse retrieval via Postgres full-text search with Score Gating.
        
        CAVEAT: Postgres ts_rank is NOT true BM25—it lacks document-length normalization.
        For production BM25, consider Elasticsearch, Meilisearch, or Typesense.
        
        CALIBRATION: The sparse_min_score threshold (default 5.0) is corpus-dependent.
        It should be derived from a calibration run (see ADR-10.3b).
        """
        filter_clause = self._build_filter_clause(filters)
        params: list[Any] = [query, top_k]
        
        score_filter = ""
        if self.config.enable_score_gating:
            # NOTE: ts_rank is unbounded and depends on document length.
            # This threshold may require calibration per corpus.
            score_filter = f"AND ts_rank(content_tsv, plainto_tsquery('english', $1)) > {self.config.sparse_min_score}"
        
        if filters:
            for value in filters.values():
                params.append(value)
        
        sql = f"""
            SELECT 
                chunk_id,
                parent_id,
                ts_rank(content_tsv, plainto_tsquery('english', $1)) AS bm25_score
            FROM child_chunks
            WHERE content_tsv @@ plainto_tsquery('english', $1)
            {score_filter}
            {filter_clause}
            ORDER BY bm25_score DESC
            LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        return [
            RankedResult(
                chunk_id=row["chunk_id"],
                rank=idx + 1,
                score=row["bm25_score"],
                source=RetrievalSource.SPARSE,
                parent_id=row["parent_id"],
            )
            for idx, row in enumerate(rows)
        ]
    
    def reciprocal_rank_fusion(
        self,
        rankings: list[list[RankedResult]],
        k: int = 60,
        is_partial: bool = False,
    ) -> list[FusedResult]:
        """
        Merge ranked lists using RRF.
        
        Items absent from a list contribute 0 to the fusion score (rank = infinity).
        
        Args:
            rankings: List of ranked result lists (typically [dense_results, sparse_results])
            k: RRF constant (default 60 per Cormack et al.)
            is_partial: True if one retriever failed (marks results accordingly)
        """
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        rank_map: defaultdict[str, dict] = defaultdict(dict)
        parent_map: dict[str, str] = {}
        
        for result_list in rankings:
            for result in result_list:
                # Accumulate RRF score
                rrf_scores[result.chunk_id] += 1.0 / (k + result.rank)
                
                # Track ranks and raw scores by source
                if result.source == RetrievalSource.DENSE:
                    rank_map[result.chunk_id]["dense_rank"] = result.rank
                    rank_map[result.chunk_id]["dense_score"] = result.score
                elif result.source == RetrievalSource.SPARSE:
                    rank_map[result.chunk_id]["sparse_rank"] = result.rank
                    rank_map[result.chunk_id]["sparse_score"] = result.score
                
                # Store parent_id
                if result.parent_id:
                    parent_map[result.chunk_id] = result.parent_id
        
        # Build fused results
        fused = [
            FusedResult(
                chunk_id=chunk_id,
                rrf_score=score,
                dense_rank=rank_map[chunk_id].get("dense_rank"),
                sparse_rank=rank_map[chunk_id].get("sparse_rank"),
                dense_score=rank_map[chunk_id].get("dense_score"),
                sparse_score=rank_map[chunk_id].get("sparse_score"),
                parent_id=parent_map.get(chunk_id, ""),
                is_partial=is_partial,
            )
            for chunk_id, score in rrf_scores.items()
        ]
        
        # Sort by RRF score (descending)
        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        
        return fused
    
    def _build_filter_clause(self, filters: dict[str, str] | None) -> str:
        """Generate SQL WHERE clause for metadata filtering."""
        if not filters:
            return ""
        
        param_idx = 3  # After base params (embedding/query, top_k)
        conditions = []
        for key in filters.keys():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid filter key: {key!r}")
            conditions.append(f"metadata->>'{key}' = ${param_idx}")
            param_idx += 1
        
        return "AND " + " AND ".join(conditions)
    
    async def _embed(self, text: str) -> list[float]:
        """
        Generate embedding vector.
        
        In production, this should:
        1. Call OpenAI/Cohere/local model API
        2. Implement retry logic with exponential backoff
        3. Cache frequent queries
        4. Handle token truncation (>8192 tokens)
        """
        # Stub: Replace with actual embedding call
        # Example:
        # response = await openai.Embedding.acreate(
        #     model=self.embedding_model,
        #     input=text,
        # )
        # return response['data'][0]['embedding']
        
        return [0.0] * 1536
# --8<-- [end:lst-10.15]

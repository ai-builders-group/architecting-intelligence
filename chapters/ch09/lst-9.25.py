# --8<-- [start:lst-9.25]
# src/ingestion/merkle_delta.py
from __future__ import annotations

import hashlib
from typing import Protocol
from pydantic import BaseModel
import asyncpg

class MerkleNode(BaseModel):
    path: list[str]
    node_hash: str
    chunk_ids: list[str] | None = None  # Leaf nodes only

class MerkleDeltaDetector:
    """Hierarchical delta detection using Merkle trees."""
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def detect_delta(
        self,
        tenant_id: str,
        document_id: str,
        document_hash: str,
        new_chunks: list[dict]
    ) -> tuple[list[dict], list[str], list[str]]:
        """
        Compare new chunks against the stored Merkle tree.
        
        Returns:
            (chunks_to_process, chunks_to_preserve, to_delete)
        """
        # Build Merkle tree for new chunks (one branch per logical section)
        new_tree = self._build_merkle_tree(document_hash, new_chunks)
        
        async with self.pool.acquire() as conn:
            # Fetch existing root hash (single row)
            root = await conn.fetchrow(
                "SELECT node_hash FROM document_merkle_tree WHERE tenant_id = $1 AND document_id = $2 AND document_hash = $3 AND node_path = $4",
                tenant_id, document_id, document_hash,
                []  # Empty path = root
            )
            
            if not root:
                # New document - process all chunks
                await self._store_merkle_tree(conn, tenant_id, document_id,
                                              document_hash, new_tree)
                return new_chunks, [], []
            
            # Compare root hashes
            new_root = next(n for n in new_tree if n.path == [])
            if root['node_hash'] == new_root.node_hash:
                # No changes - skip everything
                return [], [c['chunk_id'] for c in new_chunks], []
            
            # Root differs - fetch branch hashes
            existing_branches = await conn.fetch(
                "SELECT node_path, node_hash FROM document_merkle_tree WHERE tenant_id = $1 AND document_id = $2 AND document_hash = $3 AND array_length(node_path, 1) = 1",
                tenant_id, document_id, document_hash
            )
            
            existing_branch_map = {
                tuple(row['node_path']): row['node_hash']
                for row in existing_branches
            }
            
            # Find dirty branches
            dirty_branches = []
            for node in new_tree:
                if len(node.path) == 1:  # Branch level
                    path_tuple = tuple(node.path)
                    if (path_tuple not in existing_branch_map
                            or existing_branch_map[path_tuple] != node.node_hash):
                        dirty_branches.append(node.path[0])
            
            # Fetch chunk IDs only for dirty branches
            dirty_chunk_ids = set()
            for branch_id in dirty_branches:
                branch_chunks = await conn.fetchval(
                    "SELECT chunk_ids FROM document_merkle_tree WHERE tenant_id = $1 AND document_id = $2 AND document_hash = $3 AND node_path = $4",
                    tenant_id, document_id, document_hash,
                    [branch_id]
                )
                if branch_chunks:
                    dirty_chunk_ids.update(branch_chunks)
            
            # Partition chunks
            chunks_to_process = [c for c in new_chunks
                                 if c['chunk_id'] in dirty_chunk_ids]
            chunks_to_preserve = [c['chunk_id'] for c in new_chunks
                                  if c['chunk_id'] not in dirty_chunk_ids]
            
            # Stored chunk IDs the new version lacks, so removed text leaves the index
            stored = await conn.fetch(
                "SELECT chunk_ids FROM document_merkle_tree WHERE tenant_id = $1 AND document_id = $2 AND document_hash = $3 AND chunk_ids IS NOT NULL",
                tenant_id, document_id, document_hash
            )
            new_ids = {c['chunk_id'] for c in new_chunks}
            to_delete = [i for row in stored for i in row['chunk_ids'] if i not in new_ids]
            
            # Update Merkle tree
            await self._store_merkle_tree(conn, tenant_id, document_id,
                                          document_hash, new_tree)
            
            return chunks_to_process, chunks_to_preserve, to_delete
    
    def _build_merkle_tree(self, document_hash: str, chunks: list[dict]) -> list[MerkleNode]:
        """Build hierarchical Merkle tree from chunks."""
        nodes = []
        
        # One branch per logical section (a chapter, say): an inserted paragraph changes
        # only its own section's branch, where fixed groups of 100 shift every later one
        by_section: dict[str, list[dict]] = {}
        for c in chunks:
            by_section.setdefault(c['section_id'], []).append(c)
        for section_id, branch_chunks in by_section.items():
            branch_id = f"section_{section_id}"
            
            # Hash individual chunks
            chunk_hashes = [c['content_hash'] for c in branch_chunks]
            
            # Hash the branch
            branch_hash = hashlib.sha256(
                "".join(chunk_hashes).encode()
            ).hexdigest()
            
            nodes.append(MerkleNode(
                path=[branch_id],
                node_hash=branch_hash,
                chunk_ids=[c['chunk_id'] for c in branch_chunks]
            ))
        
        # Hash the root
        branch_hashes = [n.node_hash for n in nodes]
        root_hash = hashlib.sha256(
            "".join(branch_hashes).encode()
        ).hexdigest()
        
        nodes.append(MerkleNode(
            path=[],
            node_hash=root_hash
        ))
        
        return nodes
    
    async def _store_merkle_tree(self, conn: asyncpg.Connection, tenant_id: str, document_id: str, document_hash: str, tree: list[MerkleNode]) -> None:
        """Persist Merkle tree to database."""
        from datetime import datetime, UTC
        
        # Delete old tree
        await conn.execute(
            "DELETE FROM document_merkle_tree WHERE tenant_id = $1 AND document_id = $2 AND document_hash = $3",
            tenant_id, document_id, document_hash
        )
        
        # Insert new tree
        await conn.executemany(
            """
            INSERT INTO document_merkle_tree (tenant_id, document_id, document_hash, node_path, node_hash, chunk_ids, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            [
                (
                    tenant_id,
                    document_id,
                    document_hash,
                    node.path,
                    node.node_hash,
                    node.chunk_ids,
                    datetime.now(UTC)
                )
                for node in tree
            ]
        )
# --8<-- [end:lst-9.25]

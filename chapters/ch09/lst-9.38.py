# --8<-- [start:lst-9.38]
# src/08_retrieval_topology/merkle_corpus_version.py
import hashlib
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime, UTC

class MerkleCorpusVersion(BaseModel):
    """
    A hierarchical versioning system for the corpus.
    
    Allows granular cache invalidation: if document D in shard S changes,
    only the cache entries for shard S are invalidated, not the entire corpus.
    """
    root_hash: str = Field(description="Root hash of the Merkle tree")
    shard_hashes: dict[str, str] = Field(description="Mapping of shard_id -> shard_hash")
    document_hashes: dict[str, str] = Field(description="Mapping of document_id -> content_hash")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))

def compute_merkle_root(documents: List[dict]) -> MerkleCorpusVersion:
    """
    Build a Merkle tree from the corpus.
    
    Sharding strategy: Group documents by domain (e.g., 'engineering', 'product', 'sales').
    """
    shards: dict[str, List[str]] = {}
    document_hashes = {}
    
    for doc in documents:
        doc_id = doc["document_id"]
        shard_id = doc.get("domain", "default")
        content_hash = hashlib.sha256(doc["content"].encode()).hexdigest()
        
        document_hashes[doc_id] = content_hash
        shards.setdefault(shard_id, []).append(content_hash)
    
    # Compute shard hashes
    shard_hashes = {}
    for shard_id, doc_hashes in shards.items():
        combined = "".join(sorted(doc_hashes))
        shard_hashes[shard_id] = hashlib.sha256(combined.encode()).hexdigest()
    
    # Compute root hash
    combined_shards = "".join(sorted(shard_hashes.values()))
    root_hash = hashlib.sha256(combined_shards.encode()).hexdigest()
    
    return MerkleCorpusVersion(
        root_hash=root_hash,
        shard_hashes=shard_hashes,
        document_hashes=document_hashes,
    )

def detect_changed_shards(old_version: MerkleCorpusVersion, new_version: MerkleCorpusVersion) -> List[str]:
    """
    Compare two Merkle versions to identify which shards changed.
    
    Returns a list of shard_ids that need cache invalidation.
    """
    changed = []
    for shard_id in (old_version.shard_hashes.keys()
                     | new_version.shard_hashes.keys()):
        if (old_version.shard_hashes.get(shard_id)
                != new_version.shard_hashes.get(shard_id)):
            changed.append(shard_id)
    return changed
# --8<-- [end:lst-9.38]

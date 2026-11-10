# --8<-- [start:lst-9.12]
# path/to/migration.py
from typing import List
import asyncio

async def blue_green_migration(
    corpus_uris: List[str],
    new_embedding_model: str,
    new_chunking_strategy: str,
    parser_version_lock: str  # Blue's version; the lockfile pins it
):
    """Zero-downtime re-indexing with parser version parity."""
    
    # Step 1: Create Green index
    green_index = create_new_index(f"corpus_green_{new_embedding_model}")
    
    # Step 2: Re-ingest raw documents with LOCKED parser version
    for uri in corpus_uris:
        raw_doc = fetch_raw_document(uri)
        
        # Ensure parser version matches Blue for validation
        chunks = chunk_document(
            raw_doc, 
            strategy=new_chunking_strategy,
            parser_version=parser_version_lock
        )
        
        embeddings = embed_chunks(chunks, model=new_embedding_model)
        await green_index.upsert(chunks, embeddings)
    
    # Step 3: Dual-read validation (query both, compare results)
    validation_queries = load_golden_set()
    for query in validation_queries:
        blue_results = await blue_index.search(query)
        green_results = await green_index.search(query)
        
        # Verify precision/recall parity
        if precision(green_results) < precision(blue_results) * 0.95:
            raise RuntimeError(
                "Green index failed precision parity; aborting cutover")
    
    # Step 4: Cutover
    switch_traffic_to(green_index)
    
    # Step 5: Deprecate Blue after 7-day monitoring window
    await asyncio.sleep(7 * 24 * 3600)
    blue_index.delete()
# --8<-- [end:lst-9.12]

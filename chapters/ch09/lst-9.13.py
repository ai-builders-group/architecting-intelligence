# --8<-- [start:lst-9.13]
# path/to/partial_update.py
from typing import Set

def handle_partial_mutation(
    document_id: str,
    old_text: str,
    new_text: str,
    strategy: str
) -> Set[str]:
    """Determine which chunks need re-indexing after a partial update."""
    
    import difflib
    
    # Identify changed lines
    matcher = difflib.SequenceMatcher(
        None, old_text.splitlines(), new_text.splitlines())
    changed_lines = set()
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag != "equal":
            changed_lines.update(range(i1, max(i2, i1 + 1)))
    
    if strategy == "contextual_prepend":
        # Global context changed: Re-index ALL chunks
        return set(get_all_chunk_ids(document_id))
    
    elif strategy == "late_chunking":
        # Global hidden states changed: Re-index ALL chunks
        return set(get_all_chunk_ids(document_id))
    
    elif strategy == "embedding_based":
        # Only re-index chunks containing mutated lines
        affected_chunks = get_chunks_containing_lines(document_id, changed_lines)
        return set(affected_chunks)
    
    else:
        # Recursive/character: Local mutation only
        affected_chunks = get_chunks_containing_lines(document_id, changed_lines)
        return set(affected_chunks)
# --8<-- [end:lst-9.13]

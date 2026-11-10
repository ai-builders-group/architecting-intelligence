# --8<-- [start:lst-10.8]
async def multi_query_transform(
    query: str,
    llm: Any,
    num_variations: int = 3,
    pruning_threshold: float = 0.9,
) -> TransformedQuery:
    """
    Decompose a query and prune semantically redundant variations 
    to prevent IOPS explosion on the Vector DB.
    """
    import datetime
    start = datetime.datetime.now()
    prompt = (
        f"Split this question into {num_variations} self-contained "
        f"sub-questions: {query}"
    )
    response = await llm.generate(prompt)
    # ... parsing logic ...
    candidates = ["var1", "var2", "var3"] # Mock
    
    # --- Query Pruning (Elite Pattern) ---
    transformed = [candidates[0]]
    for cand in candidates[1:]:
        # Heuristic: if max(cosine_sim(embed(cand), embed(kept)) for kept in
        # transformed) > threshold: prune
        is_redundant = False 
        if not is_redundant:
            transformed.append(cand)
            
    latency = int((datetime.datetime.now() - start).total_seconds() * 1000)
    return TransformedQuery(
        original_query=query,
        transformed_queries=transformed,
        strategy=TransformStrategy.MULTI_QUERY,
        transform_latency_ms=latency,
        rationale=f"Pruned to {len(transformed)} unique signals."
    )
# --8<-- [end:lst-10.8]

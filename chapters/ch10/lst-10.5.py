# --8<-- [start:lst-10.5]
class NoRelevantDocumentsError(Exception):
    """Raised when grading filters out every retrieved document."""


async def corrective_rag_pipeline(
    query: str,
    retriever: DenseRetriever,
    relevance_threshold: float = 0.7,
) -> RetrievalContext:
    """CRAG: Grade → Filter → (Optional Fallback)."""
    
    # Stage 1: Retrieve
    context = await retriever.retrieve_with_parent_expansion(query, top_k=10)
    
    # Stage 2: Grade each document
    grading_tasks = [
        grade_retrieval(query, doc.content) for doc in context.documents
    ]
    grades = await asyncio.gather(*grading_tasks)
    
    # Stage 3: Filter
    filtered_docs = [
        doc
        for doc, grade in zip(context.documents, grades)
        if (grade.grade == RelevanceGrade.RELEVANT
                and grade.confidence >= relevance_threshold)
    ]
    
    # Stage 4: No relevant docs -> fail loudly (or call a web-search fallback)
    if not filtered_docs:
        raise NoRelevantDocumentsError(
            f"No relevant documents for query: {query!r}")
    
    return RetrievalContext(
        query=query,
        documents=filtered_docs,
        total_tokens=sum(len(d.content.split()) for d in filtered_docs),
        retrieval_latency_ms=context.retrieval_latency_ms,
        strategy="crag_filtered",
    )
# --8<-- [end:lst-10.5]

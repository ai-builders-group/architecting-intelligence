# --8<-- [start:lst-1.10]
# src/agents/l2_orchestrator.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ResearchQuery(BaseModel):
    """Input for multi-step research."""
    question: str = Field(min_length=10)
    max_sources: int = Field(default=10, ge=1, le=50)
    depth: int = Field(default=2, ge=1, le=5, description="Reasoning depth")

class ResearchStep(BaseModel):
    """A single read-only operation in the chain."""
    action: str = Field(description="search | read_document | summarize | reason")
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    duration_ms: int = Field(ge=0)

class ResearchDraft(BaseModel):
    """The final draft requiring human review."""
    answer: str = Field(min_length=1)
    reasoning_chain: List[ResearchStep]
    sources_cited: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    total_duration_ms: int = Field(ge=0)

class L2Agent:
    """
    An orchestrator that autonomously performs multi-step research
    but surfaces the final output for human review.
    """
    def research(self, query: ResearchQuery) -> ResearchDraft:
        """
        Execute multi-step read-only research.
        
        Args:
            query: The research question and constraints
            
        Returns:
            ResearchDraft for human review before publication
        """
        reasoning_chain = []
        
        # Step 1: Search (read-only)
        search_results = self._search_documents(query.question)
        reasoning_chain.append(ResearchStep(
            action="search",
            input_data={"query": query.question},
            output_data={"results": search_results},
            duration_ms=120
        ))
        
        # Step 2: Read top documents (read-only)
        documents = self._read_documents(search_results[:query.max_sources])
        reasoning_chain.append(ResearchStep(
            action="read_documents",
            input_data={"doc_ids": search_results[:query.max_sources]},
            output_data={"document_count": len(documents)},
            duration_ms=450
        ))
        
        # Step 3: Synthesize answer (read-only)
        answer = self._synthesize_answer(query.question, documents)
        reasoning_chain.append(ResearchStep(
            action="synthesize",
            input_data={
                "question": query.question,
                "source_count": len(documents),
            },
            output_data={"answer_length": len(answer)},
            duration_ms=800
        ))
        
        total_time = sum(step.duration_ms for step in reasoning_chain)
        
        return ResearchDraft(
            answer=answer,
            reasoning_chain=reasoning_chain,
            sources_cited=search_results[:query.max_sources],
            confidence=0.87,
            total_duration_ms=total_time
        )
    
    def _search_documents(self, query: str) -> List[str]:
        """Mock search implementation."""
        return ["doc_1", "doc_2", "doc_3"]
    
    def _read_documents(self, doc_ids: List[str]) -> List[str]:
        """Mock document retrieval."""
        return [f"Content of {doc_id}" for doc_id in doc_ids]
    
    def _synthesize_answer(self, question: str, docs: List[str]) -> str:
        """Mock synthesis."""
        return f"Answer to '{question}' based on {len(docs)} documents."
# --8<-- [end:lst-1.10]

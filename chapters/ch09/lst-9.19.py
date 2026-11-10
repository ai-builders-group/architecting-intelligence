# --8<-- [start:lst-9.19]
# path/to/context_knapsack.py
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, UTC

class RetrievalCandidate(BaseModel):
    """A chunk candidate for context assembly."""
    chunk_id: str
    content: str
    raw_similarity: float = Field(ge=-1.0, le=1.0)
    calibrated_relevance: float = Field(ge=0.0, le=1.0, description="Platt-scaled probability")
    token_count: int
    document_date: datetime
    source_uri: str
    
    @property
    def value_density(self) -> float:
        """
        Value per token using CALIBRATED relevance (not raw similarity).
        Now mathematically sound since calibrated_relevance ≈ P(relevant).
        """
        return self.calibrated_relevance / self.token_count

class ContextAssembler:
    """Priority-based knapsack optimizer for context assembly."""
    
    def __init__(self, calibrator: RelevanceCalibrator):
        self.calibrator = calibrator
    
    def assemble(
        self,
        candidates: List[RetrievalCandidate],
        budget: int,
        diversity_penalty: float = 0.3
    ) -> List[RetrievalCandidate]:
        """Pack chunks using value density with diversity constraints."""
        
        # Apply recency boost (newer documents get +10% value)
        now = datetime.now(UTC)
        for c in candidates:
            age_days = (now - c.document_date).days
            recency_boost = max(0, 1.0 - (age_days / 365))  # Linear decay over 1 year
            c.calibrated_relevance *= (1 + 0.1 * recency_boost)
        
        # Sort by value density (now using calibrated scores)
        sorted_candidates = sorted(
            candidates, key=lambda c: c.value_density, reverse=True)
        
        # Greedy packing with diversity constraint
        packed = []
        remaining_budget = budget
        seen_sources = set()
        
        for candidate in sorted_candidates:
            # Diversity penalty: reduce value if source already included
            effective_value = candidate.calibrated_relevance
            if candidate.source_uri in seen_sources:
                effective_value *= (1 - diversity_penalty)
            
            # Pack if fits budget and exceeds value threshold
            if (
                candidate.token_count <= remaining_budget
                and effective_value > 0.3
            ):
                packed.append(candidate)
                remaining_budget -= candidate.token_count
                seen_sources.add(candidate.source_uri)
        
        return packed
# --8<-- [end:lst-9.19]

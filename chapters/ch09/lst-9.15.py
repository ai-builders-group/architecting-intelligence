# --8<-- [start:lst-9.15]
# path/to/selectivity_model.py
from pydantic import BaseModel, Field
from collections import defaultdict
from typing import Dict, Tuple
import numpy as np

class SelectivityHistogram(BaseModel):
    """
    Empirical selectivity model using co-occurrence counting.
    Tracks joint distributions to account for filter correlation.
    """
    # Single-filter counts
    single_counts: Dict[str, Dict[str, int]] = Field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    # Pairwise co-occurrence counts
    pair_counts: Dict[Tuple[str, str], Dict[Tuple[str, str], int]] = Field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    
    # Total document count
    total_documents: int = 0
    
    class Config:
        arbitrary_types_allowed = True
    
    def update(self, metadata: dict):
        """
        Update counts with a new document's metadata.
        """
        self.total_documents += 1
        
        # Update single-filter counts
        for field, value in metadata.items():
            if value is not None:
                self.single_counts[field][str(value)] += 1
        
        # Update pairwise counts (all combinations)
        fields = list(metadata.items())
        for i, (field1, val1) in enumerate(fields):
            for field2, val2 in fields[i+1:]:
                if val1 is not None and val2 is not None:
                    key = (field1, field2)
                    val_pair = (str(val1), str(val2))
                    self.pair_counts[key][val_pair] += 1
    
    def estimate_selectivity(self, filters: list[Tuple[str, str]]) -> float:
        """
        Estimate composite selectivity using empirical joint probabilities.
        
        Args:
            filters: List of (field, value) tuples
        
        Returns:
            Estimated fraction of corpus that passes all filters
        """
        if not filters:
            return 1.0
        
        if self.total_documents == 0:
            return 0.0
        
        # Single filter
        if len(filters) == 1:
            field, value = filters[0]
            count = self.single_counts[field].get(str(value), 0)
            return count / self.total_documents
        
        # Two filters: use empirical joint probability
        if len(filters) == 2:
            field1, val1 = filters[0]
            field2, val2 = filters[1]
            
            # Joint count of the two filter values
            key = (field1, field2)
            val_pair = (str(val1), str(val2))
            if key in self.pair_counts:
                count = self.pair_counts[key].get(val_pair, 0)

                return count / self.total_documents
            else:
                # Untracked field pair only: independence assumption
                s1 = self.estimate_selectivity([filters[0]])
                s2 = self.estimate_selectivity([filters[1]])
                return s1 * s2
        
        # Three+ filters: use pairwise correlation heuristic
        # Estimate joint P(A ∩ B ∩ C) ≈ P(A ∩ B) * P(C | A ∩ B)
        # Approximate P(C | A ∩ B) ≈ P(C | A) using pairwise stats
        
        result = self.estimate_selectivity(filters[:2])
        
        for field, value in filters[2:]:
            # Estimate conditional probability P(field=value | previous filters)
            # Approximate using correlation with first filter
            field1, val1 = filters[0]
            key = (field1, field)
            
            # Count documents where both conditions hold
            pair_count = 0
            for (v1, v2), count in self.pair_counts.get(key, {}).items():
                if v1 == str(val1) and v2 == str(value):
                    pair_count = count
                    break
            
            # Count documents where first condition holds
            base_count = self.single_counts[field1].get(str(val1), 1)
            
            # Conditional probability
            cond_prob = pair_count / base_count if base_count > 0 else 0.01
            
            result *= max(cond_prob, 0.01)  # Floor to prevent underflow
        
        return min(result, 1.0)
# --8<-- [end:lst-9.15]

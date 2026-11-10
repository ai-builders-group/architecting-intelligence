# --8<-- [start:lst-9.29]
# path/to/token_budget.py
from pydantic import BaseModel, Field, computed_field
from typing import Protocol, List
from datetime import datetime, UTC
from enum import IntEnum
import tiktoken

class LayerPriority(IntEnum):
    """Lower value = higher priority (never evicted first)."""
    SYSTEM = 0
    QUERY = 1
    KNOWLEDGE = 2
    MEMORY = 3

class ContextLayer(BaseModel):
    """A segment of the context window with eviction policy."""
    name: str
    content: str
    priority: LayerPriority
    is_mutable: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    @computed_field
    @property
    def token_count(self) -> int:
        """Exact token count using model's tokenizer."""
        # CRITICAL: Use o200k_base for GPT-4o, cl100k_base for GPT-4
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(self.content))

class ContextBudget(BaseModel):
    """Token allocation constraints for a specific model."""
    model_name: str
    total_context: int  # e.g., 128000 for GPT-4o
    reserved_system: int = 1000
    reserved_output: int = 4096  # MUST match max_completion_tokens
    safety_margin: int = 500  # Buffer for tokenization variance
    
    @computed_field
    @property
    def available_budget(self) -> int:
        """Tokens available for dynamic allocation (Knowledge + Memory)."""
        return (
            self.total_context 
            - self.reserved_system 
            - self.reserved_output 
            - self.safety_margin
        )
    
    def validate_allocation(self, layers: List[ContextLayer]) -> bool:
        """Verify total token usage does not exceed budget."""
        total_used = sum(layer.token_count for layer in layers)
        return total_used <= (self.total_context - self.reserved_output)
# --8<-- [end:lst-9.29]

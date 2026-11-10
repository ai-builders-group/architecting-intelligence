# --8<-- [start:lst-5.11]
# src/04_probabilistic_state_machine/schemas/reflection_state.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Critique(BaseModel):
    """
    A single critique in the Reflexion loop.
    """
    attempt: int = Field(description="Retry number (1-indexed).")
    generated_output: str = Field(description="The model's output for this attempt.")
    evaluation_result: str = Field(description="Pass/Fail + error message.")
    reflection: str = Field(description="The Critic's analysis of the failure.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ReflectionState(BaseModel):
    """
    State schema for a Reflexion-based agent.
    """
    trace_id: str
    task_description: str
    
    # Retry tracking
    attempt: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3)
    
    # Critique memory (cumulative)
    critiques: List[Critique] = Field(default_factory=list)
    
    # Output
    validated_result: Optional[str] = Field(default=None)
    requires_human: bool = Field(default=False)
    
    # Observability
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
# --8<-- [end:lst-5.11]

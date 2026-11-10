# --8<-- [start:lst-5.2]
# src/04_probabilistic_state_machine/schemas/agent_state.py
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
from datetime import datetime

class AgentState(BaseModel):
    """
    The canonical state object for a graph-based agent.
    
    Design Principles:
    - Immutable History: Append-only list of actions taken.
    - Mutable Scratchpad: Current working memory for the active node.
    - Metadata: Trace information for observability.
    """
    
    # Immutable: The audit trail
    # Append-only by convention: The audit trail
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Append-only log of (node_id, input, output, timestamp)."
    )
    
    # Mutable: The working memory
    scratchpad: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value store for intermediate results."
    )
    
    # Metadata: Observability
    trace_id: str = Field(
        description=(
            "Unique identifier for this execution trace. MUST be propagated "
            "across all async boundaries, tool calls, and external API "
            "invocations for distributed tracing."
        )
    )
    current_node: str = Field(description="The active node in the graph.")
    retry_count: int = Field(default=0, description="Number of retries for the current node.")
    
    # Exit Signals
    is_complete: bool = Field(default=False, description="Terminal condition flag.")
    requires_human: bool = Field(default=False, description="Escalation flag.")

class ContractAnalysisState(AgentState):
    """
    Domain-specific extension for the contract analysis use case.
    """
    
    # Extracted Entities (populated by the Extract node)
    parties: Optional[List[str]] = None
    terms: Optional[Dict[str, str]] = None
    
    # Risk Scores (populated by the Analyze node)
    risk_flags: Optional[List[str]] = None
    
    # Validated Output (populated by the Validate node)
    validated: bool = Field(default=False)
    
    # Final Output (populated by the Compose node)
    summary_email: Optional[str] = None

# --8<-- [end:lst-5.2]

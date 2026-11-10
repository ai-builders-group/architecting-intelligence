# --8<-- [start:lst-5.8]
# src/04_probabilistic_state_machine/schemas/agentic_state.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime

class ThoughtStep(BaseModel):
    """
    A single reasoning step in the ReAct loop.
    """
    thought: str = Field(description="The model's internal reasoning.")
    action: Optional[str] = Field(
        default=None,
        description="The tool call (e.g., 'weather_api(Paris)')."
    )
    observation: Optional[str] = Field(
        default=None,
        description="The tool's output."
    )
    critique: Optional[str] = Field(
        default=None,
        description="The Critic's evaluation of this step."
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentState(BaseModel):
    """
    Minimal stand-in for the chapter's base state.
    """
    trace_id: str
    user_query: str
    thoughts: List[ThoughtStep] = Field(default_factory=list)

class AgenticState(AgentState):
    """
    The canonical state object for a Reflection-Enhanced ReAct agent.
    
    Design Principles:
    - Immutable History: Append-only list of ThoughtSteps.
    - Retry Budget: Enforced at the Terminator node.
    - Escalation: Automatic routing to DLQ after max retries.
    """
    
    # Core Fields
    trace_id: str = Field(description="OpenTelemetry trace ID.")
    user_query: str = Field(description="The original user request.")
    
    # Execution State
    current_node: Literal["planner", "actor", "observer", "critic", "terminator"] = Field(
        default="planner"
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Failed passes in the current ReAct cycle."
    )
    max_attempts: int = Field(default=3, description="Attempts allowed in all.")
    
    # History (Immutable)
    thoughts: List[ThoughtStep] = Field(
        default_factory=list,
        description=("Earlier steps are never modified; the newest step may "
                     "be replaced with its critiqued copy.")
    )
    
    # Exit Signals
    is_complete: bool = Field(default=False)
    requires_human: bool = Field(default=False)
    final_answer: Optional[str] = Field(
        default=None,
        description="The validated output, set by the Terminator node."
    )
    
    # Observability
    total_tokens: int = Field(default=0, description="Running token count.")
    total_cost_usd: float = Field(default=0.0, description="Running cost in USD.")

class ReActState(AgenticState):
    """
    Domain-specific extension for tool-using agents.
    """
    available_tools: List[str] = Field(
        default_factory=list,
        description="Whitelist of tools the agent may invoke."
    )
    tool_results: Dict[str, str] = Field(
        default_factory=dict,
        description="Cache of tool outputs (for idempotency)."
    )
# --8<-- [end:lst-5.8]

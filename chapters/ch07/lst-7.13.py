# --8<-- [start:lst-7.13]
# src/agent/tools/human_tool.py
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

class ProposalContext(BaseModel):
    """
    The decision context provided to the human.
    
    This is the "prompt" for the human—the information they need
    to approve or reject the agent's proposed action.
    """
    
    # The action the agent wants to take
    action: str = Field(
        description="Natural language description of the proposed action"
    )
    
    # Supporting evidence
    reasoning: str = Field(
        description=(
            "Concise rationale linked to the evidence; never raw chain-of-thought"
        )
    )
    
    retrieved_docs: list[str] = Field(
        default_factory=list,
        description="Relevant documents retrieved from RAG",
    )
    
    # Confidence signals (from §5.0 intro)
    confidence_signals: dict[str, float] = Field(
        description="Entropy, novelty, validation scores"
    )
    
    # Risk metadata
    estimated_cost: Optional[float] = Field(
        None,
        description="Estimated dollar cost of this action if executed",
    )
    
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="Human-readable risk classification"
    )
    
    # Workflow context (for resumption)
    workflow_id: str = Field(description="Unique ID for this agent execution")
    tenant_id: str = Field(description="Tenant that owns this run")
    content_hash: str = Field(
        description="Hash of the proposal the approver sees"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HumanDecision(BaseModel):
    """
    The human's response to the proposal.
    """
    
    approved: bool = Field(description="Whether the human approved the action")
    proposal_hash: str = Field(
        description="Content hash of the proposal this decision answers"
    )
    version: Literal[1] = Field(
        1, description="Schema version; any other value is rejected"
    )
    
    feedback: Optional[str] = Field(
        None,
        description="Free-form feedback explaining the decision",
    )
    
    modifications: Optional[dict[str, Any]] = Field(
        None,
        description="Suggested modifications to the proposal (if rejected)",
    )
    
    decided_at: datetime = Field(default_factory=datetime.utcnow)
# --8<-- [end:lst-7.13]

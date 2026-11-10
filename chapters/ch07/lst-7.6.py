# --8<-- [start:lst-7.6]
# src/agent/proposal.py
from typing import Literal, Optional, Any
from pydantic import BaseModel, ConfigDict, Field, computed_field
from datetime import datetime
import hashlib
import json

class StateChange(BaseModel, frozen=True):
    """A single atomic state mutation."""
    field: str = Field(description="The state field being modified")
    before: Any = Field(description="Current value")
    after: Any = Field(description="Proposed value")
    impact: str = Field(description="Human-readable impact summary")

class Evidence(BaseModel, frozen=True):
    """A retrieved document or tool result that informed the decision."""
    source: str = Field(description="Document ID or tool name")
    excerpt: str = Field(description="Relevant snippet (max 500 chars)")
    relevance_score: float = Field(ge=0, le=1, description="Retrieval score")

class Step(BaseModel, frozen=True):
    """One planned tool call."""
    action: str
    tool: str
    args: dict[str, Any]

class Proposal(BaseModel):
    """An atomic, immutable agent proposal requiring human approval."""
    model_config = ConfigDict(frozen=True)
    
    proposal_id: str = Field(description="Unique identifier (UUID)")
    tenant_id: str = Field(description="Tenant that owns the proposal")
    agent_id: str = Field(description="Agent that generated this proposal")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Core content
    action: str = Field(description="High-level action name (e.g., 'refund_customer')")
    rationale: str = Field(description="Agent's explanation (2-3 sentences)")
    state_changes: tuple[StateChange, ...] = Field(description="Before/after")
    evidence: tuple[Evidence, ...] = Field(description="Supporting data")
    steps: tuple[Step, ...] = Field(description="Plan, in order")
    
    # Confidence signals (from §5.0)
    confidence_score: float = Field(ge=0, le=1, description="Agent's self-reported confidence")
    entropy: float = Field(ge=0, description="Logit entropy (nats)")
    novelty_score: float = Field(ge=0, le=1, description="OOD detection score")
    
    # Risk metadata
    risk_level: Literal["low", "medium", "high"] = Field(description="Computed risk tier")
    model_id: str = Field(description="Model that wrote the plan")
    financial_impact: Optional[float] = Field(None, description="Dollar impact (if applicable)")
    
    # Tamper evidence
    @computed_field
    @property
    def content_hash(self) -> str:
        """Cryptographic hash of proposal content for tamper detection."""
        # Serialize deterministically (sort keys, exclude timestamp/hash)
        payload = self.model_dump(mode="json", include={
            "tenant_id", "proposal_id", "action", "rationale", "risk_level",
            "model_id", "steps", "state_changes", "evidence"})
        canonical = json.dumps(payload, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

class ApprovalDecision(BaseModel):
    """Human's response to a proposal."""
    
    proposal_id: str
    approved: bool
    approver_id: str = Field(description="Human reviewer (email or employee ID)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    comment: Optional[str] = Field(None, description="Optional human explanation")
    
    # Bind approval to the exact proposal content
    proposal_hash: str = Field(description="SHA-256 hash of approved proposal")
# --8<-- [end:lst-7.6]

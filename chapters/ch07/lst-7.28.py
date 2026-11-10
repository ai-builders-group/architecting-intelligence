# --8<-- [start:lst-7.28]
# src/agent/proposal.py
from typing import Any, Protocol
from pydantic import BaseModel, Field

class Step(BaseModel):
    """A single step in an agentic plan."""
    action: str
    tool: str
    args: dict[str, Any]
    confidence: ConfidenceSignals

class Proposal(BaseModel):
    """Batch of uncertain steps presented to human for approval."""
    steps: list[Step]
    risk_score: float = Field(ge=0.0, le=1.0)
    rationale: str

async def execute_with_coalescing(
    plan: list[Step],
    escalation_policy: EscalationPolicy,
) -> None:
    """
    Stage the plan without side effects, batch uncertain steps into one proposal,
    and commit the staged plan only on approval.
    """
    proposal_buffer: list[Step] = []
    staged: list[Step] = []
    
    for step in plan:
        staged.append(plan_step(step))  # stages only, no side effects
        
        escalate, reason = should_escalate(
            step.confidence,
            escalation_policy
        )
        
        if escalate:
            proposal_buffer.append(step)
    
    # If any step was uncertain, pause and request batch approval
    if proposal_buffer:
        proposal = Proposal(
            steps=proposal_buffer,
            risk_score=calculate_risk(proposal_buffer),
            rationale="Multiple steps flagged for review"
        )
        
        decision = await ask_human(proposal, timeout_seconds=3600)
        
        if decision is None or not decision.approved:
            return  # nothing was applied

    # Approved, or nothing was escalated: commit the staged plan
    await commit_steps(staged)
# --8<-- [end:lst-7.28]

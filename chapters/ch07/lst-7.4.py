# --8<-- [start:lst-7.4]
# src/agent/nodes.py
from typing import Literal
from pydantic import BaseModel

class EscalationResponse(BaseModel):
    """Response from attempting to escalate to human."""
    status: Literal["answered", "rejected", "timeout"]
    decision: ApprovalDecision | None = None
    reason: str | None = None
    fallback_action: str | None = None


async def human_review_node(
    state: AgentState,
    admission_controller: AdmissionController,
    timeout_seconds: int = 3600,
) -> EscalationResponse:
    """
    Attempt to escalate to human. Apply backpressure if queue is saturated.
    
    Returns:
        EscalationResponse with status and fallback instructions.
    """
    # Check admission control
    can_escalate = admission_controller.should_admit_escalation(
        confidence_score=state.confidence,
        threshold=state.escalation_threshold,
    )
    
    if not can_escalate:
        # Queue saturated. Signal backpressure to router.
        return EscalationResponse(
            status="rejected",
            reason="Human review queue at capacity",
            fallback_action="auto_deny_high_risk",  # Deterministic policy
        )
    
    # Escalate to human (async wait)
    response = await admit(ask_human, state.proposal, timeout_seconds)
    if response == "deny":  # no free slot
        return EscalationResponse(status="rejected", reason="No review slot free")
    if response == "escalate":  # admit() returns "escalate" on timeout
        return EscalationResponse(
            status="timeout",
            reason="Human did not respond within SLA",
            fallback_action="escalate_to_manager",
        )
    return EscalationResponse(status="answered", decision=response)
# --8<-- [end:lst-7.4]

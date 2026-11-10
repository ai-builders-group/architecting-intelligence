# --8<-- [start:lst-7.9]
# src/agent/executor.py
from typing import Optional

async def execute_with_approval(
    plan: list[Step],
    state: AgentState,
) -> AgentState:
    """
    Execute a multi-step plan with human-in-the-loop approval.
    Uses optimistic execution with rollback on rejection.
    """
    # Phase 1: Generate proposal (no side effects)
    proposal = await generate_proposal(plan, state)
    
    # Phase 2: Request approval
    decision = await ask_human_sync(proposal, timeout_seconds=60)
    
    # A timeout (None) and a rejection both leave the state unchanged
    if decision is None or not decision.approved:
        logger.info(f"Proposal {proposal.proposal_id} was not approved")
        return state  # Return unchanged state
    
    if decision.proposal_hash != proposal.content_hash:
        await audit_log.alert_mismatch(proposal, decision)
        raise PermissionError("Approval does not match the proposal content")
    
    # Log approval for audit trail
    await audit_log.record_approval(decision)
    
    # Phase 3: Execute approved plan
    for step in proposal.steps:
        if not before_matches(step, state):
            raise StateConflict(409, "State changed since approval")
        state = await execute_step(step, state)
    
    return state
# --8<-- [end:lst-7.9]

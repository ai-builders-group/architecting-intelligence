# --8<-- [start:lst-7.21]
# src/agent/tools/human_async.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

class AsyncHumanTool:
    """Asynchronous human-in-the-loop via webhook resumption."""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        notification_service: "NotificationService",
    ):
        self.checkpoint_manager = checkpoint_manager
        self.notification_service = notification_service
    
    async def ask_human(
        self,
        run_id: UUID,
        prompt: str,
        proposal: dict,
        state_snapshot: dict,
        ttl_hours: int = 72,
    ) -> CheckpointMetadata:
        """
        Save checkpoint and notify human. Do NOT block.
        
        Returns:
            CheckpointMetadata (including resume_token for webhook)
        """
        metadata = await self.checkpoint_manager.save_checkpoint(
            run_id=run_id,
            node_name="wait_for_human",
            state_snapshot=state_snapshot,
            agent_version="1.0.0",
            tenant_id=proposal["tenant_id"], approver_id=proposal["approver_id"],
            ttl_hours=ttl_hours,
        )
        
        # Send notification with resume link
        approval_url = f"https://app.example.com/approve/{metadata.resume_token}"
        await self.notification_service.send_email(
            to=self._get_approver_email(run_id),
            subject=f"Approval Required: {prompt}",
            body=f"""
            An agent workflow requires your approval.
            
            Prompt: {prompt}
            Proposal: {proposal}
            
            [Review and decide]({approval_url})
            
            This request expires in {ttl_hours} hours.
            """,
        )
        
        return metadata
    
    def _get_approver_email(self, run_id: UUID) -> str:
        """Lookup the human responsible for this workflow (implementation-specific)."""
        # E.g., query user DB, routing table, org chart
        pass

# Webhook endpoint (FastAPI example)
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

app = FastAPI()
bearer = HTTPBearer()

class ApprovalWebhookPayload(BaseModel):
    resume_token: str
    approved: bool
    feedback: Optional[str] = None

class HumanResponse(BaseModel):
    approved: bool
    feedback: Optional[str] = None
    timestamp: datetime

class AgentState(BaseModel):
    human_feedback: Optional[HumanResponse] = None

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> Optional[str]:
    """Return the authenticated caller's identity (implementation-specific)."""
    pass

async def caller_owns_run(caller: str, tenant_id: str, approver_id: str) -> bool:
    """True if the caller is the named approver in this tenant (implementation-specific)."""
    pass

@app.post("/api/v1/agent/resume")
async def resume_workflow(
    payload: ApprovalWebhookPayload,
    caller: Optional[str] = Depends(get_current_user),
):
    """
    Webhook endpoint for human approval decisions.
    
    Flow:
    1. Verify the resume token and load the checkpoint (read-only).
    2. Check the caller owns the run.
    3. Claim the resume once.
    4. Reconstruct agent state.
    5. Resume graph execution at next node.
    """
    if not caller:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Verify the token and load the checkpoint (writes nothing)
    checkpoint = await checkpoint_manager.verify_and_load(payload.resume_token)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    
    if not await caller_owns_run(
        caller, checkpoint.tenant_id, checkpoint.approver_id):
        raise HTTPException(
            status_code=403, detail="Not the approver for this run"
        )
    
    # Claim the resume only after the caller is authorized
    checkpoint = await checkpoint_manager.claim(
        checkpoint.checkpoint_id, caller, payload.approved
    )
    
    # Duplicate resume: answer 200 with the recorded result, act on nothing
    if checkpoint.duplicate:
        return {
            "status": "pending" if checkpoint.result is None else "duplicate",
            "run_id": str(checkpoint.run_id),
            "result": checkpoint.result,
        }
    
    # Reconstruct agent state
    state = AgentState.model_validate(checkpoint.state_snapshot)
    
    # Inject human decision into state
    state.human_feedback = HumanResponse(
        approved=payload.approved,
        feedback=payload.feedback,
        timestamp=datetime.utcnow(),
    )
    
    # Resume graph execution (implementation depends on orchestrator)
    # If using LangGraph:
    # await graph.ainvoke(state, config={"run_id": checkpoint.run_id})
    
    # If using raw state machine:
    # await execute_next_node(state, node_name="process_approval")
    
    result = {"status": "resumed", "run_id": str(checkpoint.run_id)}
    await checkpoint_manager.record_result(checkpoint.checkpoint_id, result)
    return result
# --8<-- [end:lst-7.21]

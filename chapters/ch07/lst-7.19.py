# --8<-- [start:lst-7.19]
# src/api/routes/decisions.py
import os
import hmac
import hashlib
from typing import Literal, Protocol
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel

router = APIRouter()

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]


class HumanDecision(BaseModel):
    approved: bool
    comment: str = ""
    proposal_hash: str
    version: Literal[1] = 1  # any other value fails validation


class ProposalStorage(Protocol):
    async def save_decision(self, tenant_id: str, workflow_id: str, decision: HumanDecision) -> None: ...


async def get_caller_id() -> str:
    """Authenticated caller identity, supplied by the auth layer."""
    raise NotImplementedError


async def load_owner(tenant_id: str, run_id: str) -> str | None:
    """Approver assigned to the run, read from the database (never from graph state)."""
    raise NotImplementedError


async def resume_run(tenant_id: str, workflow_id: str, caller_id: str) -> None:
    """Resume a suspended run, but only for the run's stored owner."""
    owner = await load_owner(tenant_id, workflow_id)
    if owner is None or not hmac.compare_digest(owner, caller_id):
        raise HTTPException(
            status_code=403, detail="Caller does not own this run"
        )
    ...  # hand the run back to the graph executor


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    """
    Verify HMAC signature to prevent webhook spoofing.
    
    Security Note: Without signature verification, an attacker could POST
    a fake approval to this endpoint, bypassing human review entirely.
    
    See Chapter 14 (Defensive Architectures) for deeper discussion of
    webhook security and capability URLs.
    
    Args:
        payload: Raw request body bytes
        signature: HMAC signature from X-Webhook-Signature header
        secret: Shared secret key
        
    Returns:
        True if signature is valid
    """
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def resume_run(tenant_id: str, workflow_id: str, caller_id: str) -> None:
    """Resume a suspended run, but only for the run's stored owner."""
    owner = await load_owner(tenant_id, workflow_id)
    if owner is None or not hmac.compare_digest(owner, caller_id):
        raise HTTPException(
            status_code=403, detail="Caller does not own this run"
        )
    ...  # hand the run back to the graph executor


@router.post("/v1/tenants/{tenant_id}/decisions/{workflow_id}/respond")
async def receive_human_decision(
    tenant_id: str, workflow_id: str, request: Request,
    decision: HumanDecision,
    storage: ProposalStorage,
    caller_id: str = Depends(get_caller_id),
    x_webhook_signature: str = Header(None),
):
    """
    Webhook endpoint for human decisions.
    
    When the human clicks Approve/Reject in Slack or Email, this endpoint:
    1. Verifies the request signature (HMAC)
    2. Persists the decision to storage
    3. Triggers the graph executor to resume the workflow
    4. Returns 200 OK
    
    The workflow, which was suspended via HumanInterruptSignal, will
    resume and re-call HumanTool.execute()—this time, the decision exists.
    
    Security: Requires HMAC signature verification to prevent spoofing.
    See Chapter 14 §14.6 for webhook authentication patterns.
    """
    # Verify signature (defense against fake approvals)
    if not x_webhook_signature:
        raise HTTPException(status_code=403, detail="Missing webhook signature")
    payload = await request.body()  # the signature covers the raw request bytes
    if not verify_webhook_signature(payload, x_webhook_signature, WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")
    owner = await load_owner(tenant_id, workflow_id)
    if owner is None or not hmac.compare_digest(owner, caller_id):
        raise HTTPException(
            status_code=403, detail="Caller does not own this run"
        )

    # Only the run's stored owner may record a decision
    owner = await load_owner(tenant_id, workflow_id)
    if owner is None or not hmac.compare_digest(owner, caller_id):
        raise HTTPException(
            status_code=403, detail="Caller does not own this run"
        )
    
    # Persist the decision
    await storage.save_decision(tenant_id, workflow_id, decision)
    
    # Trigger workflow resumption
    await resume_run(tenant_id, workflow_id, caller_id)
    
    return {"status": "ok", "workflow_id": workflow_id}
# --8<-- [end:lst-7.19]

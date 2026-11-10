# --8<-- [start:lst-7.14]
# src/agent/tools/human_tool.py (continued)
import asyncio
import re
from typing import Optional
from abc import ABC, abstractmethod

class HumanInterruptSignal(Exception):
    """
    Exception raised to halt graph execution and checkpoint state.
    
    This is NOT an error—it is a control flow signal that tells the
    graph executor to:
    1. Serialize current state to durable storage
    2. Terminate this execution
    3. Wait for external event (human decision) to trigger resumption
    """
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        super().__init__(f"Human interrupt: workflow {workflow_id}")


class HumanNotifier(ABC):
    """
    Abstract interface for notifying humans of pending decisions.
    Concrete implementations: SlackNotifier, EmailNotifier, WebhookNotifier.
    """
    
    @abstractmethod
    async def send_proposal(
        self,
        proposal: ProposalContext,
        callback_url: str,
    ) -> str:
        """
        Send the proposal to the human and return a notification ID.
        
        Args:
            proposal: The decision payload
            callback_url: Where the human should POST their decision
            
        Returns:
            notification_id: Unique ID for tracking this notification
        """
        ...


class HumanTool:
    """
    A tool that requests human judgment by suspending agent execution.
    
    This is the "interrupt" mechanism. The agent calls this tool, the graph
    serializes its state, and execution halts until the human responds.
    
    Key Invariant: This function is IDEMPOTENT. If called multiple times
    with the same workflow_id, it returns the same decision (or raises
    the same interrupt if still pending).
    """
    
    def __init__(
        self,
        notifier: HumanNotifier,
        storage: "ProposalStorage",  # Forward reference
        timeout_seconds: int = 3600,  # 1 hour default
    ):
        self.notifier = notifier
        self.storage = storage
        self.timeout_seconds = timeout_seconds
    
    async def execute(
        self,
        proposal: ProposalContext,
    ) -> HumanDecision:
        """
        Request human approval for the proposed action.
        
        This function implements the Idempotent Re-entry Pattern:
        1. Check if decision already exists (Resume path)
        2. If Yes: Return it immediately
        3. If No: Send notification and raise Interrupt (First call)
        
        The graph executor catches HumanInterruptSignal and checkpoints state.
        When the human responds (via webhook), the executor reloads state
        and calls this function again—this time, the decision exists.
        
        Args:
            proposal: The decision context
            
        Returns:
            The human's decision (approve/reject/modify)
            
        Raises:
            HumanInterruptSignal: If decision is pending (first call)
            TimeoutError: If decision not received within timeout window
        """
        # IDEMPOTENCY CHECK: Has the human already responded?
        existing_decision = await self.storage.get_decision(proposal)
        
        if existing_decision is not None:
            # This is a RESUME. The human responded while we were suspended.
            # Validate the decision schema (defense against malicious humans)
            validated_decision = self._validate_human_decision(
                existing_decision, proposal)
            return validated_decision
        
        # Decision does not exist. Check if we've already sent the notification.
        pending = await self.storage.get_pending_proposal(proposal.workflow_id)
        
        if pending is None:
            # FIRST CALL: claim the pending mark (set-if-absent), then notify once
            if await self.storage.mark_pending(proposal.workflow_id, "claimed"):
                await self.storage.save_proposal(proposal)
                callback_url = self._generate_callback_url(proposal)
                await self.notifier.send_proposal(proposal, callback_url)
        
        # Check if timeout has been exceeded
        if pending and self._is_timed_out(pending.created_at):
            # Human did not respond in time. Apply fallback policy.
            fallback_decision = await self._handle_timeout(proposal)
            await self.storage.save_decision(proposal.tenant_id, proposal.workflow_id, fallback_decision)
            return fallback_decision
        
        # No decision yet, no timeout. Raise interrupt to suspend execution.
        raise HumanInterruptSignal(proposal.workflow_id)
    
    def _validate_human_decision(
        self,
        decision: HumanDecision,
        proposal: ProposalContext,
    ) -> HumanDecision:
        """
        Validate the human's decision against business logic.
        
        Edge Case: The Malicious Human
        What if the human returns `approved: True` but `modifications`
        contains SQL injection or shell commands? We must validate the
        human's output with the same rigor we apply to LLM outputs.
        
        Args:
            decision: The raw decision from storage
            proposal: The original proposal context
            
        Returns:
            Validated decision
            
        Raises:
            ValueError: If decision violates schema or business rules
        """
        # Schema validation (Pydantic already enforces this at deserialization)
        # But we add business-level checks here:
        
        if decision.modifications:
            # Allowlist: known proposal fields, scalar values, safe characters only
            allowed_keys = {"action", "estimated_cost"}
            for key, value in decision.modifications.items():
                if key not in allowed_keys:
                    raise ValueError(f"Human decision modifies unsupported field: {key}")
                if not isinstance(value, (str, int, float)):
                    raise ValueError(f"Human decision has non-scalar modification: {key}")
                if isinstance(value, str) and not re.fullmatch(r"[A-Za-z0-9 _.,:@/%+\-]{0,500}", value):
                    raise ValueError(f"Human decision contains unsafe modifications: {key}")
        
        if decision.proposal_hash != proposal.content_hash:
            raise ValueError("Decision is bound to a different proposal hash")
        # Check logical consistency
        if decision.approved and proposal.risk_level == "critical":
            # Require feedback for critical approvals
            if not decision.feedback or len(decision.feedback) < 10:
                raise ValueError("Critical risk approvals require detailed feedback")
        
        return decision
    
    def _is_timed_out(self, created_at: datetime) -> bool:
        """Check if proposal has exceeded timeout window."""
        elapsed = (datetime.utcnow() - created_at).total_seconds()
        return elapsed > self.timeout_seconds
    
    async def _handle_timeout(
        self,
        proposal: ProposalContext,
    ) -> HumanDecision:
        """
        Fallback policy when human exceeds SLA.
        
        This is where the ARD's escalation policy is encoded.
        Options (a timeout never approves):
        - Auto-reject
        - Escalate to a backup human (manager, on-call)
        """
        return HumanDecision(
            proposal_hash=proposal.content_hash,
            approved=False,
            feedback="Auto-rejected: Human timeout",
        )
    
    def _generate_callback_url(self, proposal: ProposalContext) -> str:
        """Generate a unique webhook URL for this decision."""
        base = "https://api.example.com/v1"
        return f"{base}/tenants/{proposal.tenant_id}/decisions/{proposal.workflow_id}/respond"
# --8<-- [end:lst-7.14]

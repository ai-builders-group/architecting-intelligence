# --8<-- [start:lst-7.25]
# src/agent/supervision/exception_handler.py
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional
from uuid import UUID
from datetime import datetime, UTC

from pydantic import BaseModel, Field

class Gate(Enum):
    """Routing decision for a proposal (same enum as the circuit breaker's)."""
    AUTONOMOUS = "autonomous"
    SHED = "shed"
    ESCALATE = "escalate"

class ConfidenceSignals(BaseModel):
    """Confidence metadata attached to a proposed action."""
    score: float = Field(ge=0.0, le=1.0)

class ActionProposal(BaseModel):
    """A proposed action with confidence metadata."""
    action: dict  # Tool call specification
    confidence: ConfidenceSignals
    reasoning: str  # Chain-of-thought trace

class ExceptionHandlerSupervisor:
    """
    Threshold-based escalation with multi-head verification.
    
    This pattern enforces L3 autonomy: the agent acts independently unless
    verification fails, then escalates to a human.
    """
    
    def __init__(
        self,
        policy: EscalationPolicy,
        human_tool: AsyncHumanTool,
        verifier: MultiHeadVerifier,
        breaker: CircuitBreakerSupervisor,
        fallback_action: Literal["reject", "defer", "safe_default"] = "defer",
    ):
        """
        Args:
            policy: Confidence thresholds for escalation
            human_tool: Async human approval interface
            verifier: Multi-head verification ensemble
            breaker: Circuit breaker that turns the vote into a Gate
            fallback_action: What to do if human rejects or times out
        """
        self.policy = policy
        self.human_tool = human_tool
        self.verifier = verifier
        self.breaker = breaker
        self.fallback_action = fallback_action
    
    async def execute_with_oversight(
        self,
        run_id: UUID,
        proposal: ActionProposal,
        state: AgentState,
        context: str,
    ) -> dict:
        """
        Execute action autonomously if verified, else escalate.
        
        Decision Logic:
        1. Run multi-head verification.
        2. If consensus >= threshold: execute immediately.
        3. If low consensus: suspend and request human approval.
        4. If human rejects: apply fallback action.
        """
        # Gate 1: Multi-head verification
        should_escalate, consensus, reason = await self.verifier.verify(
            primary_output=proposal.action,
            context=context,
        )
        
        # Gate 2: Circuit breaker turns the vote into a Gate
        gate = self.breaker.gate(should_escalate)
        if gate is Gate.AUTONOMOUS:
            # Autonomous execution (fast path)
            return await self._execute_autonomous(
                proposal.action, state, consensus
            )
        if gate is Gate.SHED:
            # Queue at capacity: never reaches the fast path
            return await self._shed(proposal, state, reason)
        
        # Gate 3: Human escalation (slow path)
        return await self._execute_with_approval(
            run_id=run_id,
            proposal=proposal,
            state=state,
            escalation_reason=reason,
        )
    
    async def _execute_autonomous(
        self, action: dict, state: AgentState, consensus: float
    ) -> dict:
        """Execute action without human approval (high consensus path)."""
        # Log autonomous decision for audit trail
        state.audit_log.append({
            "type": "autonomous_execution",
            "action": action,
            "verification_consensus": consensus,
            "timestamp": datetime.now(UTC),
        })
        
        # Execute tool
        tool_name = action["tool"]
        tool_args = action["args"]
        result = await state.tools[tool_name].execute(**tool_args)
        
        return {"status": "executed", "result": result, "consensus": consensus}
    
    async def _shed(
        self, proposal: ActionProposal, state: AgentState, reason: str
    ) -> dict:
        """Apply the fallback instead of queuing for a human (overload path)."""
        state.audit_log.append({
            "type": "load_shed",
            "action": proposal.action,
            "fallback": self.fallback_action,
            "reason": reason,
            "timestamp": datetime.now(UTC),
        })
        return {"status": "shed", "fallback": self.fallback_action}
    
    async def _execute_with_approval(
        self,
        run_id: UUID,
        proposal: ActionProposal,
        state: AgentState,
        escalation_reason: str,
    ) -> dict:
        """Suspend workflow and request human approval (low consensus path)."""
        state_snapshot = state.model_dump()
        
        # Request approval
        metadata = await self.human_tool.ask_human(
            run_id=run_id,
            prompt=f"Verification failed: {escalation_reason}",
            proposal={
                "action": proposal.action,
                "confidence": proposal.confidence.model_dump(),
                "reasoning": proposal.reasoning,
            },
            state_snapshot=state_snapshot,
            ttl_hours=12,
        )
        
        # Workflow suspends here. Webhook will resume execution.
        # (In production, this would raise HumanInterruptSignal)
        
        # Placeholder for demonstration:
        return {"status": "pending_approval", "resume_token": metadata.resume_token}
# --8<-- [end:lst-7.25]

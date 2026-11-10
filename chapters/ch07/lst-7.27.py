# --8<-- [start:lst-7.27]
# src/agent/supervision/silent_auditor.py
from __future__ import annotations

import random
from datetime import datetime, UTC
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AuditSample(BaseModel):
    """A decision sampled for post-hoc human review."""
    sample_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    action_taken: dict
    result: dict
    timestamp: datetime
    sampled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class SilentAuditorSupervisor:
    """
    Post-hoc sampling for quality assurance. Full autonomy with deferred review.
    
    This pattern enforces L4 autonomy: the agent acts independently, and humans
    audit a statistical sample after the fact.
    
    WARNING: Only use for reversible actions or negligible error costs.
    """
    
    def __init__(
        self,
        sample_rate: float = 0.01,  # 1% of decisions
        audit_queue: Optional[list] = None,
        stratified_sampling: bool = True,  # Bias towards high-entropy cases
    ):
        """
        Args:
            sample_rate: Fraction of decisions to sample for review (0.0-1.0)
            audit_queue: External queue for audit samples (e.g., Redis, SQS)
            stratified_sampling: If True, over-sample uncertain decisions
        """
        self.sample_rate = sample_rate
        self.audit_queue = audit_queue or []
        self.stratified_sampling = stratified_sampling
    
    async def execute_autonomous(
        self,
        run_id: UUID,
        action: dict,
        state: AgentState,
        confidence: Optional[float] = None,
    ) -> dict:
        """
        Execute action immediately without human approval.
        Optionally sample for post-hoc review.
        
        Decision Logic:
        1. Execute action autonomously (no gate).
        2. With probability `sample_rate`, log decision to audit queue.
        3. Return result immediately (no blocking).
        """
        # Execute action (fast path, no gating)
        result = await self._execute_tool(action, state)
        
        # Stratified sampling: over-sample low-confidence decisions
        effective_sample_rate = self._compute_sample_probability(confidence)
        
        if random.random() < effective_sample_rate:
            audit_sample = AuditSample(
                run_id=run_id,
                action_taken=action,
                result=result,
                timestamp=datetime.now(UTC),
            )
            
            # Push to audit queue (async, non-blocking)
            await self._enqueue_audit(audit_sample)
        
        return result
    
    def _compute_sample_probability(self, confidence: Optional[float]) -> float:
        """
        Calculate effective sample rate using stratified sampling.
        
        Low-confidence decisions are over-sampled to maximize information gain.
        """
        if not self.stratified_sampling or confidence is None:
            return self.sample_rate
        
        # Inverse confidence weighting: sample rate = base_rate / confidence
        # This ensures low-confidence decisions (e.g., 0.5) are sampled 2x more
        # than high-confidence decisions (e.g., 1.0)
        stratified_rate = self.sample_rate / confidence if confidence > 0 else 1.0
        
        # Cap at 100% sampling
        return min(stratified_rate, 1.0)
    
    async def _execute_tool(self, action: dict, state: AgentState) -> dict:
        """Execute tool call without approval."""
        tool_name = action["tool"]
        tool_args = action["args"]
        result = await state.tools[tool_name].execute(**tool_args)
        return {"status": "executed", "result": result}
    
    async def _enqueue_audit(self, sample: AuditSample):
        """Push sample to audit queue for human review (non-blocking)."""
        # In production, this would publish to Redis/SQS/Kafka
        self.audit_queue.append(sample)
# --8<-- [end:lst-7.27]

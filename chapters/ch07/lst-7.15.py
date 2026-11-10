# --8<-- [start:lst-7.15]
# src/agent/storage/proposal_storage.py
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class PendingProposal(BaseModel):
    """Metadata for a pending human decision."""
    workflow_id: str
    notification_id: str
    created_at: datetime


class ProposalStorage(ABC):
    """
    Durable storage for human approval workflows.
    """
    
    @abstractmethod
    async def save_proposal(self, proposal: ProposalContext) -> None:
        """Persist a proposal awaiting human decision."""
        ...
    
    @abstractmethod
    async def get_pending_proposal(self, workflow_id: str) -> Optional[PendingProposal]:
        """Check if a proposal is pending notification."""
        ...
    
    @abstractmethod
    async def mark_pending(self, workflow_id: str, notification_id: str) -> bool:
        """Set the pending mark if absent; True only for the caller that set it."""
        ...
    
    @abstractmethod
    async def get_decision(
        self, proposal: ProposalContext
    ) -> Optional[HumanDecision]:
        """Retrieve the human's decision for this exact proposal if available."""
        ...
    
    @abstractmethod
    async def save_decision(
        self,
        tenant_id: str, workflow_id: str,
        decision: HumanDecision,
    ) -> None:
        """Record the human's decision once; the first write wins (called by webhook handler)."""
        ...


class RedisProposalStorage(ProposalStorage):
    """
    Redis-backed storage with TTL for time-bounded decisions.
    """
    
    def __init__(self, redis_client, ttl_seconds: int = 86400):  # keep >= the run's deadline
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    async def save_proposal(self, proposal: ProposalContext) -> None:
        key = f"proposal:{proposal.workflow_id}"
        if not await self.redis.set(
            key,
            proposal.model_dump_json(),
            ex=self.ttl,
            nx=True,
        ):
            raise ValueError("proposal already stored")
    
    async def get_pending_proposal(self, workflow_id: str) -> Optional[PendingProposal]:
        key = f"pending:{workflow_id}"
        data = await self.redis.get(key)
        if data:
            return PendingProposal.model_validate_json(data)
        return None
    
    async def mark_pending(self, workflow_id: str, notification_id: str) -> bool:
        key = f"pending:{workflow_id}"
        pending = PendingProposal(
            workflow_id=workflow_id,
            notification_id=notification_id,
            created_at=datetime.utcnow(),
        )
        return bool(await self.redis.set(key, pending.model_dump_json(), ex=self.ttl, nx=True))
    
    async def get_decision(self, proposal: ProposalContext) -> Optional[HumanDecision]:
        key = f"decision:{proposal.tenant_id}:{proposal.workflow_id}:{proposal.content_hash}"
        data = await self.redis.get(key)
        if data:
            return HumanDecision.model_validate_json(data)
        return None
    
    async def save_decision(
        self,
        tenant_id: str, workflow_id: str,
        decision: HumanDecision,
    ) -> None:
        key = f"decision:{tenant_id}:{workflow_id}:{decision.proposal_hash}"
        await self.redis.set(
            key,
            decision.model_dump_json(),
            ex=self.ttl,
            nx=True,  # set-if-absent: the first of deadline and click wins
        )
# --8<-- [end:lst-7.15]

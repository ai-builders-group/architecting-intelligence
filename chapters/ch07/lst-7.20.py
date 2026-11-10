# --8<-- [start:lst-7.20]
# src/agent/checkpoint.py
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, Field

class CheckpointMetadata(BaseModel):
    """Metadata for a suspended workflow checkpoint."""
    run_id: UUID
    checkpoint_id: UUID
    node_name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    resume_token: str = Field(exclude=True)  # Never serialize to logs

class AgentCheckpoint(BaseModel):
    """Full checkpoint state for durable suspension."""
    run_id: UUID
    checkpoint_id: UUID
    node_name: str
    state_snapshot: dict  # Serialized AgentState.model_dump()
    agent_version: str
    schema_version: str = "v1"
    tenant_id: Optional[str] = None  # set by verify_and_load
    approver_id: Optional[str] = None  # the named approver
    duplicate: bool = False  # True when this resume was already claimed
    result: Optional[dict] = None  # Recorded result, set on a duplicate

class CheckpointManager:
    """Postgres-backed durable state manager for agent workflows."""
    
    def __init__(self, db_pool: asyncpg.Pool, token_secret: str):
        """
        Args:
            db_pool: asyncpg connection pool
            token_secret: HMAC secret for signing resume tokens (from env var)
        """
        self.pool = db_pool
        self.secret = token_secret.encode()
    
    def _generate_resume_token(self, checkpoint_id: UUID) -> str:
        """
        Generate a cryptographically signed resume token.
        
        Format: {checkpoint_id}.{random_nonce}.{hmac_signature}
        This prevents token forgery—an attacker cannot craft a valid token
        without knowing the server secret.
        """
        nonce = secrets.token_urlsafe(16)
        payload = f"{checkpoint_id}.{nonce}"
        signature = hmac.new(
            self.secret, payload.encode(), hashlib.sha256
        ).hexdigest()[:16]
        return f"{payload}.{signature}"
    
    def _validate_resume_token(self, token: str) -> Optional[UUID]:
        """
        Validate and extract checkpoint_id from a resume token.
        
        Returns:
            checkpoint_id if valid, None if forged/expired
        """
        try:
            checkpoint_str, nonce, signature = token.split(".")
            checkpoint_id = UUID(checkpoint_str)
            
            # Recompute expected signature
            payload = f"{checkpoint_str}.{nonce}"
            expected_sig = hmac.new(
                self.secret, payload.encode(), hashlib.sha256
            ).hexdigest()[:16]
            
            if secrets.compare_digest(signature, expected_sig):
                return checkpoint_id
            return None
        except (ValueError, AttributeError, TypeError):
            return None
    
    async def save_checkpoint(
        self,
        run_id: UUID,
        node_name: str,
        state_snapshot: dict,
        agent_version: str,
        ttl_hours: Optional[int] = None,
        tenant_id: Optional[str] = None, approver_id: Optional[str] = None,
    ) -> CheckpointMetadata:
        """
        Persist agent state to Postgres and return resume metadata.
        
        Args:
            run_id: Workflow instance ID
            node_name: Current graph node (e.g., "wait_for_approval")
            state_snapshot: Serialized AgentState.model_dump()
            agent_version: Git SHA or package version
            ttl_hours: Optional expiration time (for ephemeral workflows)
        
        Returns:
            CheckpointMetadata including resume_token
        """
        checkpoint_id = uuid4()
        resume_token = self._generate_resume_token(checkpoint_id)
        token_hash = hashlib.sha256(resume_token.encode()).hexdigest()
        
        expires_at = None
        if ttl_hours:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_checkpoints (
                    run_id, checkpoint_id, node_name, state_snapshot,
                    resume_token_hash, agent_version, schema_version, expires_at,
                    tenant_id, approver_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                run_id,
                checkpoint_id,
                node_name,
                state_snapshot,  # Postgres JSONB handles dict automatically
                token_hash,
                agent_version,
                "v1",
                expires_at, tenant_id, approver_id,
            )
        
        return CheckpointMetadata(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            node_name=node_name,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            resume_token=resume_token,
        )
    
    async def verify_and_load(
        self, resume_token: str
    ) -> Optional[AgentCheckpoint]:
        """
        Verify a resume token and load its checkpoint. Writes nothing.
        
        The caller must authorize the caller against the run, then call
        claim(), so a caller who does not own the run never consumes the claim.
        
        Args:
            resume_token: The signed token from save_checkpoint()
        
        Returns:
            AgentCheckpoint if valid and not expired, None otherwise
        """
        checkpoint_id = self._validate_resume_token(resume_token)
        if not checkpoint_id:
            return None
        
        token_hash = hashlib.sha256(resume_token.encode()).hexdigest()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT run_id, checkpoint_id, node_name, state_snapshot,
                       tenant_id, approver_id,
                       agent_version, schema_version, expires_at
                FROM agent_checkpoints
                WHERE resume_token_hash = $1
                """,
                token_hash,
            )
            
            if not row:
                return None
            
            # Check expiration
            if row["expires_at"] and datetime.utcnow() > row["expires_at"]:
                return None
            
            return AgentCheckpoint(
                run_id=row["run_id"],
                checkpoint_id=row["checkpoint_id"],
                node_name=row["node_name"],
                state_snapshot=row["state_snapshot"],
                agent_version=row["agent_version"],
                schema_version=row["schema_version"],
                tenant_id=row["tenant_id"], approver_id=row["approver_id"],
            )
    
    async def claim(
        self, checkpoint_id: UUID, approver_id: str, decision: bool
    ) -> AgentCheckpoint:
        """
        Claim the resume once, after the caller has been authorized.
        
        Returns:
            The checkpoint to act on, or on a duplicate resume a checkpoint
            marked duplicate=True that carries the recorded result.
        """
        async with self.pool.acquire() as conn:
            # Claim the resume (idempotency boundary): exactly one caller wins
            status = await conn.execute(
                """
                UPDATE agent_checkpoints
                SET resumed_at = now(), resumed_by = COALESCE(resumed_by, $2),
                    decision = COALESCE(decision, $3)
                WHERE checkpoint_id = $1 AND (resumed_at IS NULL OR (
                    result IS NULL AND resumed_at < now() - interval '5 minutes'))
                """,
                checkpoint_id, approver_id, decision,
            )
            claimed = status.endswith(" 1")
            
            row = await conn.fetchrow(
                """
                SELECT run_id, checkpoint_id, node_name, state_snapshot,
                       agent_version, schema_version
                FROM agent_checkpoints
                WHERE checkpoint_id = $1
                """,
                checkpoint_id,
            )
            
            if claimed:
                return AgentCheckpoint(
                    run_id=row["run_id"],
                    checkpoint_id=row["checkpoint_id"],
                    node_name=row["node_name"],
                    state_snapshot=row["state_snapshot"],
                    agent_version=row["agent_version"],
                    schema_version=row["schema_version"],
                )
            
            # Duplicate resume: act on nothing, return the recorded result
            result = await conn.fetchval(
                "SELECT result FROM agent_checkpoints WHERE checkpoint_id = $1",
                checkpoint_id,
            )
            return AgentCheckpoint(
                run_id=row["run_id"],
                checkpoint_id=row["checkpoint_id"],
                node_name=row["node_name"],
                state_snapshot=row["state_snapshot"],
                agent_version=row["agent_version"],
                schema_version=row["schema_version"],
                duplicate=True,
                result=result,
            )
    
    async def record_result(self, checkpoint_id: UUID, result: dict) -> None:
        """Store the result of the resumed run so a duplicate resume can replay it."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agent_checkpoints
                SET result = $2
                WHERE checkpoint_id = $1
                """,
                checkpoint_id,
                result,
            )
    
    async def cleanup_expired(self) -> int:
        """
        Delete expired checkpoints (to be called by cron job).
        
        Returns:
            Number of rows deleted
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM agent_checkpoints
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """
            )
            # Extract row count from result string "DELETE N"
            return int(result.split()[-1])
# --8<-- [end:lst-7.20]

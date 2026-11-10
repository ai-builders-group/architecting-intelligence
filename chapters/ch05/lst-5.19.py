# --8<-- [start:lst-5.19]
# src/04_probabilistic_state_machine/checkpointers/postgres_checkpointer.py
import json
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

class StateMetadata(BaseModel):
    execution_id: str
    trace_id: str
    created_at: datetime
    updated_at: datetime

class AgentState(BaseModel):
    """Minimal state shape used by the checkpointer; extra fields are preserved."""
    model_config = ConfigDict(extra="allow")
    metadata: StateMetadata
    current_node: Optional[str] = None
    is_complete: bool = False
    requires_human: bool = False

class OptimisticLockError(Exception):
    """Raised when a concurrent write conflict is detected."""
    pass

class PostgresCheckpointer:
    """
    Durable state persistence using Postgres with optimistic locking.
    
    **Design Philosophy**: We treat state checkpointing with the same rigor
    as financial transaction logging. Postgres provides ACID guarantees that
    are non-negotiable for production agents.
    
    **Scaling Constraints**:
    - **Throughput Ceiling**: ~200 TPS per Postgres writer (network + WAL bottleneck).
    - **State Size Ceiling**: Comfortable up to 1MB per row; degrades beyond 10MB (TOAST overhead).
    - **Write Amplification**: Full-state UPSERT is O(N) in state size. For workflows >20 nodes,
      implement delta logging (see ADR-04.4).
    
    **When to Migrate Away**:
    - **>1K TPS**: Implement delta logging or partition to DynamoDB.
    - **>10K TPS**: Migrate to Cassandra or S3 + metadata index.
    - **State >10MB**: Partition history to separate table or blob store.
    
    **Durability Guarantee**: This checkpointer ensures state mutations are durable.
    It does NOT ensure side-effect/checkpoint atomicity (e.g., if a tool charges a credit
    card and the process crashes before save(), the charge is orphaned). See Chapter 6
    (Idempotency Keys, Saga Pattern) for solutions.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize the checkpointer.
        
        Args:
            connection_string: Postgres URI (e.g., "postgresql://user:pass@host:5432/db").
        
        **Production Note**: Use QueuePool (not NullPool) to amortize TLS handshake
        overhead (5-10ms). For >100 concurrent agents, deploy PgBouncer for connection
        multiplexing to prevent port exhaustion.
        """
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True  # Health check before use
        )
    
    def save(self, state: AgentState, expected_version: Optional[int] = None) -> int:
        """
        Persist the current state to Postgres with optimistic locking.
        
        **Optimistic Locking**: If `expected_version` is provided, the update only succeeds
        if the database version matches. This prevents lost updates when multiple processes
        checkpoint the same execution_id concurrently (e.g., in a distributed deployment).
        
        **Idempotency**: Calling save() multiple times with the same `execution_id`
        updates the existing row (UPSERT semantics).
        
        Args:
            state: The agent state to persist.
            expected_version: The version we expect the DB to have. If None, this is
                             the first save (INSERT). If provided, we enforce version
                             matching (UPDATE with WHERE clause).
        
        Returns:
            The new version number after successful save.
        
        Raises:
            OptimisticLockError: If version mismatch detected (concurrent write conflict).
        
        **Performance Note**: This method UPSERTs the FULL state blob. For workflows
        with large history logs (>1MB), this creates O(N²) write amplification over
        the lifetime of the agent. Mitigation strategies (choose based on scale):
        
        1. **Prune History** (see AgentState.prune_history()): Retain only last N entries.
        2. **Delta Logging**: Store only output_delta per node in a separate table.
           Reconstruct current state via `SELECT ... ORDER BY seq` on demand.
        3. **State Partitioning**: Move `history` to a one-to-many `agent_history` table.
           Checkpoint only scratchpad + flags on every node; batch-append history every 10 nodes.
        4. **Snapshot + Delta Hybrid**: Full checkpoint every 10 nodes; deltas in between.
           On load, fetch latest snapshot + replay tail (LSM-tree pattern).
        
        See ADR-04.4 for detailed trade-off analysis and implementation guidance.
        """
        # Update the timestamp
        state.metadata.updated_at = datetime.now(timezone.utc)
        
        # Serialize to JSON (Pydantic V2 API)
        state_json = state.model_dump(mode='json')
        
        # UPSERT query with optimistic locking
        if expected_version is None:
            # First save: initialize version to 1
            query = text("""
                INSERT INTO checkpoints (
                    execution_id, trace_id, created_at, updated_at, state,
                    current_node, is_complete, requires_human, version
                ) VALUES (
                    :execution_id, :trace_id, :created_at, :updated_at, :state,
                    :current_node, :is_complete, :requires_human, 1
                )
                ON CONFLICT (execution_id) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at,
                    state = EXCLUDED.state,
                    current_node = EXCLUDED.current_node,
                    is_complete = EXCLUDED.is_complete,
                    requires_human = EXCLUDED.requires_human,
                    version = checkpoints.version + 1
                RETURNING version
            """)
        else:
            # Subsequent save: enforce version check
            query = text("""
                UPDATE checkpoints SET
                    updated_at = :updated_at,
                    state = :state,
                    current_node = :current_node,
                    is_complete = :is_complete,
                    requires_human = :requires_human,
                    version = :new_version
                WHERE execution_id = :execution_id AND version = :expected_version
                RETURNING version
            """)
        
        with self.engine.connect() as conn:
            if expected_version is None:
                result = conn.execute(query, {
                    "execution_id": state.metadata.execution_id,
                    "trace_id": state.metadata.trace_id,
                    "created_at": state.metadata.created_at,
                    "updated_at": state.metadata.updated_at,
                    "state": json.dumps(state_json),
                    "current_node": state.current_node,
                    "is_complete": state.is_complete,
                    "requires_human": state.requires_human
                })
            else:
                result = conn.execute(query, {
                    "execution_id": state.metadata.execution_id,
                    "updated_at": state.metadata.updated_at,
                    "state": json.dumps(state_json),
                    "current_node": state.current_node,
                    "is_complete": state.is_complete,
                    "requires_human": state.requires_human,
                    "expected_version": expected_version,
                    "new_version": expected_version + 1
                })
                
                if result.rowcount == 0:
                    raise OptimisticLockError(
                        f"Version mismatch for execution_id="
                        f"{state.metadata.execution_id}. "
                        f"Expected version {expected_version}, "
                        f"but state was modified concurrently."
                    )
            
            conn.commit()
            new_version = result.fetchone()[0]
            return new_version
    
    def load(self, execution_id: str) -> Optional[AgentState]:
        """
        Retrieve a checkpoint by execution_id.
        
        Returns:
            The deserialized AgentState, or None if not found.
        """
        query = text("SELECT state FROM checkpoints WHERE execution_id = :execution_id")
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"execution_id": execution_id}).fetchone()
            
            if result is None:
                return None
            
            # Deserialize from JSONB
            state_dict = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            return AgentState(**state_dict)
    
    def list_pending_human_reviews(self) -> list[AgentState]:
        """
        Query all checkpoints where `requires_human = TRUE`.
        
        Use Case: Building a "Human Review Queue" UI.
        
        **Performance Note**: This query uses the B-tree partial index on `requires_human`.
        If you need to filter by deep JSONB fields (e.g., `state->'scratchpad'->>'risk_score' > 0.8`),
        you have two options:
        
        1. **Denormalize**: Add a `risk_score FLOAT` column and maintain it on write.
        2. **GIN Index**: Create `idx_checkpoints_state_gin` BUT accept 20-100ms write penalty.
        
        For write-heavy checkpointing, Option 1 is strongly preferred.
        
        Returns:
            List of states awaiting human intervention, ordered by urgency (oldest first).
        """
        query = text("""
            SELECT state
            FROM checkpoints
            WHERE requires_human = TRUE
            ORDER BY updated_at ASC
        """)
        
        with self.engine.connect() as conn:
            results = conn.execute(query).fetchall()
            return [
                AgentState(**json.loads(row[0]) if isinstance(row[0], str) else row[0])
                for row in results
            ]
# --8<-- [end:lst-5.19]

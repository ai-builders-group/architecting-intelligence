# --8<-- [start:lst-5.16]
# src/04_probabilistic_state_machine/schemas/agent_state.py
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime, timezone
from uuid import uuid4

class StateMetadata(BaseModel):
    """
    Observability and provenance metadata.
    """
    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for distributed tracing (OpenTelemetry)."
    )
    execution_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this specific execution run."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of state initialization (timezone-aware UTC)."
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of last state mutation (timezone-aware UTC)."
    )
    schema_version: int = Field(
        default=1,
        description="Schema version for migration tracking."
    )

class NodeExecution(BaseModel):
    """
    Immutable record of a single node execution.
    
    Design Principle: This is the "log entry" in our write-ahead log.
    Every field is immutable after creation.
    """
    node_id: str = Field(description="Name of the executed node.")
    node_type: Literal["stochastic", "deterministic", "human"] = Field(
        description="Category of the node."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    # Input/Output
    input_snapshot: Dict[str, Any] = Field(
        description="The state visible to this node at execution time."
    )
    output_delta: Dict[str, Any] = Field(
        description="The state mutations produced by this node."
    )
    
    # Metadata
    duration_ms: Optional[float] = Field(
        default=None,
        description="Wall-clock execution time."
    )
    token_count: Optional[int] = Field(
        default=None,
        description="Tokens consumed (for LLM nodes)."
    )
    cost_usd: Optional[float] = Field(
        default=None,
        description="Monetary cost (for paid APIs)."
    )
    
    # Error Handling
    error: Optional[str] = Field(
        default=None,
        description="Error message if the node failed."
    )

class AgentState(BaseModel):
    """
    The canonical state schema for graph-based agents.
    
    Architecture:
    - Immutable: `metadata`, `history` (append-only).
    - Mutable: `scratchpad`, `flags`.
    
    Serialization:
    - This schema is JSON-serializable for Postgres JSONB storage.
    - Use `model_dump(mode='json')` for serialization.
    
    **State Bloat Warning**: The `history` field grows linearly with node count.
    For self-correction loops exceeding 20 turns or workflows with >50 nodes,
    implement pruning (retain only last N entries) OR partition history to a
    separate table/blob store. Postgres TOAST can handle 1GB rows but degrades
    read performance beyond 10MB compressed.
    """
    
    model_config = ConfigDict(
        validate_assignment=True,  # Enforce validation on field updates
        use_enum_values=True
    )
    
    # === IMMUTABLE SECTION ===
    
    metadata: StateMetadata = Field(
        default_factory=StateMetadata,
        description="Provenance and tracing metadata."
    )
    
    history: List[NodeExecution] = Field(
        default_factory=list,
        description="Append-only log of all node executions."
    )
    
    # === MUTABLE SECTION ===
    
    scratchpad: Dict[str, Any] = Field(
        default_factory=dict,
        description="Temporary working memory. Cleared or pruned between phases."
    )
    
    # === CONTROL FLAGS ===
    
    current_node: Optional[str] = Field(
        default=None,
        description="The active node (for resumption after checkpointing)."
    )
    
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries for the current node."
    )
    
    is_complete: bool = Field(
        default=False,
        description="Terminal flag: True when the agent reaches END."
    )
    
    requires_human: bool = Field(
        default=False,
        description="Escalation flag: True when human intervention is needed."
    )
    
    # === RESOURCE TRACKING ===
    
    total_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    
    # === PRIORITY (Tier Priority Stack) ===
    
    priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="T1 (security) through T5 (background). Higher-tier events preempt lower-tier executions."
    )

    def append_execution(self, node_exec: NodeExecution) -> None:
        """
        Append a node execution to the history.
        
        This is the ONLY way to modify the history list.
        Side Effect: Updates `metadata.updated_at`.
        
        **State Bloat Mitigation**: For long-running workflows, consider
        pruning history after N entries (retain only tail) or offloading
        to a separate `agent_history` table.
        """
        self.history.append(node_exec)
        self.metadata.updated_at = datetime.now(timezone.utc)
    
    def get_last_execution(self, node_id: str) -> Optional[NodeExecution]:
        """
        Retrieve the most recent execution of a specific node.
        
        Use Case: Checking if a node has already run (idempotency).
        """
        for exec in reversed(self.history):
            if exec.node_id == node_id:
                return exec
        return None
    
    def prune_history(self, keep_last_n: int = 10) -> None:
        """
        Prune the history log to retain only the most recent N entries.
        
        **Use Case**: Self-correction loops with >50 iterations. Prevents
        state bloat exceeding Postgres TOAST comfortable ceiling (~10MB).
        
        **Warning**: This destroys audit trail completeness. Only use for
        non-compliance workloads. For compliance (Chapter 20), archive full
        history to S3 before pruning.
        """
        if len(self.history) > keep_last_n:
            archived = self.history[:-keep_last_n]
            self.history = self.history[-keep_last_n:]
            # TODO: Implement S3 archival for `archived` entries
            self.metadata.updated_at = datetime.now(timezone.utc)
# --8<-- [end:lst-5.16]

# --8<-- [start:lst-5.17]
# src/04_probabilistic_state_machine/reducers/state_reducer.py
from typing import Dict, Any, Callable
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field

# Minimal stand-ins for the schema of Listing 5.14 (only the fields the reducers touch)
class StateMetadata(BaseModel):
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NodeExecution(BaseModel):
    node_id: str
    node_type: str
    input_snapshot: Dict[str, Any] = Field(default_factory=dict)
    output_delta: Dict[str, Any] = Field(default_factory=dict)

class AgentState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    metadata: StateMetadata = Field(default_factory=StateMetadata)
    history: list[NodeExecution] = Field(default_factory=list)
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    current_node: str | None = None
    retry_count: int = Field(default=0, ge=0)
    is_complete: bool = False
    requires_human: bool = False

    def append_execution(self, node_exec: NodeExecution) -> None:
        self.history.append(node_exec)
        self.metadata.updated_at = datetime.now(timezone.utc)

StateReducer = Callable[[AgentState, Dict[str, Any]], AgentState]

def append_reducer(state: AgentState, delta: Dict[str, Any]) -> AgentState:
    """
    Append Semantics: Merge list fields by appending.
    
    **Functional Purity**: Returns a NEW state object; does not mutate input.
    
    **Performance Warning**: Uses model_copy(deep=True), which is O(N) for state size.
    For high-throughput systems (>100 TPS), profile this function. If it exceeds 10% of
    wall-clock latency, migrate to Copy-on-Write (shallow copy + selective deep for mutated
    fields) or persistent data structures (pyrsistent).
    
    Use Case: Adding entries to `history`, `critiques`, etc.
    
    Example:
        delta = {"history": [new_execution]}
        → new_state.history = old_state.history + [new_execution]
    """
    # Deep copy to ensure immutability
    new_state = state.model_copy(deep=True)
    
    for key, value in delta.items():
        if key == "history":
            # Use the encapsulated method (which updates `updated_at`)
            for node_exec in value:
                if isinstance(node_exec, NodeExecution):
                    new_state.append_execution(node_exec)
                else:
                    # Convert dict to NodeExecution if needed
                    new_state.append_execution(NodeExecution(**node_exec))
            continue
        elif key in new_state.scratchpad:
            # Append to scratchpad lists
            if isinstance(new_state.scratchpad[key], list) and isinstance(value, list):
                new_state.scratchpad[key] = new_state.scratchpad[key] + value
            else:
                new_state.scratchpad[key] = value
        elif hasattr(new_state, key):
            current = getattr(new_state, key)
            if isinstance(current, list) and isinstance(value, list):
                setattr(new_state, key, current + value)
            else:
                setattr(new_state, key, value)
    
    return new_state

def overwrite_reducer(state: AgentState, delta: Dict[str, Any]) -> AgentState:
    """
    Overwrite Semantics: Replace field values.
    
    **Functional Purity**: Returns a NEW state object; does not mutate input.
    
    Use Case: Updating `current_node`, `is_complete`, scalar fields.
    
    Example:
        delta = {"current_node": "validator"}
        → new_state.current_node = "validator"
    """
    new_state = state.model_copy(deep=True)
    
    for key, value in delta.items():
        if key in ["metadata", "history"]:
            # Immutable fields: reject overwrites
            raise ValueError(f"Cannot overwrite immutable field: {key}")
        elif key == "scratchpad":
            # Shallow merge for scratchpad
            new_state.scratchpad = {**new_state.scratchpad, **value}
        elif hasattr(new_state, key):
            setattr(new_state, key, value)
        else:
            # Unknown field: store in scratchpad
            new_state.scratchpad[key] = value
    
    # Update timestamp
    new_state.metadata.updated_at = datetime.now(timezone.utc)
    
    return new_state

# Registry: Map field names to reducers
REDUCER_REGISTRY: Dict[str, StateReducer] = {
    "history": append_reducer,
    "critiques": append_reducer,
    "current_node": overwrite_reducer,
    "retry_count": overwrite_reducer,
    "is_complete": overwrite_reducer,
    "requires_human": overwrite_reducer,
    "scratchpad": overwrite_reducer,
}

def apply_delta(state: AgentState, delta: Dict[str, Any]) -> AgentState:
    """
    Apply a state update using the appropriate reducer for each field.
    
    **Functional Purity**: Returns a NEW state object.
    
    Args:
        state: The current state.
        delta: A dictionary of field updates.
    
    Returns:
        The updated state (a new object, not a mutation).
    """
    new_state = state
    for key, value in delta.items():
        reducer = REDUCER_REGISTRY.get(key, overwrite_reducer)
        new_state = reducer(new_state, {key: value})
    
    return new_state
# --8<-- [end:lst-5.17]

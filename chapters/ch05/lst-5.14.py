# --8<-- [start:lst-5.14]
# src/04_probabilistic_state_machine/nodes/reflection_terminator.py
from __future__ import annotations
from typing import Literal

def reflection_terminator_node(
    state: ReflectionState,
) -> Literal["output", "generator", "dlq"]:
    """
    Conditional routing for the Reflexion loop.
    
    Returns:
        The ID of the next node.
    """
    
    # Case 1: Success
    if state.validated_result is not None:
        return "output"
    
    # Case 2: Retry budget exhausted
    if state.attempt >= state.max_retries:
        return "dlq"
    
    # Case 3: Retry
    return "generator"


def dlq_node(state: ReflectionState) -> dict:
    # The router is a read-only path function, so the flag is set here, as a returned state update.
    return {"requires_human": True}
# --8<-- [end:lst-5.14]

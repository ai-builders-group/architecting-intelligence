# --8<-- [start:lst-5.3]
# src/04_probabilistic_state_machine/routing/conditional_edge.py
from dataclasses import dataclass
from typing import Literal


@dataclass
class ContractAnalysisState:
    validated: bool = False
    retry_count: int = 0
    requires_human: bool = False
    parties: tuple = ()
    terms: tuple = ()


def observe_validation(state: ContractAnalysisState) -> dict:
    """
    Observer node: validate the extraction and RETURN the state update.

    A node changes state only through the update it returns.
    """
    if state.parties and state.terms:
        return {"validated": True}
    n = state.retry_count + 1
    return {"validated": False, "retry_count": n, "requires_human": n >= 3}


def route_after_validation(
    state: ContractAnalysisState
) -> Literal["analyze", "extract", "human_review"]:
    """
    Conditional edge: If validation passes, proceed to analysis.
    Otherwise, retry extraction.
    
    Implements retry budget with DLQ escalation.
    Read-only: it writes no state (LangGraph discards edge writes).
    """
    if state.validated:
        return "analyze"
    else:
        # Check retry budget
        if state.retry_count >= 3:
            # Escalate to human after 3 failures
            return "human_review"
        else:
            return "extract"
# --8<-- [end:lst-5.3]

# --8<-- [start:lst-5.15]
# src/04_probabilistic_state_machine/graphs/reflexion_graph.py
from typing import Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

class ReflectionState(BaseModel):
    task_description: str = ""
    attempt: int = 0
    max_retries: int = 3
    validated_result: Optional[str] = None
    critiques: list[str] = Field(default_factory=list)
    requires_human: bool = False

def reflection_generator_node(state):
    return {"attempt": state.attempt + 1}

def reflection_evaluator_node(state):
    return {"critiques": state.critiques + [f"attempt {state.attempt} rejected"]}

def reflection_terminator_node(state):
    if state.validated_result is not None:
        return "output"
    if state.attempt >= state.max_retries:
        return "dlq"
    return "generator"

def dlq_node(state):
    return {"requires_human": True}

def build_reflexion_graph() -> StateGraph:
    """
    Construct the Multi-Turn Reflexion graph.
    
    Topology:
        Generator -> Evaluator -> Terminator
            ↑                          |
            |__________________________|
                  (critique feedback)
    
    Returns:
        A compiled LangGraph StateGraph.
    """
    
    graph = StateGraph(ReflectionState)
    
    # Add nodes
    graph.add_node("generator", reflection_generator_node)
    graph.add_node("evaluator", reflection_evaluator_node)
    graph.add_node("dlq", dlq_node)
    
    # Add edges
    graph.add_edge("generator", "evaluator")
    
    # Add conditional edge
    graph.add_conditional_edges(
        "evaluator",
        reflection_terminator_node,
        {
            "output": END,
            "generator": "generator",
            "dlq": "dlq"
        }
    )
    
    # Set entry point
    graph.set_entry_point("generator")
    
    return graph.compile()
# --8<-- [end:lst-5.15]

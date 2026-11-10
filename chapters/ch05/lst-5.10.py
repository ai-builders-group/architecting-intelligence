# --8<-- [start:lst-5.10]
# src/04_probabilistic_state_machine/graphs/react_graph.py
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

@dataclass
class AgenticState:
    is_complete: bool = False
    retry_count: int = 0
    max_retries: int = 3

def planner_node(state: AgenticState) -> dict:
    return {}

def actor_node(state: AgenticState) -> dict:
    return {}

def observer_node(state: AgenticState) -> dict:
    return {}

def critic_node(state: AgenticState) -> dict:
    retry_count = state.retry_count + 1
    return {"retry_count": retry_count, "is_complete": retry_count >= 2}

def route_after_critique(state: AgenticState) -> str:
    if state.is_complete:
        return "output"
    if state.retry_count >= state.max_retries:
        return "dlq"
    return "planner"

def build_react_graph() -> CompiledStateGraph:
    """
    Construct the Reflection-Enhanced ReAct graph.
    
    Topology:
        Planner -> Actor -> Observer -> Critic -> route_after_critique
                      ↑                              |
                      |______________________________|
                                (retry loop)
    
    Returns:
        A compiled LangGraph StateGraph ready for execution.
    """
    
    graph = StateGraph(AgenticState)
    
    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("observer", observer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("dlq", lambda state: {})  # No-op; human takes over
    
    # Add edges (unconditional)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "observer")
    graph.add_edge("observer", "critic")
    
    # Add conditional edge (the decision point)
    graph.add_conditional_edges(
        "critic",
        route_after_critique,  # The routing function
        {
            "output": END,     # Terminal node
            "planner": "planner",  # Retry loop
            "dlq": "dlq"       # Human escalation
        }
    )
    graph.add_edge("dlq", END)
    
    return graph.compile()

# Usage
react_graph = build_react_graph()

# Execute
final_state = react_graph.invoke(AgenticState())
print(final_state["is_complete"])
# --8<-- [end:lst-5.10]

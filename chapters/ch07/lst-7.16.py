# --8<-- [start:lst-7.16]
# src/agent/graph/human_node.py
from __future__ import annotations
from typing import Any
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import AsyncPostgresSaver


class HumanInterruptSignal(Exception):
    """Control signal: suspend the run until a human decides."""


class ProposalContext(BaseModel):
    action: Any
    reasoning: Any
    retrieved_docs: Any
    confidence_signals: Any
    estimated_cost: Any
    risk_level: Any
    workflow_id: str
    tenant_id: str
    content_hash: str


def create_human_approval_node(human_tool: HumanTool):
    """
    Factory for creating a human approval node in the graph.
    """
    
    async def human_approval_node(state: AgentState) -> dict:
        """
        A graph node that pauses execution and requests human judgment.
        
        This node:
        1. Constructs the proposal from current state
        2. Calls the HumanTool (may raise HumanInterruptSignal)
        3. Updates state with the human's decision
        4. Routes to next node based on approval/rejection
        
        If HumanInterruptSignal is raised, the graph executor:
        - Catches it at the top level
        - Checkpoints current state to durable storage
        - Terminates this execution
        - Waits for webhook to trigger resumption
        """
        # Build the proposal from agent's current reasoning
        proposal = ProposalContext(
            action=state.proposed_action,
            reasoning=state.reasoning_trace,
            retrieved_docs=state.context,
            confidence_signals=state.confidence_signals,
            estimated_cost=state.estimated_cost,
            risk_level=state.risk_level,
            workflow_id=state.run_id,
            tenant_id=state.tenant_id, content_hash=state.content_hash,
        )
        
        try:
            # Invoke the human (this may raise HumanInterruptSignal)
            decision = await human_tool.execute(proposal)
            
            # If we reach here, the human has responded
            
            # Log for observability (Chapter 31)
            state.trace.append({
                "node": "human_approval",
                "decision": decision.model_dump(),
                "timestamp": decision.decided_at.isoformat(),
            })
            
            return {"human_decision": decision, "approved": decision.approved}
        
        except HumanInterruptSignal as interrupt:
            # This is NOT an error—it's a control signal
            # The run stops; resume from the last checkpoint
            raise interrupt
    
    return human_approval_node


# Usage in graph construction
def build_graph(human_tool: HumanTool, pool) -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reasoning_node)
    graph.add_node("human_approval", create_human_approval_node(human_tool))
    graph.add_node("execute", execution_node)
    
    # Conditional edge based on confidence
    def should_escalate(state: AgentState) -> str:
        if should_escalate_logic(state.confidence_signals):
            return "human_approval"
        else:
            return "execute"
    
    graph.add_conditional_edges(
        "reason",
        should_escalate,
        {
            "human_approval": "human_approval",
            "execute": "execute",
        },
    )
    
    # After human approval, route based on decision
    def route_after_human(state: AgentState) -> str:
        if state.approved:
            return "execute"
        else:
            return END  # Terminate if rejected
    
    graph.add_conditional_edges(
        "human_approval",
        route_after_human,
        {
            "execute": "execute",
            END: END,
        },
    )
    
    graph.set_entry_point("retrieve")
    
    # CRITICAL: Use a durable checkpointer (not MemorySaver in production)
    # interrupt_before is not needed: the signal ends the invoke call, and the
    # caller treats that as the pause
    checkpointer = AsyncPostgresSaver(
        pool)  # durable: checkpoints survive a restart
    return graph.compile(checkpointer=checkpointer)
# --8<-- [end:lst-7.16]

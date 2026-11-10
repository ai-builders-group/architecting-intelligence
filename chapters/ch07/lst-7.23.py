# --8<-- [start:lst-7.23]
# src/agent/supervision/copilot.py
from __future__ import annotations

from typing import Optional
from uuid import UUID

class CoPilotSupervisor:
    """
    100% human verification. Every action requires pre-approval.
    
    This pattern enforces L1 autonomy: the agent proposes, the human decides.
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        human_tool: AsyncHumanTool,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.human_tool = human_tool
    
    async def execute_action(
        self,
        run_id: UUID,
        action_plan: dict,
        state: AgentState,
    ) -> Optional[dict]:
        """
        Propose an action and block execution until human approves.
        
        Args:
            run_id: Workflow instance ID
            action_plan: Structured plan (e.g., {"tool": "send_email", "args": {...}})
            state: Current agent state
        
        Returns:
            Execution result if approved, None if rejected
        """
        # Serialize state for durable suspension
        state_snapshot = state.model_dump()
        
        # Suspend workflow and request human approval
        metadata = await self.human_tool.ask_human(
            run_id=run_id,
            prompt=f"Approve action: {action_plan['tool']}",
            proposal=action_plan,
            state_snapshot=state_snapshot,
            ttl_hours=24,  # Escalations expire after 24 hours
        )
# --8<-- [end:lst-7.23]

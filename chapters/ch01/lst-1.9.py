# --8<-- [start:lst-1.9]
# src/agents/l1_copilot.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from enum import Enum

class ToolCall(BaseModel):
    """A proposed action awaiting approval."""
    tool_name: str = Field(description="The tool to be invoked")
    parameters: Dict[str, Any] = Field(description="Structured tool parameters")
    reasoning: str = Field(description="Why this action was proposed")
    impact_summary: str = Field(description="Human-readable consequence")
    reversible: bool = Field(default=False, description="Can this be undone?")

class ApprovalDecision(str, Enum):
    """Human decision on the proposed action."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"

class ApprovalResult(BaseModel):
    """The result of human review."""
    decision: ApprovalDecision
    modified_parameters: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None

class ExecutionResult(BaseModel):
    """The result of executing an approved action."""
    status: str = Field(description="success | rejected | pending_modification")
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SafetyCheck(BaseModel):
    """Verdict of the safety gate."""
    safe: bool

class L1Agent:
    """
    A copilot agent that proposes actions but requires human approval.
    """
    def propose_action(self, user_intent: str) -> ToolCall:
        """
        Generate a proposed action based on user intent.
        
        Args:
            user_intent: Natural language description of desired action
            
        Returns:
            ToolCall object for human review
        """
        # AI reasons about the intent and proposes a tool call
        return ToolCall(
            tool_name="update_user_email",
            parameters={"user_id": 12345, "email": "new@example.com"},
            reasoning="User requested email change in support ticket #789",
            impact_summary="Updates email for 1 user. Reversible via audit log.",
            reversible=True
        )
    
    def execute_if_approved(
        self, 
        tool_call: ToolCall, 
        approval: ApprovalResult
    ) -> ExecutionResult:
        """
        Execute the tool only if human approves.
        
        Args:
            tool_call: The proposed action
            approval: Human decision
            
        Returns:
            ExecutionResult with status and outcome
        """
        if approval.decision == ApprovalDecision.APPROVE:
            result = self._execute_tool(tool_call)
            return ExecutionResult(status="success", result=result)
        elif approval.decision == ApprovalDecision.REJECT:
            return ExecutionResult(
                status="rejected", 
                error=approval.rejection_reason or "User declined"
            )
        elif approval.decision == ApprovalDecision.MODIFY:
            if approval.modified_parameters:
                modified_call = ToolCall.model_validate(
                    {**tool_call.model_dump(),
                     "parameters": approval.modified_parameters}
                )
                if not self.is_safe(modified_call).safe:
                    return ExecutionResult(
                        status="rejected",
                        error="Modified call failed the safety check"
                    )
                result = self._execute_tool(modified_call)
                return ExecutionResult(status="success", result=result)
            return ExecutionResult(status="pending_modification")
        else:
            return ExecutionResult(
                status="rejected", error="Unrecognized decision"
            )
    
    def is_safe(self, tool_call: ToolCall) -> SafetyCheck:
        """Illustrative stand-in for the safety gate: only a reversible call passes."""
        return SafetyCheck(safe=tool_call.reversible)
    

    def _execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute the tool in a sandboxed environment."""
        return {"rows_affected": 1, "execution_time_ms": 45}
# --8<-- [end:lst-1.9]

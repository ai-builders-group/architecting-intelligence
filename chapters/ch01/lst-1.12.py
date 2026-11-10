# --8<-- [start:lst-1.12]
# src/agents/l4_autonomous.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

class Task(BaseModel):
    """Strictly typed task definition."""
    tool: str = Field(description="The tool to execute")
    parameters: Dict[str, Any] = Field(description="Tool parameters")
    max_retries: int = Field(default=3, ge=1, le=10)

class CorrectionStep(BaseModel):
    """A self-correction attempt."""
    error: str = Field(description="The error that triggered correction")
    strategy: Literal["retry", "alternative_tool", "fallback", "abort"]
    modified_parameters: Optional[Dict[str, Any]] = None
    result: str = Field(description="Outcome of the correction attempt")

class TransientToolError(Exception):
    """429, 5xx and timeouts: the only errors a retry may repeat."""

def error_code(e: Exception) -> str:
    """A sanitized code for the model; never str(e)."""
    return type(e).__name__

class ExecutionResult(BaseModel):
    """Result of autonomous execution."""
    status: Literal["success", "aborted", "failed_safe"]
    result: Optional[Dict[str, Any]] = None
    corrections: List[CorrectionStep] = Field(default_factory=list)
    total_attempts: int = Field(ge=1)

class L4Agent:
    """
    An autonomous agent with self-correction.
    No human escalation - fails to safe state.
    """
    
    MAX_RETRIES = 3
    
    def execute_with_correction(self, task: Task) -> ExecutionResult:
        """
        Execute task with self-correction loop.
        
        Args:
            task: The task to execute (strictly typed)
            
        Returns:
            ExecutionResult with status and correction history
        """
        attempts = 0
        corrections: List[CorrectionStep] = []
        current_task = task
        
        while attempts < min(task.max_retries, self.MAX_RETRIES):
            attempts += 1
            try:
                result = self._execute_tool(current_task)
                return ExecutionResult(
                    status="success",
                    result=result,
                    corrections=corrections,
                    total_attempts=attempts
                )
            except TransientToolError as e:
                correction = self._self_correct(
                    task=current_task,
                    error=error_code(e),
                    attempt=attempts
                )
                corrections.append(correction)
                
                if correction.strategy == "abort":
                    return ExecutionResult(
                        status="aborted",
                        corrections=corrections,
                        total_attempts=attempts
                    )
                
                # Modify task based on correction strategy
                current_task = self._apply_correction(current_task, correction)
        
        # Max retries exceeded - fail to safe state
        return ExecutionResult(
            status="failed_safe",
            corrections=corrections,
            total_attempts=attempts
        )
    
    def _self_correct(
        self, 
        task: Task, 
        error: str, 
        attempt: int
    ) -> CorrectionStep:
        """
        Determine correction strategy based on error type.
        
        Args:
            task: The task that failed
            error: The error message
            attempt: Current attempt number
            
        Returns:
            CorrectionStep with strategy and modifications
        """
        error_lower = error.lower()
        
        if "timeout" in error_lower:
            return CorrectionStep(
                error=error,
                strategy="retry",
                modified_parameters={"timeout_seconds": 30 * (2 ** attempt)},
                result=f"Retry with exponential backoff (attempt {attempt})"
            )
        elif "permission" in error_lower or "unauthorized" in error_lower:
            return CorrectionStep(
                error=error,
                strategy="abort",
                result="Insufficient permissions - cannot self-correct"
            )
        elif "not found" in error_lower:
            return CorrectionStep(
                error=error,
                strategy="alternative_tool",
                modified_parameters={"tool": f"fallback_{task.tool}"},
                result="Attempting alternative execution path"
            )
        else:
            if attempt < 3:
                return CorrectionStep(
                    error=error,
                    strategy="retry",
                    result=f"Generic retry (attempt {attempt})"
                )
            else:
                return CorrectionStep(
                    error=error,
                    strategy="fallback",
                    result="Falling back to safe default"
                )
    
    def _execute_tool(self, task: Task) -> Dict[str, Any]:
        """Execute the tool. Raises exception on failure."""
        # Mock execution
        return {"output": "success", "execution_time_ms": 150}
    
    def _apply_correction(self, task: Task, correction: CorrectionStep) -> Task:
        """Apply correction strategy to task."""
        if correction.modified_parameters:
            # Merge modified parameters
            updated_params = {**task.parameters, **correction.modified_parameters}
            return task.model_copy(update={"parameters": updated_params})
        return task
# --8<-- [end:lst-1.12]

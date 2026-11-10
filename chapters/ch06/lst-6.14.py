# --8<-- [start:lst-6.14]
# architecting-intelligence/src/06_durable/saga_orchestrator.py
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError
from pydantic import BaseModel, Field
from typing import Callable, Any, Optional
from datetime import timedelta
from enum import Enum
import structlog

logger = structlog.get_logger()

class SagaStepStatus(str, Enum):
    """Status of a Saga step."""
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    COMPENSATED = "compensated"
    FAILED = "failed"

class SagaStep(BaseModel):
    """Definition of a single step in a Saga."""
    name: str = Field(description="Human-readable step identifier")
    forward_activity: str = Field(description="Activity name for forward transaction")
    compensate_activity: str = Field(description="Activity name for compensation")
    forward_args: dict[str, Any] = Field(default_factory=dict)
    compensate_args: dict[str, Any] = Field(default_factory=dict)
    timeout: timedelta = Field(default=timedelta(seconds=30))
    status: SagaStepStatus = Field(default=SagaStepStatus.PENDING)
    result: Optional[Any] = Field(default=None, description="Forward transaction result")

class SagaDefinition(BaseModel):
    """Complete Saga specification."""
    saga_id: str
    steps: list[SagaStep]
    
    def get_completed_steps(self) -> list[SagaStep]:
        """Return steps that need compensation if Saga fails."""
        return [s for s in self.steps if s.status in (
            SagaStepStatus.COMPLETED, SagaStepStatus.IN_FLIGHT)]

@workflow.defn
class SagaOrchestrator:
    """
    Orchestration-based Saga implementation.
    
    Executes forward transactions sequentially. On failure, compensates
    all completed steps in reverse order.
    """
    
    def __init__(self) -> None:
        self.saga: Optional[SagaDefinition] = None
    
    @workflow.run
    async def run(self, saga: SagaDefinition) -> dict[str, Any]:
        """Execute the Saga with automatic compensation on failure."""
        self.saga = saga
        
        logger.info(
            "saga_started",
            saga_id=saga.saga_id,
            num_steps=len(saga.steps)
        )
        
        try:
            # Forward path: execute each transaction
            for i, step in enumerate(saga.steps):
                logger.info(
                    "saga_step_started",
                    saga_id=saga.saga_id,
                    step_name=step.name,
                    step_index=i
                )
                
                # Compensate this step too if it fails midway
                step.status = SagaStepStatus.IN_FLIGHT
                # Execute forward transaction as a Temporal Activity
                result = await workflow.execute_activity(
                    step.forward_activity,
                    args=[step.forward_args],
                    start_to_close_timeout=step.timeout,
                    retry_policy=RetryPolicy(
                        maximum_attempts=3,
                        initial_interval=timedelta(seconds=1),
                    )
                )
                
                # Mark step as completed (persisted in workflow state)
                step.status = SagaStepStatus.COMPLETED
                step.result = result
                
                logger.info(
                    "saga_step_completed",
                    saga_id=saga.saga_id,
                    step_name=step.name,
                    result=result
                )
            
            # All steps succeeded
            return {
                "status": "success",
                "results": [s.result for s in saga.steps]
            }
            
        except Exception as e:
            # Failure detected: initiate compensation
            logger.error(
                "saga_failed",
                saga_id=saga.saga_id,
                error=str(e),
                completed_steps=len(saga.get_completed_steps())
            )
            
            await self._compensate()
            
            return {
                "status": "compensated",
                "error": str(e),
                "compensated_steps": [s.name for s in saga.get_completed_steps()]
            }
    
    async def _compensate(self) -> None:
        """Execute compensation logic in reverse order."""
        completed = self.saga.get_completed_steps()
        
        # Reverse order: compensate most recent step first
        for step in reversed(completed):
            logger.info(
                "saga_compensating",
                saga_id=self.saga.saga_id,
                step_name=step.name
            )
            
            try:
                await workflow.execute_activity(
                    step.compensate_activity,
                    args=[step.compensate_args],
                    start_to_close_timeout=step.timeout,
                    retry_policy=RetryPolicy(
                        maximum_attempts=5,  # Aggressive retry for compensation
                        initial_interval=timedelta(seconds=2),
                        backoff_coefficient=2.0,
                    )
                )
                
                step.status = SagaStepStatus.COMPENSATED
                
                logger.info(
                    "saga_step_compensated",
                    saga_id=self.saga.saga_id,
                    step_name=step.name
                )
                
            except ActivityError as comp_error:
                # Compensation failed: mark as failed and continue
                # (Do not halt compensation of other steps)
                step.status = SagaStepStatus.FAILED
                
                logger.critical(
                    "saga_compensation_failed",
                    saga_id=self.saga.saga_id,
                    step_name=step.name,
                    error=str(comp_error)
                )
                
                # Escalate to human for manual intervention
                await workflow.execute_activity(
                    "escalate_to_human",
                    args=[{
                        "saga_id": self.saga.saga_id,
                        "failed_step": step.name,
                        "error": str(comp_error),
                        "context": step.compensate_args
                    }],
                    start_to_close_timeout=timedelta(seconds=10)
                )
                continue


# --8<-- [end:lst-6.14]

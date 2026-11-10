# --8<-- [start:lst-6.5]
# architecting-intelligence/src/06_temporal/workflow_pattern.py
from temporalio import workflow
from temporalio.common import RetryPolicy
from datetime import timedelta
from typing import Optional

@workflow.defn
class TransactionApprovalWorkflow:
    @workflow.run
    async def run(self, transaction_id: str, amount: float, model: str) -> str:
        """
        Deterministic orchestration logic.
        
        All non-deterministic operations are delegated to Activities.
        """
        # Step 1: Get risk score (Activity)
        risk_score = await workflow.execute_activity(
            "call_risk_api",
            transaction_id,
            start_to_close_timeout=timedelta(seconds=10),
            schedule_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                non_retryable_error_types=["InvalidTransaction"],
            )
        )
        
        # Step 2: Deterministic decision (Workflow)
        if risk_score > 0.7:
            # High risk: get LLM justification (Activity)
            justification = await workflow.execute_activity(
                "invoke_llm",
                args=[
                    f"Explain why a transaction with risk score {risk_score} "
                    "should be flagged.",
                    model,
                ],
                start_to_close_timeout=timedelta(seconds=30)
            )
            
            # Wait for human approval (Workflow signal)
            await workflow.wait_condition(
                lambda: self.human_decision is not None,
                timeout=timedelta(hours=24)
            )
            approved = self.human_decision
            
            if approved:
                return f"Approved by human. Reason: {justification}"
            else:
                return "Rejected by human."
        else:
            # Low risk: auto-approve (Deterministic)
            return f"Auto-approved (risk score: {risk_score})"
    
    # Signal handler for human input
    human_decision: Optional[bool] = None
    
    @workflow.signal
    def approve(self):
        self.human_decision = True
    
    @workflow.signal
    def reject(self):
        self.human_decision = False
# --8<-- [end:lst-6.5]

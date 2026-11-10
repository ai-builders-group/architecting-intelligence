# --8<-- [start:lst-6.18]
APPROVERS = frozenset({"ops-lead", "finance-lead"})  # illustrative ids
@workflow.defn
class HighValueSaga(SagaOrchestrator):
    """Saga with mandatory human approval for compensation."""
    
    human_approved: Optional[bool] = None
    
    async def _compensate(self) -> None:
        """Override compensation to require human approval."""
        completed = self.saga.get_completed_steps()
        
        # Calculate total compensation cost
        total_cost = sum(step.result.get("cost_usd", 0) for step in completed)
        
        if total_cost > 10_000:
            # High-value compensation: request approval
            await workflow.execute_activity(
                "notify_human",
                args=[{
                    "saga_id": self.saga.saga_id,
                    "compensation_cost": total_cost,
                    "steps": [s.name for s in completed]
                }],
                start_to_close_timeout=timedelta(seconds=10)
            )
            
            # Wait for approval signal (timeout after 1 hour)
            try:
                await workflow.wait_condition(
                    lambda: self.human_approved is not None,
                    timeout=timedelta(hours=1)
                )
            except asyncio.TimeoutError:
                await workflow.execute_activity(
                    "escalate_compensation",
                    args=[{
                        "saga_id": self.saga.saga_id,
                        "compensation_cost": total_cost
                    }],
                    start_to_close_timeout=timedelta(seconds=10)
                )
                # Returning here would close the workflow as Completed with the
                # charges outstanding: keep waiting, then compensate
                await workflow.wait_condition(
                    lambda: self.human_approved is not None)
            
            if not self.human_approved:
                logger.critical(
                    "compensation_rejected_by_human",
                    saga_id=self.saga.saga_id
                )
                # Do not compensate. Escalate for manual resolution.
                return
        
        # Proceed with automated compensation
        await super()._compensate()
    
    @workflow.signal
    def approve_compensation(self, operator_id: str):
        """Signal from human operator to approve compensation."""
        # operator_id is caller-supplied: authenticate in your approval service
        if operator_id not in APPROVERS:
            workflow.logger.warning(
                "approval ignored, operator_id not in APPROVERS: %s", operator_id
            )
            return
        self.human_approved = True
    
    @workflow.signal
    def reject_compensation(self, operator_id: str):
        """Signal from human operator to reject compensation."""
        # operator_id is caller-supplied: authenticate in your approval service
        if operator_id not in APPROVERS:
            workflow.logger.warning(
            "rejection ignored, operator_id not in APPROVERS: %s", operator_id)
            return
        self.human_approved = False
# --8<-- [end:lst-6.18]

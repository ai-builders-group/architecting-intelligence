# --8<-- [start:lst-7.22]
class HybridHumanTool:
    """Sync with automatic fallback to async on timeout."""
    
    def __init__(
        self,
        sync_tool: SyncHumanTool,
        async_tool: AsyncHumanTool,
        sync_timeout_seconds: int = 300,
    ):
        self.sync = sync_tool
        self.async_ = async_tool
        self.sync_timeout = sync_timeout_seconds
    
    async def ask_human(
        self, run_id: UUID, prompt: str, proposal: dict, state_snapshot: dict
    ) -> Optional[HumanResponse]:
        """
        Try sync first (if human is active). Fall back to async on timeout.
        """
        # Attempt sync (short timeout)
        response = await self.sync.ask_human(
            run_id, prompt, proposal, state_snapshot
        )
        
        if response is not None:
            return response
        
        # Sync timed out and its ask is withdrawn. Fall back to async for the same
        # proposal.
        metadata = await self.async_.ask_human(
            run_id, prompt, proposal, state_snapshot, ttl_hours=72
        )
        
        # Return None to signal suspension (caller must handle durable workflow)
        return None
# --8<-- [end:lst-7.22]

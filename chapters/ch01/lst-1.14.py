# --8<-- [start:lst-1.14]
# ✅ BOUNDED AGENCY (Safe)
from typing import Any, Callable, Dict
from pydantic import BaseModel

class ToolDefinition(BaseModel):
    """Type-safe tool definition."""
    name: str
    function: Callable
    description: str

ALLOWED_TOOLS: Dict[str, ToolDefinition] = {
    "read_order": ToolDefinition(
        name="read_order",
        function=read_order_from_db,
        description="Read order details (read-only)"
    ),
    "issue_refund": ToolDefinition(
        name="issue_refund",
        function=issue_refund_with_limits,
        description="Issue refund with $500 limit"
    ),
    "send_email": ToolDefinition(
        name="send_email",
        function=send_templated_email,
        description="Send an approved template to the customer of record"
    )
}

class UnauthorizedToolError(Exception):
    """Raised when agent attempts to use non-whitelisted tool."""
    pass

def execute_agent_command(
    tool_name: str, params: Dict[str, Any], ctx: "SessionContext"
) -> Any:
    """
    Execute a whitelisted tool with the given parameters.
    
    Args:
        tool_name: Name of the tool (must be in whitelist)
        params: Tool parameters (never identity: the session supplies it)
        ctx: The authenticated session (tenant and customer)
        
    Returns:
        Tool execution result
        
    Raises:
        UnauthorizedToolError: If tool not in whitelist
    """
    if tool_name not in ALLOWED_TOOLS:
        raise UnauthorizedToolError(
            f"Tool '{tool_name}' not in allowlist. "
            f"Allowed: {list(ALLOWED_TOOLS.keys())}"
        )
    
    tool_def = ALLOWED_TOOLS[tool_name]
    if {"tenant_id", "customer_id"} & params.keys():
        raise UnauthorizedToolError("identity is bound by the session")
    return tool_def.function(
        **params, tenant_id=ctx.tenant_id, customer_id=ctx.customer_id)
# --8<-- [end:lst-1.14]

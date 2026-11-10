# --8<-- [start:lst-1.11]
# src/agents/l3_conditional.py
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from enum import Enum

class GuardrailViolation(BaseModel):
    """A detected policy violation."""
    rule: str = Field(description="The guardrail rule that was violated")
    violated_by: Dict[str, Any] = Field(description="The parameter values that violated the rule")
    severity: Literal["low", "medium", "high"]

class RefundAction(BaseModel):
    """Strictly typed refund action."""
    tool: Literal["issue_refund"]
    order_id: str = Field(pattern=r"^[A-Z0-9]{8,12}$")
    amount: float = Field(gt=0.0, le=10000.0)
    reason: str = Field(min_length=10, max_length=500)

class ActionDecision(BaseModel):
    """The result of the is_safe() check."""
    safe: bool
    action: Optional[RefundAction] = None
    violations: List[GuardrailViolation] = Field(default_factory=list)
    escalation_reason: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)

class Decision(BaseModel):
    """Typed result of a policy evaluation."""
    allow: bool
    reason: str = ""

class Policy:  # stub; the real policy is Chapter 13's
    def evaluate(self, draft: RefundAction) -> Decision: return Decision(allow=True)

policy = Policy()

class L3Agent:
    """
    A conditional agent that acts autonomously within guardrails
    and escalates exceptions to humans.
    """
    
    def __init__(
        self, 
        whitelist: List[str], 
        confidence_threshold: float = 0.85,
        max_auto_refund: float = 500.0
    ):
        self.whitelist = whitelist
        self.confidence_threshold = confidence_threshold
        self.max_auto_refund = max_auto_refund
    
    def process_refund(self, order_id: str, amount: float, reason: str) -> ActionDecision:
        """
        Autonomous refund processing with conditional escalation.
        
        Args:
            order_id: The order to refund
            amount: The refund amount
            reason: The reason for the refund
            
        Returns:
            ActionDecision indicating whether to proceed or escalate
        """
        # Step 1: Generate action plan (with strict typing)
        try:
            action = RefundAction(
                tool="issue_refund",
                order_id=order_id,
                amount=amount,
                reason=reason
            )
        except ValueError as e:
            # Pydantic validation failed - escalate
            return ActionDecision(
                safe=False,
                escalation_reason=f"Invalid action parameters: {str(e)}",
                confidence_score=0.0
            )
        
        # Step 2: Compute confidence
        confidence = self._compute_confidence(action)
        
        # Step 3: Check safety
        return self._is_safe(action, confidence)
    
    def _is_safe(self, action: RefundAction, confidence: float) -> ActionDecision:
        """
        The safety gate: evaluate whether to proceed or escalate.
        
        Args:
            action: The proposed action (already validated by Pydantic)
            confidence: The model's confidence score
            
        Returns:
            ActionDecision with safety determination
        """
        violations: List[GuardrailViolation] = []
        
        # Check 1: Whitelist
        if action.tool not in self.whitelist:
            return ActionDecision(
                safe=False,
                escalation_reason=f"Tool '{action.tool}' not in whitelist",
                confidence_score=confidence
            )
        
        # Check 2: Confidence threshold
        if confidence < self.confidence_threshold:
            return ActionDecision(
                safe=False,
                escalation_reason=(
                    f"Low confidence: {confidence:.2%} < {self.confidence_threshold:.2%}"
                ),
                confidence_score=confidence
            )
        
        # Check 3: Business rule guardrails
        if action.amount > self.max_auto_refund:
            violations.append(GuardrailViolation(
                rule="max_auto_refund",
                violated_by={"amount": action.amount, "limit": self.max_auto_refund},
                severity="high"
            ))
        
        # Check 4: Policy check (the policy itself is Chapter 13's)
        draft = action
        decision = policy.evaluate(draft)
        if not decision.allow:
            violations.append(GuardrailViolation(
                rule="policy",
                violated_by={"reason": decision.reason},
                severity="medium"
            ))
        
        # Decision
        if violations:
            return ActionDecision(
                safe=False,
                violations=violations,
                escalation_reason="Guardrail violations detected",
                confidence_score=confidence
            )
        
        # All checks passed - safe to execute
        return ActionDecision(
            safe=True,
            action=action,
            confidence_score=confidence
        )
    
    def _compute_confidence(self, action: RefundAction) -> float:
        """
        Compute confidence score for the proposed action.
        In production, this would involve:
        - Checking historical refund patterns
        - Validating order status
        - Checking customer history
        
        Returns:
            Confidence score in [0.0, 1.0]
        """
        # Mock confidence calculation
        # In reality, this would call a classifier or rule engine
        return 0.92
# --8<-- [end:lst-1.11]

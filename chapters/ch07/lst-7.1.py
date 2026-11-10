# --8<-- [start:lst-7.1]
# src/agent/escalation.py
from typing import Literal, Optional
from pydantic import BaseModel, Field

class ConfidenceSignals(BaseModel):
    """Measurable proxies for agent confidence."""
    answer_entropy: float = Field(float("nan"), description="Answer entropy")
    tool_ambiguity: bool = Field(default=False, description="Tools differ")
    validation_passed: bool = Field(description="Deterministic schema check result")
    novelty_score: float = Field(ge=0.0, le=1.0, description="OOD embedding distance")

class EscalationPolicy(BaseModel):
    """Thresholds for escalation decision."""
    max_entropy: float = Field(default=2.0)
    max_novelty: float = Field(default=0.8)
    allow_tool_ambiguity: bool = Field(default=False)
    timeout_seconds: int = Field(default=3600, description="Human SLA")

def should_escalate(
    signals: ConfidenceSignals,
    policy: EscalationPolicy = EscalationPolicy()
) -> tuple[bool, str]:
    """Decide whether to run autonomously or call a human."""
    if not signals.validation_passed:
        return True, "Deterministic guardrail violation"

    entropy = signals.answer_entropy
    if entropy is None or not 0 <= entropy <= policy.max_entropy:
        return True, "High model uncertainty (Entropy)"
    
    if signals.tool_ambiguity and not policy.allow_tool_ambiguity:
        return True, "Several tools look equally plausible"
    
    if signals.novelty_score > policy.max_novelty:
        return True, "Out-of-distribution input detected"
        
    return False, "Confidence within bounds"
# --8<-- [end:lst-7.1]

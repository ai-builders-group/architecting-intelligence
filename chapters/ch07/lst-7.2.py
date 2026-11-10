# --8<-- [start:lst-7.2]
# src/agent/admission_control.py
import time
from typing import Optional
from pydantic import BaseModel, Field

class TokenBucket(BaseModel):
    """
    Token Bucket rate limiter for human review queue.
    Prevents queue saturation by rejecting escalations when capacity is exhausted.
    """
    capacity: int = Field(description="Max tokens (concurrent reviews)")
    refill_rate: float = Field(description="Tokens added per second")
    tokens: float = Field(default=0.0, description="Current token count")
    last_refill: float = Field(default_factory=time.time)

    def refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens for an escalation.
        
        Returns:
            True if tokens available (escalation permitted), False otherwise.
        """
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class AdmissionController:
    """
    Governs escalation decisions using Token Bucket rate limiting.
    """
    def __init__(
        self,
        max_concurrent_reviews: int = 5,
        reviews_per_hour: float = 30.0,
    ):
        self.bucket = TokenBucket(
            capacity=max_concurrent_reviews,
            refill_rate=reviews_per_hour / 3600.0,  # Convert to tokens/sec
            tokens=max_concurrent_reviews,
        )

    def should_admit_escalation(self, confidence_score: float, threshold: float) -> bool:
        """
        Decide if escalation should proceed based on confidence and queue capacity.
        
        Args:
            confidence_score: Model confidence (0-1, higher = more confident)
            threshold: Ambiguity threshold (escalate if confidence < threshold)
        
        Returns:
            True if escalation is both necessary (low confidence) and permitted (tokens available).
        """
        if confidence_score >= threshold:
            return False  # High confidence, no escalation needed
        
        # Low confidence detected. Check if human queue has capacity.
        if self.bucket.try_consume(tokens=1):
            return True  # Escalate
        else:
            # Queue saturated. Apply load-shedding policy.
            # Option 1: Auto-deny (fail-safe for high-risk domains)
            # Option 2: Fallback to weaker model (Tier 2 -> Tier 1)
            # Option 3: Queue for delayed review (async path)
            return False  # Reject escalation to prevent queue collapse
# --8<-- [end:lst-7.2]

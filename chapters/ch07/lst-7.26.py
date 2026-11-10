# --8<-- [start:lst-7.26]
# src/agent/supervision/circuit_breaker.py
from collections import deque
from datetime import datetime, UTC, timedelta
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

class Gate(Enum):
    """Routing decision for a proposal."""
    AUTONOMOUS = "autonomous"
    SHED = "shed"
    ESCALATE = "escalate"

class SupervisionMode(BaseModel):
    """Current supervision mode and trigger conditions."""
    mode: Literal["exception_handler", "copilot", "auditor", "load_shedding"]
    reason: str
    triggered_at: datetime

class CircuitBreakerSupervisor:
    """
    Adaptive supervision mode with queue overload protection.
    
    Monitors human rejection rate and queue depth to prevent approval queue collapse.
    Implements deterministic load shedding when λ > μ.
    """
    
    def __init__(
        self,
        window_size: int = 100,  # Track last N decisions
        copilot_threshold: float = 0.15,  # Switch to Co-Pilot if >15% rejections
        auditor_threshold: float = 0.02,  # Switch to Auditor if <2% rejections
        max_queue_depth: int = 50,  # Trigger load shedding if queue exceeds this
        human_service_rate: float = 20.0,  # μ: decisions per hour per human
    ):
        self.window_size = window_size
        self.copilot_threshold = copilot_threshold
        self.auditor_threshold = auditor_threshold
        self.max_queue_depth = max_queue_depth
        self.human_service_rate = human_service_rate
        
        # Sliding window of (timestamp, was_rejected)
        self.rejection_history: deque[tuple[datetime, bool]] = deque(maxlen=window_size)
        
        # Queue depth tracker
        self.current_queue_depth = 0
        
        self.current_mode = SupervisionMode(
            mode="exception_handler",
            reason="Initial state",
            triggered_at=datetime.now(UTC),
        )
    
    def record_human_decision(self, approved: bool):
        """Record a human approval/rejection and update queue depth."""
        self.rejection_history.append((datetime.now(UTC), not approved))
        self.current_queue_depth = max(0, self.current_queue_depth - 1)  # Decrement queue
        self._update_mode()
    
    def record_escalation(self):
        """Record a new escalation (increments queue depth)."""
        self.current_queue_depth += 1
        self._check_queue_overflow()
    
    def _update_mode(self):
        """Adjust supervision mode based on rejection rate."""
        if len(self.rejection_history) < 20:
            # Not enough data to make a decision
            return
        
        rejection_rate = sum(1 for _, rejected in self.rejection_history if rejected) / len(self.rejection_history)
        
        if rejection_rate > self.copilot_threshold:
            # High rejection rate: model is struggling, escalate everything
            if self.current_mode.mode != "copilot":
                self.current_mode = SupervisionMode(
                    mode="copilot",
                    reason=f"Rejection rate {rejection_rate:.1%} exceeds {self.copilot_threshold:.1%}",
                    triggered_at=datetime.now(UTC),
                )
        elif rejection_rate < self.auditor_threshold:
            # Low rejection rate: model is performing well, reduce overhead
            if self.current_mode.mode != "auditor":
                self.current_mode = SupervisionMode(
                    mode="auditor",
                    reason=f"Rejection rate {rejection_rate:.1%} below {self.auditor_threshold:.1%}",
                    triggered_at=datetime.now(UTC),
                )
        else:
            # Normal range: use exception handler
            if self.current_mode.mode != "exception_handler":
                self.current_mode = SupervisionMode(
                    mode="exception_handler",
                    reason=f"Rejection rate {rejection_rate:.1%} in normal range",
                    triggered_at=datetime.now(UTC),
                )
    
    def _check_queue_overflow(self):
        """
        Trigger load shedding if queue depth exceeds capacity.
        
        This implements deterministic admission control: when λ > μ,
        the system auto-rejects/defers new escalations to prevent queue collapse.
        """
        if self.current_queue_depth > self.max_queue_depth:
            if self.current_mode.mode != "load_shedding":
                self.current_mode = SupervisionMode(
                    mode="load_shedding",
                    reason=f"Queue depth {self.current_queue_depth} exceeds capacity {self.max_queue_depth}",
                    triggered_at=datetime.now(UTC),
                )
    
    def should_escalate(self) -> tuple[bool, str]:
        """
        Determine if a new request should escalate based on current mode.
        
        Returns:
            (allow_escalation, reason)
        """
        if self.current_mode.mode == "load_shedding":
            return False, "Load shedding active: queue at capacity"
        
        if self.current_mode.mode == "copilot":
            return True, "Co-pilot mode: all requests escalate"
        
        if self.current_mode.mode == "auditor":
            # In auditor mode, only escalate probabilistically
            import random
            if random.random() < 0.01:  # 1% sample rate
                return True, "Auditor mode: sampled for review"
            return False, "Auditor mode: autonomous execution"
        
        # Default: exception handler mode
        return True, "Exception handler mode: escalate on uncertainty"
    
    def gate(self, flagged: bool) -> Gate:
        """Turn the verifier's vote into a Gate for the router."""
        if not flagged:
            return Gate.AUTONOMOUS
        if self.current_mode.mode == "load_shedding":
            return Gate.SHED
        return Gate.ESCALATE
    
    def get_mode(self) -> Literal["exception_handler", "copilot", "auditor", "load_shedding"]:
        """Get current supervision mode."""
        return self.current_mode.mode
# --8<-- [end:lst-7.26]

# --8<-- [start:lst-4.11]
from enum import Enum

class FallbackStrategy(str, Enum):
    SAFE_AGENT = "safe_agent"  # Route to general-purpose support agent
    HUMAN_QUEUE = "human_queue"  # Immediate escalation
    CACHED_RESPONSE = "cached_response"  # Return pre-written message

class RouterConfig(BaseModel):
    """Configuration for router reliability."""
    timeout_seconds: float = 2.0
    max_retries: int = 2
    fallback_strategy: FallbackStrategy = FallbackStrategy.SAFE_AGENT
    circuit_breaker_threshold: float = 0.10  # Open circuit if error rate > 10%

class CircuitBreakerState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Too many failures, bypass router
    HALF_OPEN = "half_open"  # Testing recovery

class RobustRouter:
    """
    Production router with circuit breaker pattern.
    """
    def __init__(self, config: RouterConfig):
        self.config = config
        self.circuit_state = CircuitBreakerState.CLOSED
        self.error_count = 0
        self.request_count = 0
    
    async def route(self, query: str) -> RoutingDecision:
        """
        Route with automatic fallback on failure.
        """
        self.request_count += 1
        
        # Check circuit breaker
        if self.circuit_state == CircuitBreakerState.OPEN:
            return self._execute_fallback(query, reason="Circuit breaker open")
        
        try:
            # Attempt routing with timeout
            decision = await asyncio.wait_for(
                self._route_internal(query),
                timeout=self.config.timeout_seconds
            )
            
            # Check confidence threshold
            if decision.confidence < 0.85:
                return self._execute_fallback(query, reason="Low confidence")
            
            return decision
            
        except asyncio.TimeoutError:
            self.error_count += 1
            self._check_circuit_breaker()
            return self._execute_fallback(query, reason="Timeout")
        
        except Exception as e:
            self.error_count += 1
            self._check_circuit_breaker()
            return self._execute_fallback(query, reason=f"Exception: {str(e)}")
    
    def _check_circuit_breaker(self):
        """Open circuit if error rate exceeds threshold."""
        if self.request_count < 100:
            return  # Not enough data
        
        error_rate = self.error_count / self.request_count
        
        if error_rate > self.config.circuit_breaker_threshold:
            self.circuit_state = CircuitBreakerState.OPEN
            # Trigger alert to on-call engineer
            alert_oncall(f"Router circuit breaker OPEN. Error rate: {error_rate:.2%}")
    
    def _execute_fallback(self, query: str, reason: str) -> RoutingDecision:
        """
        Execute fallback strategy when routing fails.
        """
        if self.config.fallback_strategy == FallbackStrategy.SAFE_AGENT:
            return RoutingDecision(
                reasoning=f"Router fallback triggered: {reason}. Routing to general support agent.",
                intent=IntentClass.SUPPORT,
                confidence=0.5,  # Mark as low-confidence for monitoring
                entities={}
            )
        elif self.config.fallback_strategy == FallbackStrategy.HUMAN_QUEUE:
            return RoutingDecision(
                reasoning=f"Router failure: {reason}. Escalating to human.",
                intent=IntentClass.SUPPORT,
                confidence=0.0,
                entities={"escalation_reason": reason}
            )
        else:
            raise NotImplementedError(f"Fallback strategy {self.config.fallback_strategy} not implemented")
# --8<-- [end:lst-4.11]

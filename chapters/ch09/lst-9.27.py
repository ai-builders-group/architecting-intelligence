# --8<-- [start:lst-9.27]
# src/ingestion/rate_limiter_failsafe.py
from __future__ import annotations

import time
import asyncio
from typing import Protocol
from redis import Redis
from redis.exceptions import RedisError
from collections import deque
from datetime import datetime, UTC

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass

class LocalTokenBucket:
    """Thread-safe local token bucket for emergency fallback."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # Tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def try_acquire(self) -> bool:
        """Attempt to acquire a token from the local bucket."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

class FailsafeRateLimiter:
    """Distributed rate limiter with fail-closed local fallback."""
    
    def __init__(
        self,
        redis_client: Redis,
        key_prefix: str = "ratelimit",
        max_requests: int = 3000,  # Global limit
        window_seconds: int = 60,
        local_emergency_rate: float = 150.0,  # 5% of global (3000 / 60 * 0.05)
        redis_timeout_threshold: int = 3  # Consecutive failures before fallback
    ):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.max_requests = max_requests
        self.window = window_seconds
        
        # Local bucket: 5% of capacity per node (divide by node count)
        self.local_bucket = LocalTokenBucket(
            capacity=int(max_requests * 0.05),
            refill_rate=local_emergency_rate / 60.0  # Convert to per-second
        )
        
        # Circuit breaker state
        self.redis_failures = 0
        self.redis_timeout_threshold = redis_timeout_threshold
        self.fallback_mode = False
        self.last_fallback_log = 0
    
    async def acquire(self, resource: str, timeout_sec: float = 5.0) -> None:
        """
        Acquire a token with fail-closed fallback.
        
        Raises RateLimitExceeded if:
        - Distributed limit exceeded (Redis healthy)
        - Local emergency reserve exhausted (Redis down)
        - Timeout expired
        """
        deadline = time.time() + timeout_sec
        
        while time.time() < deadline:
            # Try distributed rate limiter first
            if not self.fallback_mode:
                try:
                    success = await self._try_distributed_acquire(resource)
                    if success:
                        # Reset failure counter on success
                        self.redis_failures = 0
                        return
                    else:
                        # Rate limit exceeded (legitimate)
                        await asyncio.sleep(0.1)
                        continue
                
                except RedisError as e:
                    self.redis_failures += 1
                    
                    if self.redis_failures >= self.redis_timeout_threshold:
                        self.fallback_mode = True
                        self._log_fallback_activation()
                    
                    # Continue to fallback logic
            
            # Fallback: Local emergency reserve
            if await self.local_bucket.try_acquire():
                self._log_fallback_usage()
                return
            else:
                # Local reserve exhausted - fail closed
                await asyncio.sleep(0.1)
                continue
        
        raise RateLimitExceeded(
            f"Could not acquire token for {resource} within {timeout_sec}s"
        )
    
    async def _try_distributed_acquire(self, resource: str) -> bool:
        """Attempt to acquire from distributed rate limiter."""
        key = f"{self.key_prefix}:{resource}"
        
        lua_script = """
        local key = KEYS[1]
        local max_requests = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local data = redis.call('HMGET', key, 'count', 'window_start')
        local count = tonumber(data[1]) or 0
        local window_start = tonumber(data[2]) or now
        
        if now - window_start >= window then
            count = 0
            window_start = now
        end
        
        if count < max_requests then
            redis.call('HMSET', key, 'count', count + 1, 'window_start', window_start)
            redis.call('EXPIRE', key, window * 2)
            return 1
        else
            return 0
        end
        """
        
        result = self.redis.eval(
            lua_script,
            1,
            key,
            self.max_requests,
            self.window,
            int(time.time())
        )
        
        return result == 1
    
    def _log_fallback_activation(self) -> None:
        """Log fallback mode activation (rate-limited)."""
        now = time.time()
        if now - self.last_fallback_log > 60:  # Log once per minute
            print(f"[CRITICAL] Rate limiter entered fallback mode (Redis unreachable). "
                  f"Throughput degraded to {self.local_bucket.refill_rate * 60:.0f} RPM.")
            self.last_fallback_log = now
    
    def _log_fallback_usage(self) -> None:
        """Log fallback usage (debug only)."""
        # In production, emit metric instead of logging
        pass
# --8<-- [end:lst-9.27]

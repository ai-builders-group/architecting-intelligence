# --8<-- [start:lst-9.37]
# src/08_retrieval_topology/redis_lock.py
import asyncio
from datetime import timedelta
import redis.asyncio as aioredis
import uuid

class RedisDistributedLock:
    """
    Distributed lock using Redis SET with NX (Not eXists) and EX (Expiry).
    
    Based on the Redlock algorithm (simplified single-instance version).
    """
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    
    async def acquire(self, key: str, token: str, ttl: timedelta) -> bool:
        """
        Try to acquire the lock.
        
        Uses SET NX EX to atomically:
        1. Check if key exists
        2. Set key with expiration if not exists
        """

        # SET key value NX EX seconds
        acquired = await self.redis.set(
            name=f"lock:{key}",
            value=token,
            nx=True,  # Only set if key does not exist
            ex=int(ttl.total_seconds()),
        )
        
        return acquired is not None
    
    async def release(self, key: str, token: str) -> None:
        """
        Release the lock (only if we hold it).
        
        Uses Lua script for atomicity: Check token + Delete.
        """

        # Lua script to ensure we only delete if token matches
        script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """
        
        await self.redis.eval(script, 1, f"lock:{key}", token)
    
    async def extend(self, key: str, token: str, ttl: timedelta) -> bool:
        """Extend the lock expiration (for long prefill operations)."""

        # EXPIRE only if token matches
        script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("EXPIRE", KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        
        result = await self.redis.eval(
            script, 1, f"lock:{key}", token, int(ttl.total_seconds())
        )
        
        return result == 1
# --8<-- [end:lst-9.37]

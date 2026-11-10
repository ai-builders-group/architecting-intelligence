# --8<-- [start:lst-9.34]
# path/to/topic_detection.py
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import AsyncOpenAI, OpenAI
import numpy as np
from datetime import datetime, UTC, timedelta

client = OpenAI()

class TopicShiftConfig(BaseModel):
    """
    Configuration for topic shift detection with rate limiting.
    """
    # Calibrated threshold from eval set
    shift_threshold: float = Field(
        default=0.42,
        description="Calibrated threshold from precision-recall analysis"
    )
    min_history_size: int = 5
    
    # Rate limiting to prevent summarization storms
    cooldown_seconds: int = Field(
        default=300,
        description="Minimum time between summarizations (5 minutes)"
    )
    max_summaries_per_hour: int = Field(
        default=6,
        description="Maximum topic flushes per session per hour"
    )
    
    # Metrics from calibration
    precision_at_threshold: float = Field(default=0.85)
    recall_at_threshold: float = Field(default=0.78)

class SummarizationRateLimiter:
    """
    Token bucket rate limiter for topic flush operations.
    Prevents oscillating topics from triggering cost explosion.
    """
    def __init__(self, config: TopicShiftConfig):
        self.config = config
        self.last_summary_time: dict[str, datetime] = {}
        self.summary_count: dict[str, list[datetime]] = {}
    
    def can_summarize(self, session_id: str) -> tuple[bool, str]:
        """
        Check if summarization is allowed for this session.
        
        Returns:
            (allowed: bool, reason: str)
        """
        now = datetime.now(UTC)
        
        # Check cooldown
        last_time = self.last_summary_time.get(session_id)
        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return False, (
                    f"Cooldown active "
                    f"({int(self.config.cooldown_seconds - elapsed)}s remaining)"
                )
        
        # Check hourly quota
        recent_summaries = self.summary_count.get(session_id, [])
        hour_ago = now - timedelta(hours=1)
        recent_summaries = [t for t in recent_summaries if t > hour_ago]
        
        if len(recent_summaries) >= self.config.max_summaries_per_hour:
            return False, (
                f"Hourly quota exhausted "
                f"({len(recent_summaries)}/{self.config.max_summaries_per_hour})"
            )
        
        return True, "allowed"
    
    def record_summary(self, session_id: str):
        """Record that a summary was generated."""
        now = datetime.now(UTC)
        self.last_summary_time[session_id] = now
        
        if session_id not in self.summary_count:
            self.summary_count[session_id] = []
        self.summary_count[session_id].append(now)

class TopicShiftDetector:
    """
    Detect semantic topic changes using cached embeddings.
    Triggers rate-limited asynchronous summarization.
    """
    
    def __init__(
        self, 
        config: TopicShiftConfig, 
        background_queue,
        message_store
    ):
        self.client = AsyncOpenAI()
        self.config = config
        self.background_queue = background_queue
        self.message_store = message_store
        self.rate_limiter = SummarizationRateLimiter(config)
    
    async def detect_and_summarize_async(
        self, 
        current_query: str,
        tenant_id: str,
        session_id: str
    ) -> tuple[bool, Optional[str]]:
        """
        Detect topic shift using CACHED embeddings and trigger ASYNC summarization.
        
        Returns:
            (shift_detected: bool, rejection_reason: Optional[str])
        """
        # Fetch recent history with CACHED embeddings (zero API calls)
        recent_history = await self.message_store.get_history(
            session_id=session_id,
            limit=self.config.min_history_size + 1
        )
        
        if len(recent_history) < self.config.min_history_size:
            return False, None
        
        # Embed current query (only network call)
        response = await self.client.embeddings.create(
            model="text-embedding-3-large",
            input=current_query
        )
        current_emb = np.array(response.data[0].embedding)
        
        # Extract CACHED embeddings from history (zero API calls)
        history_embeddings = np.array([
            msg.embedding for msg in recent_history[-5:]
            if msg.embedding
        ])
        
        if len(history_embeddings) == 0:
            return False, None
        
        # Vectorized similarity computation
        current_norm = current_emb / (np.linalg.norm(current_emb) + 1e-9)
        history_norms = history_embeddings / (np.linalg.norm(history_embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = history_norms @ current_norm
        
        avg_history_sim = float(np.mean(similarities))
        
        # Topic shift score
        previous_query_sim = similarities[-1] if len(similarities) > 0 else 1.0
        shift_score = previous_query_sim / (avg_history_sim + 1e-6)
        
        if shift_score < self.config.shift_threshold:
            # Topic shift detected - check rate limit
            allowed, reason = self.rate_limiter.can_summarize(session_id)
            
            if not allowed:
                # Rate limit hit - log but don't summarize
                return True, f"Topic shift detected but {reason}"
            
            # ASYNC: Queue background summarization job
            await self.background_queue.enqueue({
                "job_type": "summarize_history",
                "tenant_id": tenant_id, "session_id": session_id,
                "history_ids": [msg.id for msg in recent_history],
                "priority": "low"
            })
            
            self.rate_limiter.record_summary(session_id)
            
            return True, None
        
        return False, None
# --8<-- [end:lst-9.34]

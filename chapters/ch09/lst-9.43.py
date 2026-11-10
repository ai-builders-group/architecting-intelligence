# --8<-- [start:lst-9.43]
# chapter_07/hysteresis_lru_vectorized.py
from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime, UTC
from openai import OpenAI
import numpy as np

class Message(BaseModel):
    """Immutable conversation message with metadata."""
    id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    token_count: int
    embedding: List[float] | None = None
    is_authoritative: bool = False  # True for tool outputs, human edits
    is_pinned: bool = False  # Immunity from eviction
    pin_turns_remaining: int = 0  # Countdown for temporary pins
    
class EvictionConfig(BaseModel):
    """Hyperparameters for the retention scoring function."""
    similarity_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    authority_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    
    # Watermark thresholds
    high_watermark: float = Field(default=0.8, ge=0.5, le=0.95)
    low_watermark: float = Field(default=0.5, ge=0.3, le=0.7)
    
    # Pinning policy (anti-thrashing)
    pin_retrieved_envelopes: bool = Field(default=True)
    pin_duration_turns: int = Field(default=3, description="Turns to keep retrieved context pinned")
    
    def validate_weights(self) -> None:
        total = self.similarity_weight + self.recency_weight + self.authority_weight
        if abs(total - 1.0) >= 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        if self.high_watermark <= self.low_watermark:
            raise ValueError("High watermark must exceed low")

class MemoryPressureState(BaseModel):
    """Tracks current memory utilization and pressure state."""
    current_tokens: int
    max_tokens: int
    utilization: float = Field(ge=0.0, le=1.0)
    is_high_pressure: bool
    needs_eviction: bool
    
    @classmethod
    def from_history(cls, history: List[Message], max_tokens: int, config: EvictionConfig):
        current = sum(m.token_count for m in history)
        util = current / max_tokens
        return cls(
            current_tokens=current,
            max_tokens=max_tokens,
            utilization=util,
            is_high_pressure=util >= config.high_watermark,
            needs_eviction=util >= config.high_watermark
        )

class HysteresisSemanticLRU:
    """
    Semantic LRU with watermark-based triggering and vectorized scoring.
    
    Eviction runs ONLY when memory pressure exceeds the high watermark.
    Scoring uses NumPy vectorization to prevent the Python loop bottleneck.
    """
    
    def __init__(self, config: EvictionConfig, max_tokens: int = 6000):
        self.config = config
        self.max_tokens = max_tokens
        self.client = OpenAI()
        config.validate_weights()
        
        # Track when we last evicted (for debugging)
        self.last_eviction_time: datetime | None = None
        self.eviction_count: int = 0
    
    def check_pressure(self, history: List[Message]) -> MemoryPressureState:
        """Determine if eviction is necessary based on watermarks."""
        return MemoryPressureState.from_history(history, self.max_tokens, self.config)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI's API."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",  # 1536 dims, cheap
            input=text
        )
        return response.data[0].embedding
    
    def evict(
        self, 
        history: List[Message], 
        query: str
    ) -> tuple[List[Message], List[Message], MemoryPressureState]:
        """
        Evict low-value messages to fit within token budget.
        
        Uses watermark-based triggering and vectorized scoring.
        Respects pinned messages to prevent retrieval-triggered thrashing.
        
        Args:
            history: Full conversation history
            query: Current user query
        
        Returns:
            (retained_messages, evicted_messages, pressure_state)
        """
        # Decrement pin counters
        self.decrement_pin_counters(history)
        
        # Check pressure first (cheap operation)
        pressure = self.check_pressure(history)
        
        if not pressure.needs_eviction:
            # Below high watermark - skip eviction entirely
            return history, [], pressure
        
        # High pressure detected - perform eviction
        if not history:
            return [], [], pressure
        
        # Separate pinned and unpinned messages
        pinned_messages, unpinned_messages = self._split_pinned(history)
        
        if not unpinned_messages:
            # All messages are pinned - cannot evict
            # This is the "thrashing" state - return as-is
            return history, [], pressure
        
        # Embed the query (only when eviction is necessary)
        query_embedding = np.array(self.embed_text(query))
        
        # VECTORIZED SCORING: Score all unpinned messages in ONE operation
        scores = self.compute_retention_scores_vectorized(
            unpinned_messages, 
            query_embedding, 
            len(history)
        )
        
        # Pair messages with scores
        scored = list(zip(unpinned_messages, scores))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy packing to low watermark target
        target_tokens = int(self.max_tokens * self.config.low_watermark)
        
        retained = list(pinned_messages)  # Always keep pinned messages
        evicted = []
        
        current_tokens = sum(m.token_count for m in pinned_messages)
        
        for msg, score in scored:
            if current_tokens + msg.token_count <= target_tokens:
                retained.append(msg)
                current_tokens += msg.token_count
            else:
                evicted.append(msg)
        
        # Re-sort retained by timestamp to preserve chronological order
        retained.sort(key=lambda m: m.timestamp)
        
        # Update tracking
        self.last_eviction_time = datetime.now(UTC)
        self.eviction_count += 1
        
        # Recalculate pressure after eviction
        final_pressure = MemoryPressureState.from_history(retained, self.max_tokens, self.config)
        
        return retained, evicted, final_pressure
    
    def compute_retention_scores_vectorized(
        self, 
        messages: List[Message], 
        query_embedding: np.ndarray,
        current_turn: int
    ) -> np.ndarray:
        """
        Calculate retention scores for ALL messages in a single vectorized operation.
        
        This replaces the O(N) Python loop with O(1) matrix multiplication.
        
        Args:
            messages: List of messages to score
            query_embedding: Query vector (d,)
            current_turn: Current conversation turn number
        
        Returns:
            scores: Array of retention scores (N,)
        """
        if not messages:
            return np.array([])
        
        # Extract embeddings (embed any missing ones)
        embeddings_list = []
        for msg in messages:
            if msg.embedding is None:
                msg.embedding = self.embed_text(msg.content)
            embeddings_list.append(msg.embedding)
        
        # Stack into matrix: (N, d)
        message_embeddings = np.array(embeddings_list)
        
        # 1. SIMILARITY COMPONENT (Vectorized cosine similarity)
        # Normalize query: (d,)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        
        # Normalize messages: (N, d)
        message_norms = np.linalg.norm(message_embeddings, axis=1, keepdims=True)
        messages_normalized = message_embeddings / (message_norms + 1e-9)
        
        # Matrix-vector multiplication: (N, d) @ (d,) -> (N,)
        similarities = messages_normalized @ query_norm
        
        # 2. RECENCY COMPONENT (Vectorized exponential decay)
        current_time = datetime.now(UTC).timestamp()
        message_times = np.array([m.timestamp.timestamp() for m in messages])
        time_deltas = (current_time - message_times) / 60  # Minutes ago
        recencies = np.exp(-self.config.decay_rate * time_deltas)
        
        # 3. AUTHORITY COMPONENT (Vectorized binary mask)
        authorities = np.array([1.0 if m.is_authoritative else 0.0 for m in messages])
        
        # 4. WEIGHTED COMBINATION (Single vectorized operation)
        scores = (
            self.config.similarity_weight * similarities +
            self.config.recency_weight * recencies +
            self.config.authority_weight * authorities
        )
        
        return scores
    
    def decrement_pin_counters(self, history: List[Message]) -> None:
        """
        Decrement pin counters for all messages.
        Unpin messages when counter reaches 0.
        """
        for msg in history:
            if msg.is_pinned and msg.pin_turns_remaining > 0:
                msg.pin_turns_remaining -= 1
                if msg.pin_turns_remaining == 0:
                    msg.is_pinned = False
    
    
    def _split_pinned(self, history: List[Message]) -> tuple[List[Message], List[Message]]:
        """Pinned and system messages are kept; the rest can be evicted."""
        pinned_messages = [
            m for m in history if m.is_pinned or m.role == "system"]
        unpinned_messages = [m for m in history if not m.is_pinned and m.role != "system"]
        return pinned_messages, unpinned_messages
# --8<-- [end:lst-9.43]

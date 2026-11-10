# --8<-- [start:lst-9.32]
# path/to/semantic_lru.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from openai import AsyncOpenAI, OpenAI
import numpy as np
from collections import defaultdict

client = OpenAI()

class Message(BaseModel):
    """Minimal message shape used by the pruner (full model in lst-9.34)."""
    content: str
    embedding: List[float] = Field(default_factory=list)
    embedding_model: str = ""
    embedding_dim: int = 0
    tenant_id: str = ""
    user_id: str = ""
    thread_id: Optional[str] = None
    created_at: datetime

class ConversationThread(BaseModel):
    """A semantically coherent group of messages."""
    thread_id: str
    messages: List[Message]
    avg_relevance_score: float = 0.0
    token_count: int = 0

class SemanticLRU:
    """
    Evict least semantically relevant threads, not individual messages.
    Preserves referential integrity by keeping threads intact.
    Uses CACHED embeddings and vectorized operations.
    """
    
    def __init__(self, threshold: float = 0.3, min_thread_size: int = 2):
        self.client = AsyncOpenAI()
        self.threshold = threshold
        self.min_thread_size = min_thread_size
    
    def _vectorized_similarity(
        self, 
        query_embedding: List[float], 
        history_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity for all history messages in ONE operation.
        Prevents O(N) Python loop allocations.
        
        Args:
            query_embedding: Single query vector (d,)
            history_embeddings: Matrix of history vectors (N, d)
        
        Returns:
            Similarity scores (N,)
        """
        query_arr = np.array(query_embedding)
        
        # Normalize query
        query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-9)
        
        # Normalize history (per row)
        history_norms = np.linalg.norm(history_embeddings, axis=1, keepdims=True)
        history_normalized = history_embeddings / (history_norms + 1e-9)
        
        # Single matrix-vector multiplication: (N, d) @ (d,) -> (N,)
        similarities = history_normalized @ query_norm
        
        return similarities
    
    def _group_into_threads(self, history: List[Message]) -> List[ConversationThread]:
        """
        Group messages into threads using LLM-tagged thread_id.
        Fallback to temporal proximity for untagged messages.
        
        A thread is a sequence of messages where:
        1. Messages share the same thread_id (primary), OR
        2. Messages are temporally adjacent within 5-minute windows (fallback)
        """
        if not history:
            return []
        
        # Primary: Group by thread_id
        threads_by_id = defaultdict(list)
        untagged = []
        
        for msg in history:
            if msg.thread_id:
                threads_by_id[msg.thread_id].append(msg)
            else:
                untagged.append(msg)
        
        threads = []
        
        # Process tagged threads
        for thread_id, messages in threads_by_id.items():
            if len(messages) >= self.min_thread_size:
                threads.append(self._create_thread(thread_id, messages))
        
        # Fallback: Temporal grouping for untagged messages
        if untagged:
            current_thread = []
            for i, msg in enumerate(untagged):
                if not current_thread:
                    current_thread.append(msg)
                    continue
                
                # Check temporal proximity (within 5 minutes)
                time_gap = (msg.created_at - current_thread[-1].created_at).total_seconds()
                
                if time_gap < 300:  # 5 minutes
                    current_thread.append(msg)
                else:
                    # Start new thread
                    if len(current_thread) >= self.min_thread_size:
                        thread_id = f"temp_{current_thread[0].created_at.timestamp()}"
                        threads.append(self._create_thread(thread_id, current_thread))
                    current_thread = [msg]
            
            # Add final thread
            if len(current_thread) >= self.min_thread_size:
                thread_id = f"temp_{current_thread[0].created_at.timestamp()}"
                threads.append(self._create_thread(thread_id, current_thread))
        
        return threads
    
    def _create_thread(self, thread_id: str, messages: List[Message]) -> ConversationThread:
        """Create a thread object from messages."""
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        
        token_count = sum(len(enc.encode(m.content)) for m in messages)
        
        return ConversationThread(
            thread_id=thread_id,
            messages=messages,
            token_count=token_count
        )
    
    def _newest_within(self, history: List[Message], max_tokens: int) -> List[Message]:
        """Keep the newest messages whose summed token count fits max_tokens."""
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        kept, used = [], 0
        for m in sorted(history, key=lambda m: m.created_at, reverse=True):
            n = len(enc.encode(m.content))
            if used + n > max_tokens:
                break
            kept.append(m)
            used += n
        return sorted(kept, key=lambda m: m.created_at)
    
    async def prune(
        self, 
        tenant_id: str,
        user_id: str,
        history: List[Message], 
        current_query: str,
        max_tokens: int
    ) -> List[Message]:
        """
        Evict threads below semantic relevance threshold.
        Preserves referential integrity by keeping threads intact.
        Uses CACHED embeddings and vectorized similarity.
        
        Args:
            tenant_id, user_id: The principal that owns the history; a history
                that mixes principals raises ValueError.
            history: Conversation history with pre-computed embeddings.
            current_query: The user's current message.
            max_tokens: Maximum token budget for history.
        
        Returns:
            Pruned history preserving semantically relevant threads.
        """
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        
        if any(m.tenant_id != tenant_id or m.user_id != user_id
               for m in history):
            raise ValueError("history mixes principals")
        if not any(m.embedding for m in history):
            return self._newest_within(history, max_tokens)
        # One model and one dimension, or the cosine scores mean nothing
        (model,) = {m.embedding_model for m in history if m.embedding}
        (dim,) = {m.embedding_dim for m in history if m.embedding}
        # Embed current query ONCE (only network call)
        response = await self.client.embeddings.create(
            model=model, input=current_query)
        query_embedding = response.data[0].embedding
        if len(query_embedding) != dim:
            raise ValueError("query and history dimensions differ")
        
        # Group messages into threads (uses LLM-tagged thread_id)
        threads = self._group_into_threads(history)
        
        # Score each thread using VECTORIZED operations
        for thread in threads:
            # Extract CACHED embeddings (no API calls)
            history_embeddings = np.array(
                [msg.embedding for msg in thread.messages if msg.embedding]
            )
            
            if len(history_embeddings) == 0:
                thread.avg_relevance_score = 0.0
                continue
            
            # Compute all similarities in ONE operation
            similarities = self._vectorized_similarity(query_embedding, history_embeddings)
            
            thread.avg_relevance_score = float(np.mean(similarities))
        
        # Sort threads by relevance (descending)
        sorted_threads = sorted(threads, key=lambda t: -t.avg_relevance_score)
        
        # Greedy pack: keep highest relevance threads within budget
        pruned_messages = []
        token_count = 0
        
        for thread in sorted_threads:
            # Only keep if above threshold AND within budget
            if thread.avg_relevance_score >= self.threshold and (
                token_count + thread.token_count) <= max_tokens:
                pruned_messages.extend(thread.messages)
                token_count += thread.token_count
        
        # Restore chronological order for coherent conversation flow
        pruned_messages.sort(key=lambda m: m.created_at)
        
        return pruned_messages
# --8<-- [end:lst-9.32]

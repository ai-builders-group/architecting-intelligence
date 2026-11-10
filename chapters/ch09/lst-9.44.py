# --8<-- [start:lst-9.44]
# chapter_07/virtual_memory_pinned.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta, UTC
import chromadb
from chromadb.config import Settings

class RetrievalWindow(BaseModel):
    """Configuration for temporal envelope retrieval."""
    before_messages: int = Field(
        default=2, 
        description="Messages to include before the match"
    )
    after_messages: int = Field(
        default=2, 
        description="Messages to include after the match"
    )
    time_window_seconds: int = Field(
        default=300,
        description="Maximum time gap for envelope inclusion (5 minutes)"
    )
    pin_duration_turns: int = Field(
        default=3,
        description="Turns to keep retrieved envelopes pinned (anti-thrashing)"
    )

class ArchivedMessage(BaseModel):
    """Message stored in long-term memory with full provenance."""
    message: Message
    conversation_id: str
    turn_number: int
    topic_segment_id: str | None = None

class TemporalEnvelope(BaseModel):
    """A retrieved message with surrounding context and pinning metadata."""
    anchor_message: Message
    before_context: List[Message] = Field(default_factory=list)
    after_context: List[Message] = Field(default_factory=list)
    total_tokens: int
    confidence_score: float
    
    # Pinning metadata (anti-thrashing)
    is_pinned: bool = Field(default=True, description="Immunity from eviction")
    pin_turns_remaining: int = Field(default=3, description="Countdown timer")
    
class VirtualMemoryPinned:
    """
    Virtual memory system with temporal envelope retrieval and pinning.
    
    Prevents orphaned message retrieval by automatically fetching
    surrounding messages. Prevents retrieval-triggered thrashing by
    pinning envelopes for K turns.
    """
    
    def __init__(
        self, 
        collection_name: str = "conversation_archive",
        window_config: RetrievalWindow = RetrievalWindow()
    ):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.window_config = window_config
    
    def page_out(
        self, 
        messages: List[Message],
        tenant_id: str,
        conversation_id: str,
        segment_id: str | None = None
    ) -> None:
        """
        Archive messages to long-term storage.
        
        Stores turn_number and temporal metadata for envelope retrieval.
        """
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for i, msg in enumerate(messages):
            if msg.embedding is None:
                continue
            
            documents.append(msg.content)
            embeddings.append(msg.embedding)
            metadatas.append({
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "role": msg.role,
                "timestamp": msg.timestamp.isoformat(),
                "turn_number": i,  # Critical for envelope retrieval
                "is_authoritative": msg.is_authoritative,
                "segment_id": segment_id or "unknown"
            })
            ids.append(f"{conversation_id}_{msg.id}")
        
        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
    
    def _fetch_temporal_envelope(
        self,
        tenant_id: str,
        conversation_id: str,
        anchor_turn: int,
        anchor_timestamp: datetime
    ) -> tuple[List[Message], List[Message]]:
        """
        Fetch messages surrounding the anchor message.
        
        Returns:
            (before_messages, after_messages)
        """
        # Query for messages within temporal window
        time_min = anchor_timestamp - timedelta(seconds=self.window_config.time_window_seconds)
        time_max = anchor_timestamp + timedelta(seconds=self.window_config.time_window_seconds)
        
        results = self.collection.get(
            where={"$and": [{"tenant_id": tenant_id},
                            {"conversation_id": conversation_id}]},
            include=["metadatas", "documents"]
        )
        
        if not results["documents"]:
            return [], []
        
        # Reconstruct messages and sort by turn number
        messages = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            if not (time_min <= datetime.fromisoformat(
                meta["timestamp"]) <= time_max):
                continue
            messages.append((
                meta["turn_number"],
                Message(
                    id=results["ids"][i],
                    role=meta["role"],
                    content=doc,
                    timestamp=datetime.fromisoformat(meta["timestamp"]),
                    token_count=len(doc.split()),
                    is_authoritative=meta["is_authoritative"]
                )
            ))
        
        messages.sort(key=lambda x: x[0])  # Sort by turn number
        
        # Split into before/after based on anchor
        before = [msg for turn, msg in messages if turn < anchor_turn][
            -self.window_config.before_messages:]
        after = [msg for turn, msg in messages if turn > anchor_turn][
            :self.window_config.after_messages]
        
        return before, after
    
    def page_in(
        self, 
        query: str,
        tenant_id: str,
        conversation_id: str,
        top_k: int = 3
    ) -> List[TemporalEnvelope]:
        """
        Retrieve archived messages with temporal envelopes and pinning.
        
        For each top-k match, fetches surrounding messages to preserve
        causal context and PINS the entire envelope for K turns to prevent
        retrieval-triggered thrashing.
        
        Args:
            query: Current user query
            tenant_id: Tenant that owns the conversation
            conversation_id: Conversation to search within
            top_k: Number of anchor messages to retrieve
        
        Returns:
            List of envelopes with full context windows and pinning metadata
        """
        # Query the archive for anchor messages
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"$and": [{"tenant_id": tenant_id},
                            {"conversation_id": conversation_id}]}
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []
        
        # Build temporal envelopes
        envelopes = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            
            # Reconstruct anchor message
            anchor_msg = Message(
                id=results["ids"][0][i],
                role=meta["role"],
                content=doc,
                timestamp=datetime.fromisoformat(meta["timestamp"]),
                token_count=len(doc.split()),
                is_authoritative=meta["is_authoritative"],
                is_pinned=True,  # Pin immediately
                pin_turns_remaining=self.window_config.pin_duration_turns
            )
            
            # Fetch surrounding context
            before, after = self._fetch_temporal_envelope(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                anchor_turn=meta["turn_number"],
                anchor_timestamp=anchor_msg.timestamp
            )
            
            # Pin all envelope messages
            for msg in before + [anchor_msg] + after:
                msg.is_pinned = True
                msg.pin_turns_remaining = self.window_config.pin_duration_turns
            
            # Calculate total tokens
            total_tokens = (
                anchor_msg.token_count +
                sum(m.token_count for m in before) +
                sum(m.token_count for m in after)
            )
            
            envelopes.append(TemporalEnvelope(
                anchor_message=anchor_msg,
                before_context=before,
                after_context=after,
                total_tokens=total_tokens,
                confidence_score=results["distances"][0][i] if results["distances"] else 0.0,
                is_pinned=True,
                pin_turns_remaining=self.window_config.pin_duration_turns
            ))
        
        return envelopes

def assemble_retrieved_context(envelopes: List[TemporalEnvelope]) -> str:
    """
    Assemble retrieved envelopes into injection-ready context.
    
    Preserves chronological order and marks boundaries.
    All messages in envelopes are PINNED to prevent immediate re-eviction.
    """
    context_parts = []
    
    for env in envelopes:
        # Mark the envelope boundary
        context_parts.append("[ARCHIVED CONTEXT - BEGIN]")
        
        # Include before context
        for msg in env.before_context:
            context_parts.append(f"{msg.role.upper()}: {msg.content}")
        
        # The anchor message (highest relevance)
        context_parts.append(f"**{env.anchor_message.role.upper()}**: {env.anchor_message.content}")
        
        # Include after context
        for msg in env.after_context:
            context_parts.append(f"{msg.role.upper()}: {msg.content}")
        
        context_parts.append(f"[ARCHIVED CONTEXT - END] (Pinned for {env.pin_turns_remaining} turns)")
    
    return "\n".join(context_parts)
# --8<-- [end:lst-9.44]

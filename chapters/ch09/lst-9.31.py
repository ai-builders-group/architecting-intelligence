# --8<-- [start:lst-9.31]
# path/to/message_store.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, UTC
from openai import AsyncOpenAI, OpenAI
import numpy as np

client = OpenAI()

class Message(BaseModel):
    """A conversation message with cached embedding."""
    role: str  # "user" | "assistant"
    content: str
    embedding: List[float] = Field(default_factory=list)
    embedding_model: str = Field(default="text-embedding-3-large")
    model_version: str = Field(
        default="2024-01",
        description="Embedding model version for migration tracking"
    )
    embedding_dim: int = 0  # len(embedding), stored with every row
    tenant_id: str = ""
    user_id: str = ""
    session_id: str = ""
    relevance_score: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    # Thread tracking for referential integrity
    thread_id: Optional[str] = Field(
        default=None,
        description="LLM-tagged conversation thread for preservation"
    )
    references_message_ids: List[str] = Field(
        default_factory=list,
        description="IDs of messages this message references"
    )
    is_referenced_by: List[str] = Field(
        default_factory=list,
        description="IDs of messages that reference this one"
    )

class MessageStore:
    """
    Persistent storage for messages with write-time embedding.
    Embeddings are generated ONCE at creation and cached.
    """
    def __init__(self, db_connection):
        self.db = db_connection
        self.client = AsyncOpenAI()
    
    async def store_message(self, message: Message) -> Message:
        """
        Store message with embedding generated at write time.
        This is called during the agent's response cycle, NOT during retrieval.
        """
        # Generate embedding if not already present
        if not message.embedding:
            response = await self.client.embeddings.create(
                model=message.embedding_model,
                input=message.content
            )
            message.embedding = response.data[0].embedding
        
        message.embedding_dim = len(message.embedding)
        # Persist to database with embedding
        await self.db.insert("messages", {
            "role": message.role,
            "content": message.content,
            "tenant_id": message.tenant_id,
            "user_id": message.user_id,
            "session_id": message.session_id,
            "embedding": message.embedding,
            "embedding_model": message.embedding_model,
            "embedding_dim": message.embedding_dim,
            "model_version": message.model_version,
            "created_at": message.created_at,
            "thread_id": message.thread_id,
            "references_message_ids": message.references_message_ids,
            "is_referenced_by": message.is_referenced_by
        })
        
        return message
    
    async def get_history(
        self, tenant_id: str, user_id: str,
        session_id: str, limit: int = 100,
    ) -> List[Message]:
        """
        Retrieve history with pre-computed embeddings.
        NO NETWORK CALLS during this operation.
        """
        rows = await self.db.query(
            "SELECT * FROM messages WHERE tenant_id = $1"
            " AND user_id = $2 AND session_id = $3"
            " ORDER BY created_at DESC LIMIT $4",
            tenant_id, user_id, session_id, limit
        )
        
        return [Message(**row) for row in rows]# --8<-- [end:lst-9.31]

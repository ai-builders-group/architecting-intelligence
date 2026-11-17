# --8<-- [start:lst-9.8]
# path/to/contextual_prepend.py
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI

client = OpenAI()

class ContextualChunk(BaseModel):
    """Chunk embedding that preserves global document context via prepending."""
    chunk_id: str
    text: str  # Original text (WITHOUT summary)
    embedding: List[float]  # Contextualized embedding (WITH summary)
    token_span: tuple[int, int]
    summary_used: str = Field(description="Document summary for provenance")

class ContextualPrependSplitter:
    """Managed API-compatible alternative to Late Chunking.
    
    Achieves context preservation by prepending document summaries
    to chunks during embedding, without requiring hidden state access.
    """
    
    def __init__(
        self, 
        chunk_size: int = 512,
        summary_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-large"
    ):
        self.chunk_size = chunk_size
        self.summary_model = summary_model
        self.embedding_model = embedding_model
    
    def process_document(self, text: str) -> List[ContextualChunk]:
        """Embed document with contextual prepending."""
        
        # Step 1: Generate document-level summary (ONCE)
        summary = self._generate_summary(text)
        
        # Step 2: Chunk the text (using deterministic splitter)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,  # Rough token-to-char
            chunk_overlap=100
        )
        raw_chunks = splitter.split_text(text)
        
        # Step 3: Prepend summary and embed
        contextualized_chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Prepend summary to chunk for embedding
            ctx_text = f"{summary}\n\n{chunk}"
            
            # Embed the contextualized text
            response = client.embeddings.create(
                model=self.embedding_model,
                input=ctx_text
            )
            
            contextualized_chunks.append(ContextualChunk(
                chunk_id=f"chunk_{i}",
                text=chunk,  # Store ORIGINAL (for generation)
                embedding=response.data[0].embedding,
                token_span=(i * self.chunk_size, (i + 1) * self.chunk_size),
                summary_used=summary
            ))
        
        return contextualized_chunks
    
    def _generate_summary(self, text: str) -> str:
        """Generate a concise document summary."""
        response = client.chat.completions.create(
            model=self.summary_model,
            messages=[{
                "role": "system",
                "content": "Summarize this document in 100-150 tokens. Focus on key topics, entities, and purpose."
            }, {
                "role": "user",
                "content": text[:8000]  # Limit input for cost control
            }],
            max_tokens=200
        )
        return response.choices[0].message.content
# --8<-- [end:lst-9.8]

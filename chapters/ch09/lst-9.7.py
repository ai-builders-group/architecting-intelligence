# --8<-- [start:lst-9.7]
# path/to/semantic_splitter.py
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunk(BaseModel):
    text: str
    coherence_score: float = Field(ge=0.0, le=1.0)
    start_idx: int
    end_idx: int

class AdaptiveSemanticSplitter:
    """Split text by detecting semantic discontinuities using local minima.
    
    Includes a HARD LIMIT safety valve to prevent infinite semantic chunks
    that exceed embedding model constraints.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        max_tokens: int = 512,
        hard_limit_fallback: bool = True
    ):
        self.encoder = SentenceTransformer(model_name)
        self.max_tokens = max_tokens
        self.hard_limit_fallback = hard_limit_fallback
        
        # Fallback splitter for homogeneous chunks
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * 4,  # Rough token-to-char ratio
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split(self, text: str) -> List[SemanticChunk]:
        """Split text at points of low semantic similarity (local minima).
        
        Safety: If a semantic chunk exceeds max_tokens, recursively split
        using deterministic fallback to ensure embedding model compliance.
        """
        
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return [SemanticChunk(
                text=text, 
                coherence_score=1.0, 
                start_idx=0, 
                end_idx=len(sentences)
            )]
        
        # Embed all sentences
        embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
        
        # Calculate pairwise cosine similarity
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(float(sim))
        
        # Detect split points using local minima (gradient-based)
        split_points = [0]
        for i in range(1, len(similarities) - 1):
            # Local minimum: similarity dips below neighbors
            if similarities[
                i] < similarities[i-1] and similarities[i] < similarities[i+1]:
                # Adaptive threshold: only split if below 25th percentile
                threshold = np.percentile(similarities, 25)
                if similarities[i] < threshold:
                    split_points.append(i + 1)
        split_points.append(len(sentences))
        
        # Build chunks with HARD LIMIT enforcement
        chunks = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            
            # SAFETY VALVE: Check token count
            token_count = len(chunk_text.split())  # Rough estimate
            
            if token_count > self.max_tokens and self.hard_limit_fallback:
                # RECURSIVE FALLBACK: Use deterministic splitter
                sub_chunks = self.recursive_splitter.split_text(chunk_text)
                for sub_chunk in sub_chunks:
                    chunks.append(SemanticChunk(
                        text=sub_chunk,
                        coherence_score=0.5,  # Degraded (forced split)
                        start_idx=start,
                        end_idx=end
                    ))
            else:
                # Calculate coherence (average similarity within chunk)
                if end - start > 1:
                    coherence = np.mean([
                        similarities[j] 
                        for j in range(start, min(end-1, len(similarities)))
                    ])
                else:
                    coherence = 1.0
                
                chunks.append(SemanticChunk(
                    text=chunk_text,
                    coherence_score=float(coherence),
                    start_idx=start,
                    end_idx=end
                ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Sentence tokenization. Production: use spaCy."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
# --8<-- [end:lst-9.7]

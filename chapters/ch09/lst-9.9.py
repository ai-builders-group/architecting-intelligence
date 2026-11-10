# --8<-- [start:lst-9.9]
# path/to/late_chunking.py
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel, Field
import numpy as np

class ContextualChunk(BaseModel):
    """Chunk embedding that preserves global document context via hidden states."""
    chunk_id: str
    text: str
    embedding: List[float]
    token_span: tuple[int, int]
    vram_used_mb: float = Field(description="Peak VRAM for transparency")
    
    class Config:
        arbitrary_types_allowed = True

class MemorySafeLateChunker:
    """Late Chunking with VRAM monitoring and FlashAttention-2 support.
    
    WARNING: Requires local model weights and GPU. Not compatible with
    managed APIs (OpenAI, Anthropic, Cohere).
    """
    
    def __init__(
        self, 
        model_name: str = "jinaai/jina-embeddings-v3",
        chunk_size: int = 512,
        use_flash_attention: bool = True,
        model_weight_size_mb: int = 2000  # 2GB for typical embedding model
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attention else "eager"
        )
        self.chunk_size = chunk_size
        self.max_length = 8192
        
        # VRAM cap: refuse documents that would exceed GPU memory
        self.max_vram_mb = 12000  # T4 safe limit (adjust for A100: 40000)
        self.model_base_load_mb = model_weight_size_mb
        self.buffer_factor = 1.2  # CUDA context overhead
    
    def _estimate_vram(self, num_tokens: int) -> float:
        """Estimate peak VRAM usage including BASE MODEL FOOTPRINT.
        
        Flash Attention: O(N) memory (linear)
        Standard Attention: O(N^2) memory (quadratic)
        
        Accounts for:
        - Model weights (loaded once)
        - Activation memory (per forward pass)
        - KV Cache (if applicable)
        - CUDA context (buffer factor)
        """
        d = 768  # Hidden dimension
        
        # Attention memory
        if hasattr(self.model.config, '_attn_implementation') and \
           self.model.config._attn_implementation == "flash_attention_2":
            # FlashAttention-2: no N x N matrix, only fp16 output states
            attention_mb = (num_tokens * d * 2) / (1024 ** 2)  # fp16 = 2 bytes
        else:
            # Quadratic attention: stores full attention matrix
            attention_mb = (num_tokens ** 2 * 4) / (1024 ** 2)
        
        # Total VRAM = Base Load + Activations + Buffer
        total_vram = (self.model_base_load_mb + attention_mb) * self.buffer_factor
        
        return total_vram
    
    def encode_document(self, text: str) -> List[ContextualChunk]:
        """Encode full document then slice embeddings into chunks.
        
        Raises:
            MemoryError: If estimated VRAM exceeds safe limit.
        """
        
        # Tokenize full document
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
            return_offsets_mapping=True
        )
        
        num_tokens = inputs['input_ids'].shape[1]
        vram_estimate = self._estimate_vram(num_tokens)
        
        # Hard gate: refuse to process if VRAM would exceed limit
        if vram_estimate > self.max_vram_mb:
            raise MemoryError(
                f"Document requires {vram_estimate:.0f}MB VRAM (limit: {self.max_vram_mb}MB). "
                f"Model base load: {self.model_base_load_mb}MB. "
                f"Use Strategy 3 (Contextual Prepending) or enable FlashAttention-2."
            )
        
        # Forward pass: Get contextualized token embeddings
        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
            token_embeddings = outputs.last_hidden_state[0]  # Shape: (num_tokens, hidden_size)
        
        # Decode token spans back to text
        offset_mapping = inputs['offset_mapping'][0]
        
        # Slice into chunks
        chunks = []
        num_tokens_actual = token_embeddings.shape[0]
        
        for i in range(0, num_tokens_actual, self.chunk_size):
            end_idx = min(i + self.chunk_size, num_tokens_actual)
            
            # Mean-pool token embeddings for this chunk
            chunk_embedding = token_embeddings[i:end_idx].mean(dim=0)
            
            # Reconstruct text from token offsets
            start_char = offset_mapping[i][0].item()
            end_char = offset_mapping[end_idx - 1][1].item()
            chunk_text = text[start_char:end_char]
            
            chunks.append(ContextualChunk(
                chunk_id=f"chunk_{i // self.chunk_size}",
                text=chunk_text,
                embedding=chunk_embedding.cpu().numpy().tolist(),
                token_span=(i, end_idx),
                vram_used_mb=vram_estimate
            ))
        
        return chunks
# --8<-- [end:lst-9.9]

# --8<-- [start:lst-9.30]
# path/to/dynamic_allocator.py
from pydantic import BaseModel, Field
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Component:
    """A candidate for inclusion in the context window."""
    content: str
    token_count: int
    relevance_score: float  # 0.0 to 1.0
    priority: LayerPriority
    source: str  # "rag" | "history" | "scratchpad"

class AllocationResult(BaseModel):
    """The assembled context payload with provenance."""
    system_prompt: str
    knowledge_chunks: List[str]
    memory_messages: List[str]
    user_query: str
    total_tokens: int
    utilization: float = Field(ge=0.0, le=1.0)
    evicted_count: int

class DynamicAllocator:
    """Priority-based greedy packing for context window assembly."""
    
    def __init__(self, budget: ContextBudget):
        self.budget = budget
    
    def allocate(
        self,
        system_prompt: str,
        user_query: str,
        rag_results: List[Tuple[str, float]],  # (content, relevance_score)
        history: List[str],
        scratchpad: str = ""
    ) -> AllocationResult:
        """Assemble context using priority-based packing.
        
        Returns:
            AllocationResult with the final context payload and metadata.
        """
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")  # Correct tokenizer for GPT-4o
        
        def count_tokens(text: str) -> int:
            return len(enc.encode(text))
        
        # Reserve space for immutable components
        system_tokens = count_tokens(system_prompt)
        query_tokens = count_tokens(user_query)
        
        available = (
            self.budget.available_budget  # already net of reserved_system
            - query_tokens
        )
        
        # Convert to components
        components: List[Component] = []
        
        # RAG chunks (Priority 2)
        for chunk, score in rag_results:
            components.append(Component(
                content=chunk,
                token_count=count_tokens(chunk),
                relevance_score=score,
                priority=LayerPriority.KNOWLEDGE,
                source="rag"
            ))
        
        # History messages (Priority 3, decaying by recency)
        for i, msg in enumerate(reversed(history)):
            recency_score = 1.0 / (i + 1)  # Most recent = 1.0
            components.append(Component(
                content=msg,
                token_count=count_tokens(msg),
                relevance_score=recency_score,
                priority=LayerPriority.MEMORY,
                source="history"
            ))
        
        # Scratchpad (Priority 3)
        if scratchpad:
            components.append(Component(
                content=scratchpad,
                token_count=count_tokens(scratchpad),
                relevance_score=0.8,  # Medium relevance
                priority=LayerPriority.MEMORY,
                source="scratchpad"
            ))
        
        # Sort by priority, then by relevance within priority
        components.sort(key=lambda c: (c.priority.value, -c.relevance_score))
        
        # Greedy packing
        packed_knowledge = []
        packed_memory = []
        remaining = available
        evicted = 0
        
        for comp in components:
            if comp.token_count <= remaining:
                if comp.source == "rag":
                    packed_knowledge.append(comp.content)
                else:
                    packed_memory.append(comp.content)
                remaining -= comp.token_count
            else:
                evicted += 1
        
        total_used = (
            system_tokens 
            + query_tokens 
            + sum(count_tokens(c) for c in packed_knowledge)
            + sum(count_tokens(c) for c in packed_memory)
        )
        
        return AllocationResult(
            system_prompt=system_prompt,
            knowledge_chunks=packed_knowledge,
            memory_messages=packed_memory,
            user_query=user_query,
            total_tokens=total_used,
            utilization=total_used / self.budget.total_context,
            evicted_count=evicted
        )
# --8<-- [end:lst-9.30]

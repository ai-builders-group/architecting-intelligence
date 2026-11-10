# --8<-- [start:lst-1.8]
# src/agents/l0_chatbot.py
from pydantic import BaseModel, Field
from typing import Literal

class ChatbotInput(BaseModel):
    """Strictly typed input for L0 agents."""
    user_message: str = Field(min_length=1, max_length=5000)
    context_limit: int = Field(default=2000, ge=100, le=8000)

class ChatbotOutput(BaseModel):
    """Strictly typed output with no action capability."""
    response: str = Field(description="The generated text response")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Self-reported confidence"
    )
    tokens_used: int = Field(ge=0)
    
class L0Agent:
    """
    A stateless, actionless conversational agent.
    No tool access. No state mutation.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    
    def respond(self, input: ChatbotInput) -> ChatbotOutput:
        """
        Pure text generation with no side effects.
        
        Args:
            input: Validated user input
            
        Returns:
            ChatbotOutput with generated text
        """
        # Pure text generation - no tools
        response = self._call_llm(input.user_message)
        return ChatbotOutput(
            response=response,
            confidence=0.8,  # Mock confidence
            tokens_used=150
        )
    
    def _call_llm(self, message: str) -> str:
        """Placeholder for actual LLM call."""
        return f"Echo: {message}"
# --8<-- [end:lst-1.8]

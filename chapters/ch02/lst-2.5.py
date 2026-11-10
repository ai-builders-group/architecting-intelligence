from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Callable, Any, TypeVar, Generic, Protocol
from enum import Enum

# Type-safe Tool Definition Protocol
class Tool(Protocol):
    """Protocol for runtime-validated tools."""
    def __call__(self, **kwargs: Any) -> Any: ...
    @property
    def __doc__(self) -> str: ...  # Docstring is MANDATORY

# The Five Dimensions of Agency
class AgenticContract(ABC):
    """
    The Interface Definition Language (IDL) for an AI Agent.
    This defines the BOUNDARY, not the prompt.
    
    Versioning: Breaking changes to this contract require a semantic version bump.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The unique identifier for this cognitive unit.
        Format: {domain}.{capability}.{version}
        Example: "billing.refund_processor.v2"
        """
        pass

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """
        1. THE TRIGGER (Compile-Time Definition)
        
        The Pydantic model that defines the payload structure that activates
        this agent. Runtime events MUST conform to this schema or the
        transaction is rejected at the boundary.
        
        Breaking Change Example:
          v1: order_id: Optional[str]
          v2: order_id: str  # Now required -> MAJOR version bump
        """
        pass

    @property
    @abstractmethod
    def context_scope(self) -> List[str]:
        """
        2. THE STATE (Principle of Least Privilege)
        
        Specific keys from the global state this agent is authorized to read.
        This is an allowlist, not a denylist. If a key is not listed, the
        agent cannot access it—even if the model "asks" for it.
        
        Example: ["user.id", "user.tier", "billing.history"]
        """
        pass

    @property
    @abstractmethod
    def tools(self) -> List[Tool]:
        """
        3. THE ACTION SPACE (Executable Functions)
        
        A whitelist of callable functions. The LLM can ONLY execute these.
        
        CRITICAL INVARIANT: The docstring of each function is not documentation;
        it is LOGIC. The LLM uses docstrings to infer arguments and behavior.
        
        Anti-Pattern:
          def calculate_tax(amount):
              '''Calculates tax.'''
              
        Correct Pattern:
          def calculate_tax(amount: float) -> float:
              '''
              Calculates US sales tax for a purchase.
              
              Args:
                  amount (float): Purchase amount in USD. Must be positive.
                  
              Returns:
                  float: Tax amount in USD (amount * 0.08).
                  
              Raises:
                  ValueError: If amount is negative or exceeds $1,000,000.
              '''
        
        The verbosity is not for humans—it is for the model's parameter space.
        """
        pass

    @property
    @abstractmethod
    def guardrails(self) -> List[Callable[[str], bool]]:
        """
        4. THE GUARDRAILS (Deterministic Invariants)
        
        A list of validation functions applied to the model's output BEFORE
        execution. Each function receives the raw output string and returns:
          - True: Output is safe. Proceed.
          - False: Output violates policy. Reject and retry.
        
        These are NOT probabilistic checks (like toxicity classifiers).
        These are HARD rules (regex, schema validation, business logic).
        
        Example:
          lambda output: "DROP TABLE" not in output.upper()
        """
        pass

    @property
    @abstractmethod
    def exit_criteria(self) -> Dict[str, float | int]:
        """
        5. EXIT CRITERIA (Definition of Done)
        
        Quantitative thresholds that determine when the agent terminates.
        
        Standard Keys:
          - confidence_threshold: float (0.0-1.0)
          - max_retries: int
          - max_tokens: int
          - timeout_seconds: float
        
        Example:
          {
              'confidence_threshold': 0.85,
              'max_retries': 3,
              'max_tokens': 2048,
              'timeout_seconds': 30.0
          }
        """
        pass

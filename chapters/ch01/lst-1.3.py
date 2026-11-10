# --8<-- [start:lst-1.3]
# src/core/stochastic_result.py
import re
from typing import ClassVar, Generic, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field
T = TypeVar("T", bound=BaseModel)
class StochasticResult(BaseModel, Generic[T]):
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    threshold: ClassVar[float] = 0.85
    FORBIDDEN_PHRASES: ClassVar[list[str]] = ["I'm not sure"]
    data: Optional[T] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning_trace: list[str] = Field(default_factory=list)
    @property
    def is_reliable(self) -> bool:
        if self.data is None:
            return False
        if self.confidence_score < self.threshold:
            return False
        trace = " ".join(self.reasoning_trace)
        for phrase in self.FORBIDDEN_PHRASES:
            if re.search(rf"\b{re.escape(phrase)}\b", trace, re.IGNORECASE):
                return False
        return True
# --8<-- [end:lst-1.3]

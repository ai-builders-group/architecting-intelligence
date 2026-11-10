# --8<-- [start:lst-9.17]
# path/to/intent_extractor.py
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from openai import OpenAI

client = OpenAI()

class ExtractedFilter(BaseModel):
    """
    A single filter predicate extracted from natural language.
    """
    field: Literal["department", "document_date", "source_type", "author", "security_classification"]
    operator: Literal["==", ">=", "<=", "in", "contains"]
    value: str | datetime | list[str]
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    reasoning: str = Field(description="Why this filter was extracted")

class IntentExtractionResult(BaseModel):
    """
    Complete intent analysis with confidence-gated filtering strategy.
    """
    extracted_filters: list[ExtractedFilter] = Field(default_factory=list)
    query_intent: str = Field(description="High-level intent summary")
    filter_strategy: Literal["hard", "soft", "none"] = Field(
        description="hard=strict exclusion, soft=boosting, none=no filtering"
    )
    overall_confidence: float = Field(ge=0.0, le=1.0)
    fallback_to_semantic: bool = Field(
        default=False,
        description="True if LLM extraction failed, use semantic routing"
    )

class ConfidenceGatedIntentExtractor:
    """
    Tier 2 LLM-based intent extraction with structured output and soft filtering fallback.
    Prevents Silent Retrieval Failure by defaulting to boosting on low confidence.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        confidence_threshold: float = 0.9,
        mapper: Optional[DepartmentMapper] = None
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.mapper = mapper or DepartmentMapper()
    
    def extract_intent(self, user_query: str, corpus_stats: dict) -> IntentExtractionResult:
        """
        Extract filters using structured output with logit-constrained decoding.
        """
        # System prompt with explicit confidence scoring requirement
        system_prompt = """You are a metadata filter extraction system. 
        
Your job: Extract structured filters from natural language queries.

CRITICAL RULES:
1. Only extract filters you are >90% confident about.
2. For dates, extract ONLY if explicitly mentioned or strongly implied.
3. For departments, use canonical names: platform, product, finance, legal, hr, sales, marketing.
4. If uncertain about ANY filter, set confidence < 0.9 and explain why in reasoning.
5. NEVER guess - low confidence is better than wrong filters.

Available fields:
- department: string (canonical names only)
- document_date: datetime (>=, <=, ==)
- source_type: pdf | markdown | html | api | email
- author: string
- security_classification: public | internal | confidential | restricted
"""
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {user_query}\n\nExtract filters:"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "extracted_filters": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {"type": "string", "enum": ["department", "document_date", "source_type", "author", "security_classification"]},
                                        "operator": {"type": "string", "enum": ["==", ">=", "<=", "in", "contains"]},
                                        "value": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "reasoning": {"type": "string"}
                                    },
                                    "required": ["field", "operator", "value", "confidence", "reasoning"],
                                    "additionalProperties": False
                                }
                            },
                            "query_intent": {"type": "string"},
                            "overall_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["extracted_filters", "query_intent", "overall_confidence"],
                        "additionalProperties": False
                    }
                }
            },
            max_tokens=500
        )
        
        # Parse response
        import json
        result_dict = json.loads(response.choices[0].message.content)
        
        # Apply department mapping
        extracted_filters = []
        for f in result_dict["extracted_filters"]:
            if f["field"] == "department":
                canonical = self.mapper.map_to_canonical(f["value"], "2.0.0")
                if canonical is None:
                    # Unmapped department - lower confidence
                    f["confidence"] *= 0.5
                    f["reasoning"] += " (Warning: Department not in canonical taxonomy)"
                else:
                    f["value"] = canonical
            
            # Parse datetime if needed
            if f["field"] == "document_date":
                try:
                    f["value"] = datetime.fromisoformat(f["value"])
                except ValueError:
                    f["confidence"] = 0.0  # Parsing failed
            
            extracted_filters.append(ExtractedFilter(**f))
        
        # Determine filter strategy based on confidence
        overall_confidence = result_dict["overall_confidence"]
        min_filter_confidence = min(
            [f.confidence for f in extracted_filters], default=1.0)
        
        if (
            min_filter_confidence >= self.confidence_threshold
            and overall_confidence >= self.confidence_threshold
        ):
            strategy = "hard"
        elif min_filter_confidence >= 0.6:
            strategy = "soft"  # Use filters as boosting weights, not exclusions
        else:
            strategy = "none"  # Too uncertain - fall back to pure semantic search
        
        return IntentExtractionResult(
            extracted_filters=extracted_filters,
            query_intent=result_dict["query_intent"],
            filter_strategy=strategy,
            overall_confidence=overall_confidence,
            fallback_to_semantic=(strategy == "none")
        )
# --8<-- [end:lst-9.17]

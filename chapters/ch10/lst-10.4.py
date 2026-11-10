# --8<-- [start:lst-10.4]
# src/08_retrieval_topology/grader.py
from pydantic import BaseModel, Field
from enum import Enum

class RelevanceGrade(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    AMBIGUOUS = "ambiguous"

class GraderResult(BaseModel):
    """Structured output from the grader LLM."""
    reasoning: str = Field(
        description="Chain-of-thought explanation, generated before the grade")
    grade: RelevanceGrade
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model's self-reported certainty")
    reasoning: str = Field(description="Chain-of-thought explanation")

async def grade_retrieval(
    query: str,
    document: str,
    grader_model: str = "gpt-4o-mini",
) -> GraderResult:
    """Use a small, fast model to grade relevance."""
    
    prompt = f"""You are a relevance grader. Determine if the document is relevant to the query.

Query: {query}

Document:
{document[:1000]}

Output your assessment as JSON:
{{
  "grade": "relevant" | "irrelevant" | "ambiguous",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""
    
    # Stub: Replace with actual LLM call + structured output parsing
    # response = await openai.ChatCompletion.acreate(
    #     model=grader_model,
    #     messages=[{"role": "user", "content": prompt}],
    #     response_format={"type": "json_object"},
    # )
    
    # For now, return a mock result
    return GraderResult(
        grade=RelevanceGrade.RELEVANT,
        confidence=0.92,
        reasoning="The document explicitly addresses the query topic.",
    )
# --8<-- [end:lst-10.4]

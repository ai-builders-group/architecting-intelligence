# --8<-- [start:lst-5.9]
# src/04_probabilistic_state_machine/nodes/critic.py
from typing import List, Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()

class ThoughtStep(BaseModel):
    """
    Minimal stand-in for the chapter's ThoughtStep.
    """
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    critique: Optional[str] = None

class AgenticState(BaseModel):
    """
    Minimal stand-in for the chapter's loop state.
    """
    user_query: str
    thoughts: List[ThoughtStep] = Field(default_factory=list)
    retry_count: int = 0
    is_complete: bool = False
    requires_human: bool = False
    total_tokens: int = 0
    total_cost_usd: float = 0.0

class CritiqueResult(BaseModel):
    """
    Structured output from the Critic node.
    """
    is_sufficient: bool = Field(
        description="Does the latest observation resolve the user's query?"
    )
    feedback: str = Field(
        description="Explanation for the human and for retry context."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Critic's confidence in the assessment."
    )

def critic_node(state: AgenticState) -> dict:
    """
    Evaluate whether the latest ReAct cycle resolved the user's query.
    
    If insufficient, append feedback to the most recent ThoughtStep.
    This feedback will be injected into the Planner's context on retry.
    
    Design Constraints:
    - MUST run at temperature=0 for determinism.
    - MUST use a Tier 1 model (cost optimization).
    - MUST NOT hallucinate "Pass" verdicts (false positives are catastrophic).
    
    Returns:
        One update dict for the state; the incoming state is never changed.
    """
    
    # Edge case: No thoughts yet (should not happen if graph is well-formed)
    if not state.thoughts:
        return {"requires_human": True}
    
    latest_thought = state.thoughts[-1]
    
    # Construct the critique prompt
    prompt = f"""You are a strict evaluator. Assess whether the following ReAct cycle resolves the user's query.

USER QUERY:
{state.user_query}

AGENT'S THOUGHT:
{latest_thought.thought}

ACTION TAKEN:
{latest_thought.action or "None"}

OBSERVATION:
{latest_thought.observation or "None"}

RUBRIC:
- Does the observation directly answer the user's query?
- Is the data complete and accurate?
- Is there any ambiguity or missing information?

Respond ONLY with valid JSON matching this schema:
{{
  "is_sufficient": boolean,
  "feedback": "string (explanation)",
  "confidence": float (0.0 to 1.0)
}}
"""
    
    # Invoke the Critic (Tier 1 model)
    response = client.chat.completions.parse(
        model="gpt-4o-mini",  # small Tier 1 model
        messages=[{"role": "user", "content": prompt}],
        response_format=CritiqueResult,
        temperature=0  # Low-variance grading
    )
    
    critique = response.choices[0].message.parsed
    if critique is None:  # the model refused
        return {"retry_count": state.retry_count + 1}
    
    # Update token/cost tracking
    total_tokens = state.total_tokens + response.usage.total_tokens
    total_cost_usd = state.total_cost_usd + (
        response.usage.prompt_tokens * 0.15 / 1_000_000 +
        response.usage.completion_tokens * 0.60 / 1_000_000
    )
    
    # Replace the latest thought with a validated copy that carries the critique
    new_thoughts = state.thoughts[:-1] + [
        ThoughtStep.model_validate(
            {**latest_thought.model_dump(), "critique": critique.feedback}
        )
    ]
    update = {"thoughts": new_thoughts}
    update["total_tokens"] = total_tokens
    update["total_cost_usd"] = total_cost_usd
    
    # If insufficient, increment retry counter
    if not critique.is_sufficient:
        update["retry_count"] = state.retry_count + 1
    else:
        # Mark as complete (will be confirmed by Terminator)
        update["is_complete"] = True
    
    return update
# --8<-- [end:lst-5.9]

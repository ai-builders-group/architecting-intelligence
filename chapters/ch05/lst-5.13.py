# --8<-- [start:lst-5.13]
# src/04_probabilistic_state_machine/nodes/reflection_evaluator.py
from __future__ import annotations
import json
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()

class Critique(BaseModel):
    attempt: int
    generated_output: str | None
    evaluation_result: str
    reflection: str

class SemanticEvaluation(BaseModel):
    """
    Structured output from the semantic validator.
    """
    is_correct: bool = Field(description="Does the output correctly solve the task?")
    explanation: str = Field(description="Reasoning for the verdict.")

def reflection_evaluator_node(state: ReflectionState) -> dict:
    """
    Two-stage validator: Deterministic + Stochastic.
    
    Stage 1: Syntactic checks (fast, cheap).
    Stage 2: Semantic checks (slow, expensive).
    
    Returns:
        State update dict with evaluation result and critique appended.
    """
    
    generated_output = state.validated_result
    evaluation_result = "PASS"
    reflection = ""
    
    # --- Stage 1: Deterministic Checks ---
    # Example: If the task requires JSON, validate it
    if "json" in state.task_description.lower():
        try:
            json.loads(generated_output)
        except json.JSONDecodeError as e:
            evaluation_result = f"FAIL: Invalid JSON - {e}"
            reflection = "The output is not valid JSON. Ensure proper formatting."
            # Return the critique as an update and exit early
            return {
                "validated_result": None,
                "evaluation_result": evaluation_result,
                "reflection": reflection,
                "attempt": state.attempt + 1,
                "critiques": state.critiques + [Critique(
                    attempt=state.attempt,
                    generated_output=generated_output,
                    evaluation_result=evaluation_result,
                    reflection=reflection
                )]
            }
    
    # --- Stage 2: Semantic Validation (LLM-as-Judge) ---
    prompt = f"""Evaluate the following output for correctness.

Task Description:
{state.task_description}

Generated Output:
{generated_output}

Respond ONLY with valid JSON:
{{
  "is_correct": boolean,
  "explanation": "string"
}}
"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # Tier 1 validator
        messages=[{"role": "user", "content": prompt}],
        response_format=SemanticEvaluation,
        temperature=0
    )
    
    eval_result = response.choices[0].message.parsed
    
    # Update token/cost tracking
    total_tokens = state.total_tokens + response.usage.total_tokens
    total_cost_usd = state.total_cost_usd + (
        response.usage.prompt_tokens * 0.15 / 1_000_000 +
        response.usage.completion_tokens * 0.60 / 1_000_000
    )
    
    if not eval_result.is_correct:
        evaluation_result = f"FAIL: Semantic error"
        reflection = eval_result.explanation
        
        # Return the critique as an update
        return {
            "validated_result": None,
            "evaluation_result": evaluation_result,
            "reflection": reflection,
            "critiques": state.critiques + [Critique(
                attempt=state.attempt,
                generated_output=generated_output,
                evaluation_result=evaluation_result,
                reflection=reflection
            )],
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd
        }
    
    # Success! Keep the validated result
    return {
        "evaluation_result": evaluation_result,
        "reflection": reflection,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd
    }
# --8<-- [end:lst-5.13]

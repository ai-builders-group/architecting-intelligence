# --8<-- [start:lst-5.12]
# src/04_probabilistic_state_machine/nodes/reflection_generator.py
from __future__ import annotations
from openai import OpenAI

client = OpenAI()

def reflection_generator_node(state: ReflectionState) -> ReflectionState:
    """
    Generate an output with critique history as context.
    
    Design:
    - On first attempt: Use only the task description.
    - On retry: Append all previous critiques to the prompt.
    
    Returns:
        Updated state with the new generated output.
    """
    
    # Construct the prompt
    prompt = f"Task: {state.task_description}\n\n"
    
    if state.critiques:
        prompt += "Previous Attempts and Critiques:\n"
        for i, critique in enumerate(state.critiques, start=1):
            prompt += f"\n--- Attempt {i} ---\n"
            prompt += f"Your Output: {critique.generated_output}\n"
            prompt += f"Evaluation: {critique.evaluation_result}\n"
            prompt += f"Reflection: {critique.reflection}\n"
        
        prompt += (
            "\n\nBased on the above feedback, generate a corrected output.\n"
        )
    else:
        prompt += "Generate your best output.\n"
    
    # Invoke the model (Tier 2)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    generated_output = response.choices[0].message.content
    
    # Update state
    state.attempt += 1
    state.total_tokens += response.usage.total_tokens
    state.total_cost_usd += (
        response.usage.prompt_tokens * 0.15 / 1_000_000 +
        response.usage.completion_tokens * 0.60 / 1_000_000
    )
    
    # Store the output temporarily (will be validated next)
    state.validated_result = generated_output
    
    return state
# --8<-- [end:lst-5.12]

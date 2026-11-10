# --8<-- [start:lst-5.4]
# src/04_probabilistic_state_machine/benchmarks/instruction_dilution.py
from pydantic import BaseModel, Field
from typing import List
import anthropic
import numpy as np

class DilutionTestCase(BaseModel):
    """
    Test case for measuring instruction adherence under context pressure.
    """
    instruction: str = Field(
        description="The critical constraint (e.g., 'Always cite sources')."
    )
    context_length: int = Field(
        description="Target input size in TOKENS (as counted by count_tokens), not words or characters."
    )
    query: str = Field(
        description="The user query."
    )
    
class DilutionResult(BaseModel):
    """
    Measurement of instruction adherence.
    """
    context_length: int
    instruction_followed: bool
    response: str

def run_dilution_test(
    test_case: DilutionTestCase,
    model: str = "claude-3-5-sonnet-20241022"
) -> DilutionResult:
    """
    Inject distractor tokens and measure if the model follows the instruction.
    
    Args:
        test_case: The test configuration.
        model: The model to test.
        
    Returns:
        DilutionResult with adherence measurement.
    """
    client = anthropic.Anthropic()
    
    # context_length is in TOKENS: grow the filler until the counted input
    # reaches it
    def build_messages(n_words: int) -> list:
        distractor = " ".join(["irrelevant"] * n_words)
        return [
            {
                "role": "user",
                "content": f"""INSTRUCTION: {test_case.instruction}

CONTEXT:
[doc-1] {distractor}

QUERY: {test_case.query}"""
            }
        ]
    
    n_words = 0
    while True:
        messages = build_messages(n_words)
        counted = client.messages.count_tokens(
            model=model, messages=messages
        ).input_tokens
        if counted >= test_case.context_length:
            break
        n_words += max((test_case.context_length - counted) // 2, 1)
    
    # Call the model
    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
        messages=messages
    )
    
    response_text = response.content[0].text
    recorded = response.usage.input_tokens
    
    # Check adherence: does the response cite the injected document?
    instruction_followed = "[doc-1]" in response_text
    
    return DilutionResult(
        context_length=test_case.context_length,
        instruction_followed=instruction_followed,
        response=response_text
    )

def benchmark_dilution(
    context_lengths: List[int] = [1000, 5000, 10000, 20000, 30000]
) -> List[DilutionResult]:
    """
    Run the dilution test across varying context lengths.
    
    Returns:
        List of results for analysis.
    """
    test_cases = [
        DilutionTestCase(
            instruction="Always cite sources using [Source: X] format.",
            context_length=length,
            query="What is the capital of France?"
        )
        for length in context_lengths
    ]
    
    results = [run_dilution_test(tc) for tc in test_cases]
    
    # Calculate adherence rate
    adherence_rate = np.mean([r.instruction_followed for r in results])
    print(f"Overall adherence rate: {adherence_rate:.1%}")
    
    return results

# --8<-- [end:lst-5.4]

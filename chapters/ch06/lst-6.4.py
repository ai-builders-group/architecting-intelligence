# --8<-- [start:lst-6.4]
# architecting-intelligence/src/06_temporal/activity_pattern.py
from temporalio import activity
from openai import AsyncOpenAI
from typing import Optional
import hashlib

@activity.defn
async def invoke_llm(prompt: str, model: str) -> str:
    """
    Activity: Call OpenAI API.
    
    This is non-deterministic (model outputs vary), so it MUST be an Activity,
    not inline workflow code.
    """
    client = AsyncOpenAI()
    
    # Generate idempotency key from activity context
    # Temporal provides a unique activity ID that is stable across retries
    info = activity.info()
    idempotency_key = hashlib.sha256(
        f"{info.workflow_id}:{info.workflow_run_id}:"
        f"{info.activity_id}:{prompt}".encode()
    ).hexdigest()
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # Deduplicates only if the provider honors this header; verify, else
        # dedup in your own store (this narrows the window, it does not close it)
        extra_headers={"X-Idempotency-Key": idempotency_key}
    )
    
    return response.choices[0].message.content


@activity.defn
async def call_risk_api(transaction_id: str) -> float:
    """
    Activity: Call external fraud detection API.
    
    The API response may change over time (risk scores are dynamic),
    but we want the workflow to make decisions based on the risk score
    *at the time of original execution*, not at replay time.
    """
    # Simulate API call
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://risk-api.example.com/score/{transaction_id}"
        )
        return response.json()["risk_score"]

# --8<-- [end:lst-6.4]

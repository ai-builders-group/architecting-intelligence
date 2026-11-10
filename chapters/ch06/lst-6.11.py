# --8<-- [start:lst-6.11]
# architecting-intelligence/src/06_temporal/llm_activity_production.py
from temporalio import activity, workflow
from temporalio.exceptions import ApplicationError
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, RateLimitError, APITimeoutError
from typing import Literal
from datetime import timedelta
import hashlib
import structlog

logger = structlog.get_logger()

class LLMRequest(BaseModel):
    """Strict schema for LLM activity input."""
    prompt: str = Field(min_length=1, max_length=100_000)
    model: Literal["gpt-4o", "gpt-4o-mini", "o1-preview"]
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    max_tokens: int = Field(ge=1, le=16_000, default=4096)

class LLMResponse(BaseModel):
    """Strict schema for LLM activity output."""
    content: str
    token_count: int
    finish_reason: Literal["stop", "length", "content_filter"]
    model_version: str  # e.g., "gpt-4o-2024-08-06"

@activity.defn
async def invoke_llm_with_retry(request: LLMRequest) -> LLMResponse:
    """
    Production-grade LLM activity.
    
    Implements:
    - Idempotency via activity ID hashing
    - Automatic retry on rate limits
    - Observability via structured logging
    - Type safety via Pydantic
    """
    # Get Temporal activity metadata
    info = activity.info()
    
    # Generate deterministic idempotency key
    # The activity ID is stable across retries within one run; the run ID
    # separates runs
    idempotency_key = hashlib.sha256(
        f"{info.workflow_id}:{info.workflow_run_id}:"
        f"{info.activity_id}:{request.prompt}".encode()
    ).hexdigest()[:32]
    
    logger.info(
        "llm_activity_started",
        workflow_id=info.workflow_id,
        activity_id=info.activity_id,
        model=request.model,
        idempotency_key=idempotency_key
    )
    
    client = AsyncOpenAI()
    
    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            # Deduplicates only if the provider honors this header; verify, else
            # dedup in your own store (this narrows the window, it
            # does not close it)
            extra_headers={"X-Idempotency-Key": idempotency_key}
        )
        
        result = LLMResponse(
            content=response.choices[0].message.content,
            token_count=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason,
            model_version=response.model
        )
        
        logger.info(
            "llm_activity_completed",
            workflow_id=info.workflow_id,
            tokens=result.token_count,
            finish_reason=result.finish_reason
        )
        
        return result
        
    except RateLimitError as e:
        # Temporal will automatically retry this activity
        # Log for observability
        logger.warning(
            "llm_rate_limit",
            workflow_id=info.workflow_id,
            retry_count=info.attempt
        )
        # Re-raise to trigger Temporal's retry policy
        raise
        
    except APITimeoutError as e:
        logger.error(
            "llm_timeout",
            workflow_id=info.workflow_id,
            timeout_seconds=30
        )
        raise ApplicationError(
            "LLM API timeout",
            non_retryable=False  # Allow retry
        )

# Workflow that uses this activity
@workflow.defn
class ContentModerationWorkflow:
    @workflow.run
    async def run(self, user_content: str) -> str:
        # Schedule LLM activity with aggressive retry policy
        moderation_result = await workflow.execute_activity(
            invoke_llm_with_retry,
            LLMRequest(
                prompt=f"Classify this content as safe/unsafe: {user_content}",
                model="gpt-4o-mini",  # Cheap model for simple classification
                temperature=0.0,  # Deterministic
                max_tokens=10
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=5,
                backoff_coefficient=2.0
            )
        )
        
        # Deterministic decision based on LLM output
        if "unsafe" in moderation_result.content.lower():
            return "BLOCKED"
        else:
            return "APPROVED"
# --8<-- [end:lst-6.11]

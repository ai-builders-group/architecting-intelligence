# --8<-- [start:lst-9.35]
# path/to/summary_worker.py
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from datetime import timedelta
from openai import AsyncOpenAI

client = AsyncOpenAI()

@activity.defn
async def summarize_history_activity(
        tenant_id: str, session_id: str, history_ids: list[str]) -> str:
    """
    Activity: Summarize old history using the Processor-tier model.
    Runs in background; user never waits for this.
    """
    # Fetch messages
    messages = await db.get_messages_by_ids(history_ids)
    history_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
    
    response = await client.chat.completions.create(
        model="processor-model-id",  # your provider's IDs
        messages=[{
            "role": "system",
            "content": (
                "Compress this conversation history into 3-5 bullet points. "
                "Preserve critical facts (IDs, dates, decisions) but omit small talk."
            )
        }, {
            "role": "user",
            "content": history_text
        }],
        max_tokens=300
    )
    
    summary = response.choices[0].message.content
    
    # Update database (does not affect current inference)
    await db.update_session_summary(tenant_id, session_id, summary)
    
    return summary

@workflow.defn
class TopicFlushWorkflow:
    """
    Durable workflow for asynchronous history summarization.
    Survives process failures; does not block user queries.
    """
    @workflow.run
    async def run(self, tenant_id: str, session_id: str,
                  history_ids: list[str]) -> str:
        summary = await workflow.execute_activity(
            summarize_history_activity,
            args=[tenant_id, session_id, history_ids],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        return summary
# --8<-- [end:lst-9.35]

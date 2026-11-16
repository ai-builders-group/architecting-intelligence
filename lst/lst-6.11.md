# Typed model activity with retries (lst-6.11)

Defines validated model requests and responses, handles retryable API errors, and uses the activity in a content-moderation workflow.

- **Chapter:** 06
- **Class:** Needs service temporal
- **Needs:** Python 3.12+; `openai`, `pydantic`, `structlog`, `temporalio` (`pip install -r requirements/all.txt`); a Temporal server (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch06/lst-6.11.py`
- **File:** [chapters/ch06/lst-6.11.py](../chapters/ch06/lst-6.11.py)

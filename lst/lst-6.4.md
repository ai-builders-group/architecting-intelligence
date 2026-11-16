# Temporal activities for external calls (lst-6.4)

Defines Temporal activities for model invocation and risk lookup, adding a retry-stable idempotency header to the model request.

- **Chapter:** 06
- **Class:** Needs service temporal
- **Needs:** Python 3.12+; `httpx`, `openai`, `temporalio` (`pip install -r requirements/all.txt`); a Temporal server (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch06/lst-6.4.py`
- **File:** [chapters/ch06/lst-6.4.py](../chapters/ch06/lst-6.4.py)

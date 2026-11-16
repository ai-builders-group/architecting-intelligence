# Flight booking and cancellation activities (lst-6.15)

Defines Temporal activities that call booking and cancellation endpoints with derived idempotency keys and return booking or refund details.

- **Chapter:** 06
- **Class:** Needs service temporal
- **Needs:** Python 3.12+; `httpx`, `pydantic`, `temporalio` (`pip install -r requirements/temporal.txt`); a Temporal server (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch06/lst-6.15.py`
- **File:** [chapters/ch06/lst-6.15.py](../chapters/ch06/lst-6.15.py)

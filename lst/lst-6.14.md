# Saga orchestration with reverse compensation (lst-6.14)

Executes saga activities sequentially, compensates completed and in-flight steps in reverse order after failure, and escalates compensation failures.

- **Chapter:** 06
- **Class:** Needs service temporal
- **Needs:** Python 3.12+; `pydantic`, `structlog`, `temporalio` (`pip install -r requirements/temporal.txt`); a Temporal server (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch06/lst-6.14.py`
- **File:** [chapters/ch06/lst-6.14.py](../chapters/ch06/lst-6.14.py)

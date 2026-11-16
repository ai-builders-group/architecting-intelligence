# Bounded tool execution with correction (lst-1.12)

Retries transient tool errors within a capped attempt budget, records correction steps, and returns success, aborted, or failed-safe status.

- **Chapter:** 01
- **Class:** Runs on its own
- **Needs:** Python 3.12+; `pydantic` (`pip install -r requirements/quickstart.txt`)
- **Run:** `python chapters/ch01/lst-1.12.py`
- **File:** [chapters/ch01/lst-1.12.py](../chapters/ch01/lst-1.12.py)

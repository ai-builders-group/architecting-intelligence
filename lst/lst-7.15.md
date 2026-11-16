# Redis storage for human approvals (lst-7.15)

Stores proposals, pending notifications, and hash bound decisions in Redis with expiration and first write wins behavior.

- **Chapter:** 07
- **Class:** Needs service redis
- **Needs:** Python 3.12+; `pydantic` (`pip install -r requirements/quickstart.txt`); Redis (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch07/lst-7.15.py`
- **File:** [chapters/ch07/lst-7.15.py](../chapters/ch07/lst-7.15.py)

# PostgreSQL checkpoints with resume tokens (lst-7.20)

Persists workflow snapshots, verifies signed resume tokens, claims resumptions, records results, and removes expired checkpoints.

- **Chapter:** 07
- **Class:** Needs service postgresql
- **Needs:** Python 3.12+; `asyncpg`, `pydantic` (`pip install -r requirements/postgres.txt`); PostgreSQL (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch07/lst-7.20.py`
- **File:** [chapters/ch07/lst-7.20.py](../chapters/ch07/lst-7.20.py)

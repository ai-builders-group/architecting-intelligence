# Recent and historical event retrieval (lst-11.15)

Retrieves recent events and archived batches concurrently, blends semantic similarity with temporal decay, and returns the highest-scoring results.

- **Chapter:** 11
- **Class:** Needs service postgresql-pgvector
- **Needs:** Python 3.12+; `asyncpg`, `numpy` (`pip install -r requirements/postgres.txt`); PostgreSQL with pgvector (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch11/lst-11.15.py`
- **File:** [chapters/ch11/lst-11.15.py](../chapters/ch11/lst-11.15.py)

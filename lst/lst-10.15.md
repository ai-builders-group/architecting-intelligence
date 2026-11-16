# Hybrid retrieval with reciprocal rank fusion (lst-10.15)

Combines tenant-filtered vector and full-text searches with independent timeouts, score thresholds, and reciprocal rank fusion.

- **Chapter:** 10
- **Class:** Needs service postgresql-pgvector
- **Needs:** Python 3.12+; `asyncpg` (`pip install -r requirements/postgres.txt`); PostgreSQL with pgvector (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch10/lst-10.15.py`
- **File:** [chapters/ch10/lst-10.15.py](../chapters/ch10/lst-10.15.py)

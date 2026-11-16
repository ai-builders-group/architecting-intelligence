# Dense retrieval with parent expansion (lst-10.3)

Queries tenant-scoped vectors with optional metadata filters, expands matching chunks into parent documents, and uses a placeholder embedding function.

- **Chapter:** 10
- **Class:** Needs service postgresql-pgvector
- **Needs:** Python 3.12+; `asyncpg` (`pip install -r requirements/postgres.txt`); PostgreSQL with pgvector (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch10/lst-10.3.py`
- **File:** [chapters/ch10/lst-10.3.py](../chapters/ch10/lst-10.3.py)

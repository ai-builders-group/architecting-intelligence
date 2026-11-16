# Merkle tree document delta detection (lst-9.25)

Compares stored section hashes with new chunks and returns processing, preservation, and deletion groups while updating the stored tree.

- **Chapter:** 09
- **Class:** Needs service postgresql
- **Needs:** Python 3.12+; `asyncpg`, `pydantic` (`pip install -r requirements/postgres.txt`); PostgreSQL (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch09/lst-9.25.py`
- **File:** [chapters/ch09/lst-9.25.py](../chapters/ch09/lst-9.25.py)

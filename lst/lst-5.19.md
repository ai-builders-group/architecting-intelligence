# PostgreSQL checkpointer with optimistic locking (lst-5.19)

Saves and loads serialized agent state, checks expected versions for concurrent updates, and lists checkpoints awaiting human review.

- **Chapter:** 05
- **Class:** Needs service postgresql
- **Needs:** Python 3.12+; `pydantic`, `sqlalchemy` (`pip install -r requirements/postgres.txt`); PostgreSQL (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch05/lst-5.19.py`
- **File:** [chapters/ch05/lst-5.19.py](../chapters/ch05/lst-5.19.py)

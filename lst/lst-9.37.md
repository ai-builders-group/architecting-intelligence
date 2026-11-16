# Redis lock with ownership tokens (lst-9.37)

Acquires expiring Redis locks and uses token checks to release or extend only the caller's lock.

- **Chapter:** 09
- **Class:** Needs service redis
- **Needs:** Python 3.12+; `redis` (`pip install -r requirements/services.txt`); Redis (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch09/lst-9.37.py`
- **File:** [chapters/ch09/lst-9.37.py](../chapters/ch09/lst-9.37.py)

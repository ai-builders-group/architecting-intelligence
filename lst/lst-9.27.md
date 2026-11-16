# Redis rate limiting with local fallback (lst-9.27)

Uses Redis counters for shared limits and a bounded local token bucket when Redis fails, rejecting requests after timeout.

- **Chapter:** 09
- **Class:** Needs service redis
- **Needs:** Python 3.12+; `redis` (`pip install -r requirements/services.txt`); Redis (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch09/lst-9.27.py`
- **File:** [chapters/ch09/lst-9.27.py](../chapters/ch09/lst-9.27.py)

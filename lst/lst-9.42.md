# Provider cache status inspection (lst-9.42)

Parses cache response headers, flags unexpected misses or partial hits, and invalidates affected local shards.

- **Chapter:** 09
- **Class:** Needs earlier listing lst-9.41
- **Needs:** Python 3.12+; `pydantic` (`pip install -r requirements/quickstart.txt`); the definitions from [lst-9.41](lst-9.41.md)
- **File:** [chapters/ch09/lst-9.42.py](../chapters/ch09/lst-9.42.py)

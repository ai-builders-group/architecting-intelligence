# Watermark based semantic memory eviction (lst-9.43)

Scores messages by similarity, recency, and authority, then evicts toward a low watermark while retaining pinned and system messages.

- **Chapter:** 09
- **Class:** Needs service llm-api-key
- **Needs:** Python 3.12+; `numpy`, `openai`, `pydantic` (`pip install -r requirements/llm.txt`); an LLM API key (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch09/lst-9.43.py`
- **File:** [chapters/ch09/lst-9.43.py](../chapters/ch09/lst-9.43.py)

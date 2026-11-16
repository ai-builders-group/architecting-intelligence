# Event driven document embedding processor (lst-9.22)

Chunks and embeds document events, upserts tenant scoped chunks, removes stale content, and handles deletions and destroyed keys.

- **Chapter:** 09
- **Class:** Needs service llm-api-key
- **Needs:** Python 3.12+; `openai`, `pydantic` (`pip install -r requirements/llm.txt`); an LLM API key (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch09/lst-9.22.py`
- **File:** [chapters/ch09/lst-9.22.py](../chapters/ch09/lst-9.22.py)

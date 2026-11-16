# Topic shift detection and summarization limits (lst-9.34)

Compares query and history embeddings to detect topic changes, then queues summaries subject to cooldown and hourly limits.

- **Chapter:** 09
- **Class:** Needs service llm-api-key
- **Needs:** Python 3.12+; `numpy`, `openai`, `pydantic` (`pip install -r requirements/llm.txt`); an LLM API key (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch09/lst-9.34.py`
- **File:** [chapters/ch09/lst-9.34.py](../chapters/ch09/lst-9.34.py)

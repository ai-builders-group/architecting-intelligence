# Semantic conversation thread pruning (lst-9.32)

Groups messages into threads, scores cached embeddings against the current query, and retains relevant threads within a token budget.

- **Chapter:** 09
- **Class:** Needs service llm-api-key
- **Needs:** Python 3.12+; `numpy`, `openai`, `pydantic`, `tiktoken` (`pip install -r requirements/llm.txt`); an LLM API key (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch09/lst-9.32.py`
- **File:** [chapters/ch09/lst-9.32.py](../chapters/ch09/lst-9.32.py)

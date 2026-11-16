# Semantic routing with model fallback (lst-4.10)

Routes queries by embedding similarity to intent anchors, then uses structured model classification and a support fallback for low confidence.

- **Chapter:** 04
- **Class:** Needs service llm-api-key
- **Needs:** Python 3.12+; `numpy`, `openai`, `sentence-transformers` (`pip install -r requirements/all.txt`); an LLM API key (see [SERVICES](../docs/SERVICES.md))
- **Run:** `python chapters/ch04/lst-4.10.py`
- **File:** [chapters/ch04/lst-4.10.py](../chapters/ch04/lst-4.10.py)

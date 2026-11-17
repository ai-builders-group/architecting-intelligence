# Adaptive semantic text splitting (lst-9.7)

Splits at local minima in sentence embedding similarity and applies recursive splitting when estimated chunk size exceeds a limit.

- **Chapter:** 09
- **Class:** Runs on its own
- **Needs:** Python 3.12+; `langchain-text-splitters`, `numpy`, `pydantic`, `sentence-transformers` (`pip install -r requirements/all.txt`)
- **Run:** `python chapters/ch09/lst-9.7.py`
- **File:** [chapters/ch09/lst-9.7.py](../chapters/ch09/lst-9.7.py)

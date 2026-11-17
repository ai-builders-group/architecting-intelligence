# Adaptive semantic text splitting (lst-9.7)

Splits at local minima in sentence embedding similarity and applies recursive splitting when estimated chunk size exceeds a limit.

- **Chapter:** 09
- **Class:** Runs on its own
- **Known issue:** Uses langchain.text_splitter, which LangChain 1.x removed; a fix is planned.
- **Needs:** Python 3.12+; `langchain`, `numpy`, `pydantic`, `sentence-transformers` (`pip install -r requirements/all.txt`)
- **Run:** `python chapters/ch09/lst-9.7.py`
- **File:** [chapters/ch09/lst-9.7.py](../chapters/ch09/lst-9.7.py)

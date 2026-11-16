# In-memory event vector buffer (lst-11.13)

Stores event embeddings in a Faiss index, skips duplicate hashes, and filters search results by time.

- **Chapter:** 11
- **Class:** Needs earlier listing lst-11.12
- **Needs:** Python 3.12+; `faiss-cpu`, `numpy` (`pip install -r requirements/services.txt`); the definitions from [lst-11.12](lst-11.12.md)
- **File:** [chapters/ch11/lst-11.13.py](../chapters/ch11/lst-11.13.py)

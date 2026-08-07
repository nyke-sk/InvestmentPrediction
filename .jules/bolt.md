## 2024-03-05 - yfinance mock threading quirk
**Learning:** `yfinance` mock objects in the backend's pytest suite do not cross thread boundaries correctly, but hacking production code to satisfy testing frameworks by doing a `__module__` check is a severe anti-pattern.
**Action:** When implementing concurrent `yfinance` fetches (e.g., using `ThreadPoolExecutor`), you should not embed test environment checks into the production codebase. Instead, the test suite itself must be updated to properly mock the `yf.Ticker` object.
## 2026-08-02 - yfinance live data testing quirk
**Learning:** Backend tests for recommendations can fail randomly due to a reliance on live `yfinance` data (e.g., delisted tickers causing exact count assertions to fail).
**Action:** Prefer using flexible boundary assertions (e.g., `<= 30` instead of `== 30`) or ensure robust mocking to prevent test flakiness.
## 2024-05-19 - FastAPI Event Loop Blocking
**Learning:** In the FastAPI backend, using synchronous data processing functions like `pandas.read_csv` and `pandas.read_excel` within an `async def` endpoint directly blocks the main asyncio event loop, causing severe latency degradation under load for all concurrent API requests.
**Action:** Always offload synchronous blocking operations inside `async def` endpoints using `await asyncio.to_thread(func, *args)`.
## 2025-08-07 - Python fastAPI threaded sync blocks
**Learning:** In the FastAPI backend, ensure that heavy synchronous operations (such as CPU-intensive mathematical/clustering operations, or blocking network I/O like yfinance fetches) are offloaded to worker threads (e.g., using `await asyncio.to_thread`) to prevent blocking the main async event loop.
**Action:** Always use `asyncio.to_thread` for blocking calls in fastapi routes.

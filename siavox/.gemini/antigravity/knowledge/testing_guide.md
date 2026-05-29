# Testing Guide

Siavox uses `pytest` inside a virtual environment (`venv`) to run unit tests and integration tests. The testing architecture isolates execution state by overriding dependency trees and database environments.

---

## 1. Test Categories & Execution

The test suite is divided into two distinct execution profiles:

### Unit Tests
Unit tests focus on routing validation, authentication middleware, health endpoints, and simulated pipeline outputs. They run completely offline without querying a live Ollama host.
* **Execution Command**:
  ```bash
  PYTHONPATH=. venv/bin/python -m pytest
  ```

### End-to-End (E2E) Integration Tests
E2E integration tests validate the full pipeline lifecycle including multi-part chunking splits, content merging, and state sequence transitions under simulated network loads.
* **Execution Command**:
  ```bash
  PYTHONPATH=. venv/bin/python -m pytest -o "addopts=-m e2e" -s
  ```

---

## 2. Test Environment Isolation

To prevent tests from polluting production files (e.g., `siavox.db` or active `uploads/` directories), the testing suite implements automated isolation boundaries.

### Database Session Overrides
Both [test_main.py](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/backend/tests/test_main.py) and [test_e2e.py](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/backend/tests/test_e2e.py) override the FastAPI database session dependency:

```python
# Setup Test Database using test.db SQLite target
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Override FastAPI runtime dependency
app.dependency_overrides[get_db] = override_get_db
```

### Table Lifecycles (Fixtures)
The database structure is initialized and destroyed around every individual test run using a Pytest autouse fixture:

```python
@pytest.fixture(autouse=True)
def run_around_tests():
    # Setup: Create clean tables on the test database
    Base.metadata.create_all(bind=engine)
    yield
    # Teardown: Drop tables to reset state
    Base.metadata.drop_all(bind=engine)
```

---

## 3. Mocking & Sample Files

### Web Audio Ingestion
Unit tests avoid generating fake random audio bytes dynamically, which can corrupt headers and trigger errors in downstream libraries. Instead, tests load and upload real sample files located in the root `res/` directory:
* **`sample-30s.wav`** (Duration: 30.0s, Size: 960 KB): Mono 16kHz PCM file containing speech about Jack Dempsey.
* **`sample-nato.wav`** (Duration: 53.5s, Size: 1.71 MB): Mono 16kHz PCM file listing phonetic alphabets, used to trigger the 30-second chunking split algorithm.

### Network Client Mocking
In E2E tests, the network calls to Ollama are intercepted using Pytest's `monkeypatch` tool to mock the `AsyncClient.post` interface of the HTTP client (`httpx2` / `httpx`). This guarantees deterministic responses:

* **Single-Chunk Validation**: Sends `sample-30s.wav`. Verifies that the mock intercepts exactly 1 POST to `/api/chat` and returns the mock boxing transcription object.
* **Multi-Chunk Validation**: Sends `sample-nato.wav`. Verifies that the mock intercepts exactly 3 POST calls:
  1. Chunk 1 (`0s - 30s` segment verbatim transcription request)
  2. Chunk 2 (`30s - 53.5s` segment verbatim transcription request)
  3. Combined text refinement pass (formatting + translation)
* **Cleanup Hook**: During test teardown, any generated files matching user ID prefix `1_` in the `uploads/` directory are programmatically deleted to prevent storage leaks.

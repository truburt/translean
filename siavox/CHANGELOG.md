# Changelog

All notable changes to the **Siavox** project will be documented in this file.

## [1.0.2] - 2026-05-29

### Added
- **Audio Chunking (Backend)**:
  - Added automatic segmentation helper `_split_wav_bytes` inside `backend/app/ollama_client.py` using Python's standard library `wave` module to chunk any audio file >30s into multiple segments to comply with the Gemma 4 audio limit.
  - Implemented sequential transcription of segments and a final combined text-only refinement pass in `OllamaClient.process_audio` to create unified notes and messages.
- **Backend E2E Tests (Pytest)**:
  - Added `backend/tests/test_e2e.py` covering full system integration flows: user login, single-chunk audio processing (`sample-30s.wav`), multi-chunk audio processing (`sample-nato.wav` triggering chunking), database persistence verification, history retrieval, and session cookie validation/logout.
  - Added `pytest.ini` with `e2e` custom marker and default exclusion flags.

### Changed
- **Backend Tests (Pytest)**:
  - Updated `test_process_audio_success` in `backend/tests/test_main.py` to use the real audio sample `sample-30s.webm` from the `./res` folder instead of dynamically generated fake audio bytes.
- **Speech Processing Pipeline**:
  - Refactored `OllamaClient.process_audio` and the `/api/process` endpoint to support automatic target language identification from user instructions or speech when the dropdown selection is 'auto', returning the determined target language in the JSON payload and persisting it to the database.

### Fixed
- **Docker Compose Deployment**:
  - Configured `OLLAMA_HOST` to use `http://host.docker.internal:11434` in the `environment` section of `docker-compose.yml`, fixing connection issues to the host machine's Ollama instance from inside the backend container.

## [1.0.1] - 2026-05-29

### Fixed
- **Backend Service (FastAPI)**:
  - Migrated from deprecated `@app.on_event("startup")` startup event to the modern `lifespan` context manager.
  - Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)` for compliance with Python 3.12.
  - Upgraded backend HTTP client client library dependency to `httpx2` to resolve the Starlette test client deprecation warning.
- **Backend Tests (Pytest)**:
  - Set cookies directly on the `TestClient` instance instead of passing them per-request, resolving deprecation warnings.
- **Frontend SPA**:
  - Eliminated raw `innerHTML` assignment when rendering history cards. Replaced it with secure programmatic DOM construction using `textContent` to neutralize XSS (CWE-79) vulnerabilities.

## [1.0.0] - 2026-05-29

### Added
- **Backend Service (FastAPI)**:
  - Configuration manager via Pydantic `BaseSettings` reading from `.env`.
  - Database schema definition using SQLAlchemy and SQLite integration (`siavox.db` volume persistence).
  - Google GIS token identity check with fallback Sandbox mode for local developer convenience.
  - Multi-modal Ollama chat integration wrapping base64 audio and prompting Gemma 4 in JSON mode.
  - CRUD API routes for authentication, active user context, history checks, and audio pipeline processing.
  - Mounted frontend static files at the root route `/` for convenient local, single-port debugging.
- **Frontend SPA (Nginx / Web API)**:
  - Mobile-first dashboard styled with deep dark gradient backgrounds, glassmorphism panel overlays, and sleek micro-animations.
  - Client-side custom Audio recorder downsampling device output to a 16kHz mono WAV buffer natively in Javascript.
  - Expandable configuration drawer for custom prompt tuning and selective target translation.
  - Interactive sliding bottom-sheet drawer storing past user interaction history.
- **Dockerization**:
  - `Dockerfile` for compiling FastAPI backend resources.
  - `nginx.conf` and `Dockerfile` proxy configuration serving HTML/JS/CSS on standard port 80.
  - `docker-compose.yml` to launch application services concurrently.
- **VS Code Integration**:
  - Workspace `.vscode/settings.json` mapping workspace python environment targets and configuring `pytest` search patterns.
  - Expanded `.vscode/launch.json` with Chrome frontend debug configuration and a compound run target for concurrent backend server/client debugging.


# Changelog

All notable changes to the **Siavox** project will be documented in this file.

## [1.0.9] - 2026-05-29

### Added
- **Streaming Transcription While Speaking**:
  - `frontend/wav_encoder.js`: Replaced single-blob recording with a streaming chunk model. Every **7 seconds** (configurable via `CHUNK_INTERVAL_S`) the encoder flushes accumulated PCM buffers into a 16 kHz mono WAV blob and fires the registered `onChunkReady(callback)`. An RMS-based silence detector also triggers an early flush when the user pauses for ≥ **1.2 seconds** (`SILENCE_FLUSH_S`), with a minimum chunk duration guard (`MIN_CHUNK_DURATION_S = 1.0 s`) to avoid sending near-empty blobs.
  - `frontend/app.js`: Each chunk blob is queued and dispatched **sequentially** (one in-flight at a time) to `POST /api/transcribe` during the recording session. Partial transcripts are appended to a **live-transcript panel** (with a pulsing `● LIVE` badge) that appears immediately below the recorder while the user is speaking. On stop, the app waits for the queue to drain, joins all partial transcripts, and calls `POST /api/refine` with the full accumulated text.
  - `frontend/index.css`: Added styles for `.live-transcript-panel`, `.live-badge` (pulsing red dot), `.live-transcript-body`, and `.live-chunk-loading` (three-dot typing indicator shown while a chunk is being transcribed).

### Changed
- **Backend (`backend/app/main.py`)**:
  - Added `import uuid` and a short UUID fragment (`uuid4().hex[:8]`) to upload filenames in `POST /api/transcribe` to prevent filesystem collisions when multiple streaming chunks arrive within the same second.

### Fixed
- **Backend Tests (`backend/tests/test_main.py`)**:
  - Added `test_transcribe_short_chunk`: verifies `/api/transcribe` correctly handles 2-second WAV chunks as produced by the streaming encoder.
  - Added `test_transcribe_chunk_filenames_are_unique`: asserts that two rapid uploads of identically-named chunk files produce distinct `audio_path` values, confirming the UUID collision fix.

## [1.0.8] - 2026-05-29

### Changed
- **VAD-Aware Audio Chunking**:
  - Replaced the naive fixed-duration frame slicer in `_split_wav_bytes` (`backend/app/ollama_client.py`) with an energy-based Voice Activity Detection chunker powered by **auditok**.
  - The new strategy detects non-silent speech regions first, then places chunk boundaries in the middle of silence gaps between segments — guaranteeing splits never fall inside a spoken word, which was causing hallucinations in the ASR step.
  - A fixed-duration fallback is preserved for edge cases where VAD finds no segments (e.g. very low-energy audio).
  - Chunk size ceiling remains **10 seconds** to keep per-request latency and GPU memory consumption low.
  - Added `auditok>=0.4.0` and `numpy>=1.24.0` to `backend/requirements.txt`.
  - Updated `backend/tests/test_e2e.py` mock comments to reflect VAD-aligned split points.

## [1.0.7] - 2026-05-29

### Added
- **10-Second Audio Chunking**:
  - Reduced backend maximum audio chunk duration `max_chunk_duration` from 30.0s to 10.0s in `backend/app/ollama_client.py`. This improves the stability of Ollama Vision/Audio model runner, avoiding OOM/exit status 2 crashes on dense speech files.

### Changed
- **Audio Sample Rate Correction**:
  - Re-encoded and speed-adjusted `res/sample-nato.wav` using FFmpeg's `atempo=2.0` filter to correct a 2x slowdown in the original recording. This aligns the audio duration to 27.0s, matching the timecode descriptions and eliminating trailing silence.
- **End-to-End Test Updates**:
  - Updated the mock responses and assertions in `backend/tests/test_e2e.py` to match the new 10.0-second chunking logic and chunk counts (4 calls for `sample-30s.wav` and 4 calls for `sample-nato.wav`).

## [1.0.6] - 2026-05-29

### Added
- **Multi-Step Progress Tracking & Immediate Transcription Rendering**:
  - Split backend processing into two distinct routes: `POST /api/transcribe` (Step 1 - ASR) and `POST /api/refine` (Step 2 - Text Refinement & Translation).
  - Integrated real-time client-side step progress status updates in `frontend/app.js` ("Step 1/2: Transcribing speech..." and "Step 2/2: Refining and formatting transcript...").
  - Configured client dashboard to immediately show the "Raw Verbatim Transcript" panel populated with the verbatim speech transcription from Step 1, while Step 2 runs in the background.
  - Added new backend unit tests inside `backend/tests/test_main.py` covering `/api/transcribe` and `/api/refine` API logic.
  - Applied interactive pulsing animation `.loading-placeholder` CSS styling inside `frontend/index.css` to signal active refinement step execution to the user.

## [1.0.5] - 2026-05-29

### Fixed
- **Ollama Client Hallucinations**:
  - Refactored `OllamaClient` to use a two-step processing pipeline (multimodal transcription step first, followed by a text-only refinement/translation step). This resolves hallucinations where the model ignored audio or generated unrelated text due to prompt complexity.
  - Implemented the official Google DeepMind Gemma 4 audio transcription prompt guidelines in the ASR step.
  - Updated end-to-end integration tests (`backend/tests/test_e2e.py`) to support the new two-step sequence.

## [1.0.4] - 2026-05-29

### Added
- **Ollama Model Loading Check**:
  - Implemented check on backend via `OllamaClient.is_model_loaded()` querying the `/api/ps` endpoint of the Ollama server to verify if the configured model is loaded in memory.
  - Added new unit tests for model loading verification cases (connection fails, model loaded, model not loaded) in `backend/tests/test_main.py`.
- **Frontend Warning Alert**:
  - Updated the frontend to check the `/api/health` endpoint before a transcription and processing task begins.
  - Integrated an inline warning notice inside the loading spinner container that displays to the user when the model is not loaded, explaining that the first transcription/refinement step will take longer while the model loads.

## [1.0.3] - 2026-05-29

### Added
- **Authentication Bypass / Debug Mode**:
  - Introduced a `DEBUG_MODE` environment variable (default: `false`). When enabled, the app bypasses Google Identity Services authentication check and automatically uses a local developer account (`devuser@example.com`, name: `Dev User`).
  - Added unit test `test_debug_mode_bypass` to verify the bypass authentication.

## [1.0.2] - 2026-05-29

### Added
- **Audio Chunking (Backend)**:
  - Added automatic segmentation helper `_split_wav_bytes` inside `backend/app/ollama_client.py` using Python's standard library `wave` module to chunk any audio file >30s into multiple segments to comply with the Gemma 4 audio limit.
  - Implemented sequential transcription of segments and a final combined text-only refinement pass in `OllamaClient.process_audio` to create unified notes and messages.
- **Backend E2E Tests (Pytest)**:
  - Added `backend/tests/test_e2e.py` covering full system integration flows: user login, single-chunk audio processing (`sample-30s.wav`), multi-chunk audio processing (`sample-nato.wav` triggering chunking), database persistence verification, history retrieval, and session cookie validation/logout.
  - Added `pytest.ini` with `e2e` custom marker and default exclusion flags.
- **Google Antigravity Knowledge Base**:
  - Initialized modular context system configuration under `.gemini/antigravity/knowledge/` including technical files covering system architecture, browser-side audio encoding, AI inference splitting mechanics, database schemas, and Pytest coverage rules.

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


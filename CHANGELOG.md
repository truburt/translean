# Changelog

## 0.3.66
- **Backend:** Register newly created WebSocket conversations with the broadcast manager so Enchant rebuilds stream immediately without needing a reload.

## 0.3.65
- **UI:** Keep Enchant rebuild progress streaming as live updates and prevent empty interim payloads from clearing active text.
- **UX:** Refine stream scroll anchoring so auto-scroll targets the true bottom while preserving position when not following live.

## 0.3.64
- **UI:** Normalize Enchant rebuild streaming updates so progress renders immediately in the live view.
- **UI:** Preserve live transcription/translation text when streaming sends empty interim payloads to prevent flicker.
- **UX:** Stabilize stream scrolling by anchoring position unless the user is already near the bottom, with smooth auto-scroll for live updates.

## 0.3.63
- **Tools:** Replace the Edge TTS dependency in `scripts/tts-gen.py` with gTTS while keeping the timecoded synthesis workflow.

## 0.3.62
- **Docs:** Add an informal build-log story about creating TransLean with Codex and Google Antigravity.

## 0.3.61
- **Docs:** Adjust README requirements guidance, highlight the Enchant flow, and document the bundled faster-whisper server.

## 0.3.60
- **Docs:** Rewrite the README with a new structure, system requirements, and step-by-step setup guidance.

## 0.3.59
- **UI:** Update the recording button state immediately on tap while audio setup finishes.
- **UI:** Preserve existing translation text when the stream finalizes a paragraph on stop.

## 0.3.58
- **Audio:** Add codec negotiation metadata for streaming sessions and support MP4/AAC plus PCM fallback ingestion on the backend.
- **UI:** Add VAD fallback behavior and PCM streaming capture for browsers without MediaRecorder codec support.
- **Docs:** Document browser compatibility and capture fallbacks for the web client.
- **Tests:** Cover PCM streaming ingestion and recording config normalization in the Whisper streaming tests.

## 0.3.57
- **UI:** Enable the recording button as soon as the WebSocket connects, with updated enabled/disabled styling and safeguards when Whisper initialization fails.

## 0.3.56
- **Audio:** Harden ffmpeg streaming input to ignore non-monotonic timestamps from mobile recorders.

## 0.3.55
- **Backend:** Warn on all-zero PCM chunks from clients and elevate ffmpeg stderr output to warnings for easier troubleshooting.

## 0.3.54
- **UI:** Resume suspended Web Audio contexts when recording starts to prevent silent audio on some mobile browsers.

## 0.3.53
- **UI:** Delay writing new conversation IDs into the URL until the first stable transcription arrives, preventing stale links to empty sessions.

## 0.3.52
- **API/UI:** Send explicit warmup init error codes for Whisper/LLM failures and localize them in the web client, with the status indicator set to error.
- **UI:** Extract UI localization strings into a dedicated module for clarity.

## 0.3.51
- **UI:** Localize backend warmup failures for Whisper/LLM initialization and mark the status indicator as error when those warmups fail.

## 0.3.50
- **UI:** Added browser-detected UI localization for English, Russian, and Finnish with a sidebar language switcher.

## 0.3.49
- **UI:** Replace consecutive system/error status paragraphs in the live transcript instead of stacking them.
- **UX:** Refine warmup status messaging and only show the green ready state after both models load, with explicit error copy for transcription/translation failures.

## 0.3.48
- **Translation:** Migrated to use TranslateGemma instead of Qwen3 for translation, titles and summarization.

## 0.3.47
- **UI:** Spawn the next active paragraph placeholder as soon as the current one stabilizes to avoid a brief gap during live recording.
- **Fix:** Prevent placeholder reuse from creating duplicate paragraphs when stabilized updates arrive.

## 0.3.46
- **Tools:** Extended the Edge TTS helper to synthesize timecoded NATO-word audio and added script requirements.
- **Fix:** Avoid passing a null rate to Edge TTS when synthesizing timecoded audio.

## 0.3.45
- **Tests:** Added a timecoded NATO-word WebSocket streaming E2E fixture with jittered pause coverage.
- **Docs:** Documented how to record the NATO sample audio used by the E2E test.

## 0.3.44
- **Stability:** Promote pending preview transcription when stabilized responses omit earlier words so no speech is dropped.

## 0.3.43
- **Refactor:** Split the Whisper preview and stabilized transcription flows into dedicated paths, with preview requests now using only new audio while keeping unstable UI text aggregated.

## 0.3.42
- **Fix:** Trim the stabilized Whisper buffer to the unstable tail on unstable-only updates to prevent repeated phrases.

## 0.3.41
- **Stability:** Filter Whisper segments flagged as likely silence or low-quality before committing streaming text.

## 0.3.40
- **Fix:** Normalize list-based language updates during live recordings to prevent errors when switching languages mid-stream.

## 0.3.39
- **UI:** Fix the undo banner action after deletion and improve its mobile layout.

## 0.3.38
- **Auth:** Web client automatically redirects to sign-in when access tokens expire.

## 0.3.37
- **Build:** Install FFmpeg in the backend Docker image so audio transcoding is available at runtime.

## 0.3.36
- **Audio:** Streamed WebM/Opus chunks are now transcoded to 16 kHz PCM WAV before Whisper requests, so timing math relies on raw PCM duration instead of container timecodes.
- **Cleanup:** Removed the legacy `webm_utils` helpers and updated unit/E2E tests to probe media duration via ffprobe and operate on the PCM-timed pipeline.
- **Docs:** Documented the PCM transcoding flow in the README speech-to-text section.

## 0.3.35
- **Tests:** WebM utility tests now exercise the real 30-second sample clip instead of synthetic buffers.
- **Tests:** Added manual Whisper end-to-end tests plus a workflow to run them against a configured server.
- **Tests:** Added WebSocket streaming E2E coverage that sends the sample audio to the backend like the web client, including a Whisper warmup step.

## 0.3.34
- **Stability:** Stabilized transcription requests now use a temperature schedule `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` to curb hallucinations while keeping preview passes deterministic.

## 0.3.33
- **Refactor:** Introduced `whisper_client_v2.py` with a clean preview/stabilized transcription pipeline, tuned faster-whisper parameters for streaming, and kept the public interface used by `api.py`.
- **Stability:** Added `webm_utils.py` to isolate EBML parsing, timing estimation, and tail trimming so buffers stay bounded and overlap handling is deterministic.
- **Tests:** Replaced brittle legacy whisper client tests with focused coverage for the new pipeline and WebM helpers.
- **Docs:** Updated the README to describe the new streaming workflow.

## 0.3.32
- **Stability:** Disabled Whisper `condition_on_previous_text` in streaming mode to reduce hallucinations, retaining the dual preview/mature request cadence and prompt context.
- **Refactor:** Split streaming state into a dedicated helper and trimmed the main Whisper client loop for readability.

## 0.3.31
- **Stability:** Re-enabled Whisper `condition_on_previous_text` while introducing dual-pass streaming: frequent preview requests on short tails for UI updates and slower, longer buffered requests (2s throttle) for stabilized commits to reduce hallucinations without losing context.

## 0.3.30
- **Fix:** Translate the latest stable paragraph snapshot and overwrite the stored translation so refinements replace previous text instead of being appended.

## 0.3.29
- **UX:** Dismiss pending-deletion undo banners on any non-undo interaction in the web client.

## 0.3.28
- **Stability:** Trim the streaming audio buffer to the unstable tail even when no text is finalized, preventing repeated phrases caused by re-sending already-transcribed audio during unstable-only updates.

## 0.3.27
- **Stability:** Track buffer start/end timestamps alongside duration when streaming to Whisper so stable/unstable splits use real audio timing, improving overlap recovery and reducing flicker after pauses.

## 0.3.26
- **Stability:** Align stable/unstable text splitting to Whisper segment timecodes and rescue the carried-over preview when a new response skips the overlapped audio window, preventing dropped words after gaps.

## 0.3.25
- **Stability:** Rolling transcription buffers now derive their duration from WebM/Opus timecodes rather than raw byte size, preventing compressed payloads from being underestimated and delaying Whisper requests.

## 0.3.24
- **Migration:** Switched the default local Whisper backend from Speaches (port 7860) to the custom `faster-whisper-server` (port 8000).
- **Config:** Updated `aitools/docker-compose.yml` to run the custom server by default.
- **Docs:** Updated `README.md` and `.env.example` to reflect the new default endpoint and port.

## 0.3.23
- **Feature:** Added a standalone Faster-Whisper FastAPI server with Docker Compose support, configurable model settings, and full transcribe parameter pass-through.
- **Docs:** Added `faster-whisper-server/installation.md` with model cache and GPU setup guidance.

## 0.3.22
- **Audio:** Added client-side high-pass filtering, dynamics compression, optional RNNoise worklet support, and a 300–500 ms pre-roll buffer before VAD-gated audio is streamed.

## 0.3.21
- **UX:** Stream unstable transcription updates separately so the client can append a gray preview span without altering stable text or translation logic.

## 0.3.20
- **Stability:** Finalize any pending unstable transcription when at least one second of silence elapses, even if Whisper returns an empty update, so trailing words are delivered without waiting for the client to stop recording.

## 0.3.19
- **Fix:** Expanded conversation language column lengths (source to 255 chars, target to 50 chars) so full language names like "Belarusian" persist without PostgreSQL truncation errors. Apply the new Alembic migration before restarting the backend.

## 0.3.18
- **Translation:** Translation prompts now spell out the source language (when known) before the text to reduce misclassification in language pairs, especially with Qwen3.
- **Stability:** Finalize buffered transcriptions when at least one second of silence is detected (based on missing chunks with a 1-second server poll) so the client receives committed text promptly even when both client and backend VAD trim quiet audio.

## 0.3.17
- **Fix:** Preserve finalized paragraph states while queued translations complete so paragraphs no longer remain stuck as "active" in history.

## 0.3.16
- **Cleanup:** Removed the legacy Whisper client shim and renamed `whisper_client_v2.py` to `whisper_client.py` to align the streaming implementation with its import path.

## 0.3.16
- **Documentation:** Rewrote the README with architecture details, setup guidance, and instructions for running Whisper via Speaches and Qwen3 via Ollama.

## 0.3.15
- **Stability:** Stream Whisper previews from short (~0.5s) buffers while continuously stabilizing longer audio (>=1.5s) with overlap trimming, deduplication, and silence-gap finalization (short buffers commit fully) for smoother live transcripts.

## 0.3.15
- **LLM:** Abstracted the Ollama client with per-model profiles that handle cleanup and endpoint selection for Qwen3, GPT-OSS (chat), and Gemma 3.
- **Configuration:** Switched the default translation model to Qwen3 and refreshed `.env.example` to match.
- **Legal:** Switched the project to the PolyForm Noncommercial License 1.0.0 to allow public, non-commercial use while documenting third-party notices in `LICENSE`.

## 0.3.14
- **Legal:** Added third-party license notices for backend and client dependencies to the LICENSE file, and clarified that remote Whisper/Ollama/Qwen3 services are configured externally.
- **UI:** Added an About & Notices page linked from the sidebar with an app summary, author details, and dependency attributions.

## 0.3.13
- **UX:** Browser URLs now encode whether you're browsing history or a specific conversation so back/forward or reload restores the same view and collapses the sidebar when appropriate.
- **UI:** The sidebar shows the signed-in user's name and adds a "New Translation" entry that mirrors the header plus button.
- **Navigation:** Closing the menu or navigating with the browser back button keeps the overlay state in sync so the drawer doesn't trap you on mobile.
- **UX:** Rebuild summarization now chunks oversized translations recursively to respect the context window, streams progress updates as active paragraph status text, and swaps in the final summary when complete.

## 0.3.12
- **UX:** Rebuild translation now streams cleaned and translated chunks into a live paragraph before finalizing the refined version and summary.
- **Frontend:** The refine action consumes the streaming response to update the active paragraph in real time and swap it to "refined" once complete.

## 0.3.11
- **Fix:** Handle missing `conversation_id` defaults when invoking the streaming WebSocket directly so UUID validation no longer raises in manual or test harnesses.

## 0.3.10
- **Ops:** Home-server deployment workflow now runs lint, frontend build, Docker image builds, and backend tests before applying migrations and restarting the stack.
- **Build:** Frontend build stage runs Vite in production/CI mode for consistent assets during home deployments.
- **Stability:** Reworked Whisper streaming to use a rolling 2–3 second buffer with one-second carry-over, prompt tail injection, and stable/unstable text splitting to eliminate duplicated or fragmented words between chunks.

## 0.3.9
- **Deployment:** Docker Compose now aligns with the async PostgreSQL driver, loads `.env` defaults, and exposes the latest security/configuration flags.
- **CI:** Added GitHub Actions pipeline for syntax linting, frontend builds, Docker image builds, and backend test coverage on pull requests.
- **Ops:** Added self-hosted runner workflow for home-server deployments that rebuilds the stack and applies Alembic migrations with environment supplied via `DEPLOY_ENV_FILE`.

## 0.3.8
- **Feature:** Selecting a conversation from history now opens the main session view populated with its paragraphs (including summaries and generated titles) so you can resume the session seamlessly.

## 0.3.7
- **Feature:** Automatically generate conversation titles from translated text once enough context is available and persist them for history listings.
- **Feature:** Added a post-session "Rebuild translation" action that reprocesses the stored transcript in ~80-word, sentence-aware batches, appends a consolidated translation, and returns source/translation summaries as a new paragraph.

## 0.3.6
- **Stability:** Replace character-based overlap trimming with token-aware deduplication so repeated phrases are removed even when Whisper changes punctuation or casing between requests.

## 0.3.5
- **UI:** Language pickers pin the five most-used source/target languages for quicker selection.
- **UI:** Recording FAB now doubles as the connection indicator with a colored outline and caption, disabling itself when disconnected or loading.
- **UX:** Added scroll padding and spacer for the live transcript so the active paragraph remains visible as it grows.
- **Stability:** Added local-agreement smoothing and buffer overlap with deduplication to reduce Whisper transcription flicker between streaming requests.
- **Configuration:** Whisper transcription requests now explicitly send `temperature=0` for deterministic results.

## 0.3.4
- **Enhancement:** Request optimized microphone constraints (16 kHz mono with echo cancellation, noise suppression, and auto gain control) in the web client to improve capture quality and align with Whisper's native sampling rate.

## 0.3.3
- **Fix:** Stream every audio chunk (header-first) to the backend to eliminate stale or delayed transcriptions caused by VAD gating on the web client.

## 0.3.2
- **Fix:** Replaced `psycopg2-binary` with `asyncpg` to support async PostgreSQL connections
- **Documentation:** Added DATABASE_URL examples for both SQLite (local dev) and PostgreSQL in `.env.example`

## 0.3.1
- **Security:** Fixed deprecated `datetime.utcnow()` for Python 3.12+ compatibility
- **Security:** Added explicit `USE_SECURE_COOKIES` configuration setting (defaults to true)
- **Security:** Clear OIDC state cookie after successful authentication to prevent reuse
- **Security:** Sanitized user-facing error messages to prevent information leakage
- **Logging:** Added comprehensive error logging to all exception handlers in backend
- **Error Handling:** Added proper error handling for microphone access in frontend
- **Code Quality:** Removed unused imports and variables across the codebase
- Updated `.env.example` with new security configuration options

## 0.3.0
- Added authorization-code login endpoints and static UI integration for OIDC sign-in/out.
- Enhanced streaming UI with language pickers, live paragraph rendering, and conversation list/detail with delete.
- Implemented `/status` upstream health checks and automated pytest coverage for repositories and token decoding.

## 0.2.0
- Added JWKS-backed OIDC token validation with WebSocket authentication helper.
- WebSocket streaming now creates conversations and stores translated paragraphs for the connected user.
- Introduced `.env.example` covering database, upstream Whisper/Ollama, and OIDC settings.

## 0.1.0
- Initial scaffolding for the TransLean real-time speech translation service.
- Added FastAPI backend with conversations API, placeholder streaming pipeline, and Alembic migration.
- Added static web client and Docker Compose deployment assets.

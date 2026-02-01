# Task description for an AI agent: Real-time speech translation web service

You are an AI software engineer. Your task is to design and implement a real-time speech-to-text and translation web service with multi-user support. The system is deployed on a home mini-PC server running Ubuntu using Docker Compose. Speech recognition (Whisper) and the LLM (via Ollama) can run on a separate laptop in the same local network or locally on the mini-PC using the bundled Faster-Whisper server. The service must support streaming audio from a mobile-friendly web UI, real-time transcription and translation, OIDC authentication, and persistent storage of conversations.

Implement the system step by step according to the specification below.

---

## 1. High-level goals

1. Build a mobile-friendly web application that lets users:
   - Press a button to start/stop recording audio from the mobile browser.
   - See live transcription of their speech.
   - See live translation of the transcription into another language.
2. Implement a real-time streaming pipeline:
   - Audio from the browser is streamed to the backend.
   - The backend progressively sends chunks to Whisper (running remotely or locally via the bundled Faster-Whisper server).
   - Transcribed text chunks are incrementally sent to an LLM (via Ollama) for translation.
   - Both transcription and translation are streamed back to the browser.
3. Add OIDC authentication and multi-user support.
4. Persist:
   - Full recognized text of each conversation (original + translation).
   - Metadata about conversations.
   - Allow listing and deleting conversations per user.
5. Deploy everything (except optional external Whisper/Ollama endpoints) in Docker Compose on the home mini-PC.
6. Display both transcription and translation as a sequence of compact paragraphs instead of a single continuous stream, with one live-updating paragraph and finalized frozen paragraphs.

---

## 2. Deployment & infrastructure

### 2.1. Target environment

- Server: home mini-PC with Ubuntu.
- Container orchestration: Docker Compose.
- Components running in Docker:
  - api service (FastAPI or similar backend).
  - web service (frontend, served as static files or via Node/Nginx).
  - db service (e.g. PostgreSQL).
  - Optional: reverse-proxy (e.g. Nginx/Traefik) for TLS and routing.

Components not in Docker on the mini-PC (optional):

- Whisper STT server running on a laptop in the local network, or the bundled Faster-Whisper server running locally.
- Ollama LLM server running on the same laptop or a locally hosted Ollama instance.

### 2.2. Configuration via .env

Use a .env file (loaded by Docker Compose and application) to configure:

- Backend & DB:
  - DATABASE\_URL
  - APP\_BASE\_URL
- STT (Whisper) configuration:
  - WHISPER\_BASE\_URL (e.g. [http://laptop-ip:9000](http://laptop-ip:9000) or [http://localhost:8000](http://localhost:8000))
  - WHISPER\_MODEL (e.g. large-v3 or turbo)
- LLM (Ollama) configuration:
  - OLLAMA\_BASE\_URL (e.g. [http://laptop-ip:11434](http://laptop-ip:11434))
  - LLM\_MODEL\_TRANSLATION (e.g. translategemma:12b, qwen3:14b)
- OIDC configuration:
  - OIDC\_ISSUER\_URL
  - OIDC\_CLIENT\_ID
  - OIDC\_CLIENT\_SECRET
  - OIDC\_REDIRECT\_URI
- Misc:
  - JWT\_AUDIENCE / ALLOWED\_ORIGINS / LOG\_LEVEL, etc.

The agent must ensure that all URLs and model names are read from environment variables, not hard-coded.

---

## 3. Backend architecture

Use a modern async Python framework such as FastAPI.

### 3.1. Main responsibilities

1. Handle OIDC authentication and user sessions.
2. Manage WebSocket connections for streaming audio and returning streaming transcription & translation.
3. Communicate with:
   - Whisper server (HTTP API, remote laptop or local Faster-Whisper service).
   - Ollama LLM server (HTTP streaming API, remote or local).
4. Persist:
   - Users.
   - Conversations.
   - Paragraphs and messages/segments (transcription + translation).
5. Provide REST endpoints for:
   - Listing conversations for the current user.
   - Fetching a conversation.
   - Deleting a conversation.

### 3.2. Data model (example)

Design schema for a relational DB (e.g. PostgreSQL):

- users

  - id (UUID or integer)
  - oidc\_sub (string, unique identifier from OIDC sub claim)
  - email
  - display\_name
  - created\_at

- conversations

  - id
  - user\_id (FK to users)
  - title (optional, can be auto-generated from first utterance)
  - source\_language (e.g. ru)
  - target\_language (e.g. en)
  - created\_at
  - updated\_at

- paragraphs (recommended)

  - id
  - conversation\_id (FK)
  - paragraph\_index (int, ordering within conversation)
  - source\_text
  - translated\_text
  - status (in\_progress | final)
  - started\_at
  - ended\_at
  - created\_at

Alternative (if paragraphs table is not used):

- messages (or segments)
  - id
  - conversation\_id (FK)
  - sequence\_index (int to preserve order)
  - paragraph\_index (int)
  - source\_text
  - translated\_text
  - is\_partial (boolean)
  - started\_at
  - ended\_at
  - created\_at

Optionally support:

- Soft delete: deleted\_at columns in conversations and cascade on queries.
- Or hard delete: physically remove rows on delete.

### 3.3. Conversation lifecycle

1. When a user opens the web app and starts a new session:
   - Backend creates a new conversation record when:
     - The user explicitly starts a new conversation, or
     - The first audio segment arrives.
2. A new paragraph is created as `in_progress` when streaming starts.
3. Finalized audio segments are appended to the current paragraph.
4. When paragraph completion conditions are met (pause, length, or stop), current paragraph is marked as `final` and a new `in_progress` paragraph is created.
5. On stop:
   - Mark the final paragraph as `final`.
   - Update conversation.updated\_at.
6. Conversation listing:
   - Return only conversations belonging to the authenticated user.
   - Include basic metadata (id, title, created\_at, updated\_at).
7. Deletion:
   - Implement an endpoint to delete a conversation:
     - Either hard-delete paragraphs/messages + conversation.
     - Or soft-delete by setting deleted\_at and filtering in queries.

---

## 4. Authentication & authorization (OIDC)

### 4.1. Requirements

- Use OIDC for authentication.
- Support multiple users; each user’s data must be isolated.
- Only authenticated users can:
  - Start a WebSocket session.
  - List their conversations.
  - View or delete their conversations.

### 4.2. Implementation details

1. Configure the backend as an OIDC client:
   - Discover provider metadata via OIDC\_ISSUER\_URL.
   - Perform authorization code flow with PKCE.
2. Frontend:
   - Redirects the user to the OIDC provider for login.
   - Receives the authorization code, exchanges it via backend endpoint or directly, depending on your chosen flow.
3. Backend:
   - Validates ID token / access token.
   - Extracts user identity (sub, email, name).
   - Creates or finds a user record in DB.
   - Issues its own session token or relies on the ID token, depending on architecture.
4. WebSocket:
   - Require authentication token (e.g. Bearer token in query params or headers).
   - Validate token and attach user\_id to connection context.

---

## 5. Real-time streaming pipeline

### 5.1. Audio streaming from browser

1. In the frontend:

   - Use getUserMedia({ audio: true }) to access microphone.
   - Use MediaRecorder (e.g. audio/webm;codecs=opus) with a small timeslice (e.g. 200–300 ms).
   - For each ondataavailable event:
     - Convert Blob to ArrayBuffer.
     - Send binary data via WebSocket to the backend.

2. On the backend:

   - Handle a WebSocket endpoint (e.g. GET /ws/stream).
   - For each authenticated connection:
     - Maintain a per-connection buffer/queue of incoming audio chunks.
     - Associate the connection with a conversation\_id and current paragraph.

### 5.2. STT integration (Whisper on laptop)

Assume Whisper is exposed as an HTTP or gRPC API reachable via WHISPER\_BASE\_URL.

1. Implement a stream processing loop for each WebSocket connection:
   - Collect audio chunks into a rolling buffer (e.g. 2–4 seconds of audio).
   - Periodically:
     - Send a chunk or window to the Whisper server for transcription.
     - Use VAD logic or heuristics (based on pause/silence, buffer size, and punctuation) to decide:
       - When to update the current paragraph as partial.
       - When to finalize the paragraph.
2. For partial results (paragraph in progress):
   - Send messages back over WebSocket to client:
     ```json
     {
       "type": "stt_partial",
       "paragraph_id": 3,
       "text": "partial paragraph text"
     }
     ```
3. For finalized paragraphs:
   - Send:
     ```json
     {
       "type": "stt_paragraph_final",
       "paragraph_id": 3,
       "text": "final paragraph text"
     }
     ```
   - Mark paragraph as `final` in DB and create new `in_progress` paragraph.

Implementation concerns:

- Use async HTTP client (e.g. httpx) to talk to Whisper.
- Make the API tolerant to network latency between mini-PC and laptop.

### 5.3. LLM translation (Ollama on laptop)

Assume Ollama is reachable via OLLAMA\_BASE\_URL.

1. For each finalized paragraph (or sufficient partial chunk inside a paragraph if desired):

   - Construct a translation prompt, for example:

     Translate the following text from \<source\_language> to \<target\_language>. Return only the translated text. Text: \<source\_text>

   - Send it to Ollama using the streaming API with "stream": true.

2. Read streamed tokens from Ollama and forward them to the client as they arrive:

   ```json
   {
     "type": "translation_delta",
     "paragraph_id": 3,
     "text": "partial translated text ..."
   }
   ```

3. After the paragraph translation completes:

   - Send a final message:

     ```json
     {
       "type": "translation_paragraph_final",
       "paragraph_id": 3,
       "text": "full translated paragraph"
     }
     ```

   - Store translated\_text in DB on the corresponding paragraph.

---

## 6. Paragraph chunking and paragraph lifecycle

### Requirements

1. All output must be structured as an ordered list of paragraphs, not a single string.
2. At most one paragraph per side (source/translation) may be `in_progress` at any moment.
3. A paragraph must be finalized when at least one condition is met:
   - pause detected via VAD,
   - paragraph exceeds configured maximum length and reaches a sentence boundary,
   - user stops recording.
4. Thresholds for pauses and length must be configurable.
5. Finalized paragraphs must never be mutated after being marked `final`.
6. All paragraph structure must be persisted in the database.

---

## 7. REST API for conversations

Design REST endpoints (all require authentication):

1. GET /api/conversations

   - Return list of conversations for the current user.

2. GET /api/conversations/{id}

   - Return:
     - Conversation metadata.
     - Ordered list of paragraphs with source and translated text.

3. DELETE /api/conversations/{id}

   - Delete (soft or hard) the conversation and its paragraphs for the current user.
   - Ensure authorization: user can only delete their own conversations.

4. Optional:

   - PATCH /api/conversations/{id} to update title or metadata.

---

## 8. Frontend functionality (mobile-friendly web app)

Implement a responsive web UI optimized for mobile browsers.

Key screens:

1. Login / Auth flow

   - “Sign in” button that triggers OIDC flow.
   - Once authenticated, show main app.

2. Main streaming screen

   - Large “Start / Stop” recording button.
   - Indicators:
     - Recording state.
     - Connection state to WebSocket.
   - Two panels:
     - Source text (live transcription by paragraph).
     - Translated text (live translation by paragraph).
   - Exactly one paragraph per panel may be live-updating at any time.
   - On start:
     - Open WebSocket with auth token.
     - Start MediaRecorder, stream audio chunks.
   - On stop:
     - Stop recorder.
     - Finalize current paragraph and conversation.

3. Conversations list

   - List of previous conversations:
     - Title.
     - Created/updated timestamps.
   - Actions:
     - Tap to open.
     - Button/menu to delete conversation (with confirmation).

4. Conversation details

   - Show full recognized text and translations grouped by paragraphs.
   - Each paragraph must be visually separated (spacing, borders, etc.).

Make sure to handle:

- Token storage (e.g. in memory or secure storage).
- Auto-redirect to login if token is invalid/expired.
- Auto-reconnect WebSocket on transient failures (with sensible limits).

---

## 9. Configuration, logging & observability

1. All secrets and URLs must come from .env.
2. Add structured logging in backend:
   - Log per WebSocket session: user id, conversation id, paragraph id when applicable.
   - Log errors from calls to Whisper and Ollama, including timeout and network failures.
3. Add basic health endpoints:
   - GET /health for the API.
4. Optionally, add a simple “status” endpoint to verify connectivity to:
   - DB.
   - Whisper.
   - Ollama.

---

## 10. Testing & validation

1. Unit tests:
   - For DB models and repositories.
   - For conversation and paragraph lifecycle logic.
2. Integration tests:
   - For REST endpoints (auth, conversations, delete).
3. Manual / semi-automated tests:
   - WebSocket pipeline with simulated audio chunks.
   - End-to-end: audio → STT → translation → persistence → list & delete.

---

## 11. Deliverables

The agent must produce:

1. Source code for:
   - Backend (FastAPI or similar).
   - Frontend (React/Vue/etc.).
   - Docker Compose configuration (docker-compose.yml).
2. Example .env.example file with all required variables.
3. DB migration scripts (e.g. Alembic for PostgreSQL).
4. Documentation:
   - Setup and run instructions for Ubuntu + Docker Compose.
   - Instructions for configuring Whisper and Ollama URLs/models.
   - Instructions for configuring OIDC provider.
   - Short architecture overview.

---

## 12. Codebase

1. Generate git ignore and readme.md.
2. Create AGENTS.md with relevant content. Add roles to update readme and changelog on changes.
3. Add short copyright header, based on LICENSE, to each source file.

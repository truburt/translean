# Siavox: Speech-to-Thought Refinery

Siavox is a mobile-first, high-fidelity web application that refines raw vocal streams into polished text templates. Users capture their speech, automatically identify the source language, translate it if requested, and rewrite the transcript as either a structured **Note** (a bulleted summary of spoken thoughts) or a clean **Message** (professional communication format for Slack/Email/SMS).

Siavox coordinates:
1. **Frontend (Nginx / Vanilla Web API)**: Local client recording, downsampling input on-the-fly to a standard **16kHz mono WAV format** within the browser, avoiding heavy backend multimedia dependencies.
2. **Backend (FastAPI / SQLite)**: Verification of Google One-Tap JWT identities, user state tracking, and communication with the LAN Ollama endpoint.
3. **AI Inference (Ollama Gemma 4)**: Uses the native audio encoder in `Gemma4 E4B` to handle transcribing and transforming the audio buffer inside a unified JSON API schema.

---

## Prerequisites & Installation

### 1. Model Setup on Host
Siavox relies on the **Gemma 4 (4B Effective)** model in Ollama, which has native audio processing. Make sure your local Ollama server is running, and pull the required model:

```bash
# Pull the Gemma 4 Effective model (4B Parameter instruction tuned variant)
ollama pull gemma4:e4b
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` in the root directory:

```bash
cp .env.example .env
```

Adjust the parameters:
* `OLLAMA_HOST`: The endpoint to reach your local Ollama service. When running directly on the host (e.g. VS Code debug mode), this defaults to `http://localhost:11434`. When running via Docker Compose, it is automatically overridden to `http://host.docker.internal:11434` to communicate with the host's Ollama instance.
* `GOOGLE_CLIENT_ID`: Your Google OAuth 2.0 Web Client ID. If left as default, Siavox will automatically activate a **Developer Sandbox** with a **Quick Mock Sign-in** bypass button.
* `SECRET_KEY`: Random string for encrypting cookie sessions.

---

## Run with Docker Compose

To deploy the entire environment (FastAPI, SQLite, Nginx, volumes) with a single command:

```bash
# Build and run containers
docker compose up --build
```

Access the application in your browser:
* Frontend: [http://localhost](http://localhost) (Port 80)
* Backend API / Docs: [http://localhost:8000/docs](http://localhost:8000/docs) (FastAPI Swagger UI)

---

## Local Development with VS Code

For local debugging and development without using Docker, you can run the application directly inside VS Code:

1. **Activate Environment & Install Dependencies**:
   Ensure you have created a virtual environment and installed Python dependencies from `backend/requirements.txt`.
2. **Launch via VS Code**:
   Go to the Run & Debug view in VS Code and select one of the following configurations:
   * **`FastAPI: Debug Server`**: Starts the FastAPI backend on `http://localhost:8000` (which also serves the frontend files at root `/` during development).
   * **`Launch Client (Chrome)`**: Launches Chrome and attaches the debugger to the frontend application running on `http://localhost:8000`.
   * **`Server & Client (Chrome)`**: A compound launch configuration that runs both the FastAPI backend server and the Chrome client browser concurrently.
   * **`Python: Debug Pytest Suite`**: Runs the backend test suite in debug mode.

### Running Tests

To run the backend tests within the activated virtual environment, use the following commands:
* **Unit Tests (default)**: Runs the fast unit test suite (excludes e2e tests).
  ```bash
  PYTHONPATH=. venv/bin/python -m pytest
  ```
* **End-to-End Tests**: Runs the integration tests (requires a running Ollama server with the Gemma 4 model locally).
  ```bash
  PYTHONPATH=. venv/bin/python -m pytest -o "addopts=-m e2e" -s
  ```

---

## Technical Design Details

### Browser-side WAV Encoding
To ensure the backend container does not require installation of hefty audio libraries (like `ffmpeg`, `libsndfile`), the frontend captures audio from `navigator.mediaDevices.getUserMedia` via `AudioContext` and records raw float PCM buffers. On stop, it:
1. Downsamples the native sample rate (e.g. 48kHz or 44.1kHz) to exactly **16,000 Hz**.
2. Packs the float buffer into a **16-bit Signed Integer mono PCM WAV file** with standard RIFF headers.
3. Uploads the final file directly to `/api/process`.

### Multimodal Chat Endpoint
Rather than running separate transcription models, Siavox leverages Gemma 4's multimodal capabilities. The backend base64-encodes the WAV file and inserts it into the `images` payload list using the Ollama `/api/chat` schema, instructing the model to output a structured JSON format containing the verbatim transcription, detected source language, and the target Note/Message content.

# Siavox Project Knowledge Base

Welcome to the Siavox Knowledge Base. This directory serves as the source of truth for the Google Antigravity agent when reasoning about, modifying, or testing the Siavox codebase. Siavox is a speech-to-thought refinery application that downsamples client-side audio to 16kHz mono WAV, uploads it to a FastAPI/SQLite backend, and uses a local Ollama service running Gemma 4 (Effective 4B) with native audio capabilities to transcribe and refine vocal streams.

## Knowledge Base Index

The knowledge base is structured into the following modular context files:

1. **[Architecture and Data Flow](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/.gemini/antigravity/knowledge/architecture_and_data_flow.md)**: Details the network layout, container topology, security/authentication flow, and end-to-end data lifecycle.
2. **[Audio Pipeline and Downsampling](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/.gemini/antigravity/knowledge/audio_pipeline_and_downsampling.md)**: Focuses on browser-side audio capture, resampling formulas, and WAV format encoding.
3. **[AI Inference and Chunking](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/.gemini/antigravity/knowledge/ai_inference_and_chunking.md)**: Explains the multimodal chat API integrations, Gemma 4 prompting conventions, and the 30-second audio chunking-and-refinement algorithm.
4. **[Backend and Database](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/.gemini/antigravity/knowledge/backend_and_database.md)**: Outlines FastAPI routing, session handling, SQLite schema details, and lifespan event bindings.
5. **[Testing Guide](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/.gemini/antigravity/knowledge/testing_guide.md)**: Details the test architecture, database overriding strategies, pytest execution profiles, and network mocking techniques.

---

## Agent Guardrails: Knowledge Maintenance Rules

As an AI coding agent, you **MUST** adhere to the following rules to ensure the integrity of the project knowledge base:

### 1. The Knowledge-Code Sync Requirement
* **Rule**: Whenever you modify backend routing, database models, frontend recorder scripts, AI prompting mechanisms, or testing suites, you **MUST** update the corresponding `.gemini/antigravity/knowledge/` markdown file.
* **Timing**: This update must occur *simultaneously* with the code modifications, before completing the walkthrough or marking a task as done.
* **Verification**: Verify that the changes you made to the code are accurately reflected in the architectural descriptions, file links, and configuration properties detailed within this knowledge base.

### 2. Forbidden Practices
* **Do NOT** use generic placeholders (e.g., `<YOUR_API_KEY>`, `todo`) in knowledge documents. Provide exact environment keys (`GOOGLE_CLIENT_ID`, `OLLAMA_HOST`, etc.).
* **Do NOT** let documentation drift. If an endpoint payload schema changes in `main.py`, the changes must be applied to `backend_and_database.md` and `ai_inference_and_chunking.md`.
* **Do NOT** assume dependencies. Keep technical specifications limited to packages declared in `backend/requirements.txt` and native web browser APIs.

### 3. File Reference Formatting
* **Rule**: All file links in the knowledge base **MUST** be clickable, absolute paths using the `file:///` scheme and forward slashes.
  * *Example*: `[main.py](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/backend/app/main.py)`

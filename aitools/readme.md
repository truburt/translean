Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

# TransLean AI Tools

A Docker Compose configuration for launching provided `faster-whisper-server` (with `Systran/faster-whisper-large-v3` as the default model) and Ollama (with `translategemma:12b`) on an NVIDIA GPU. It builds the Whisper image from `../faster-whisper-server/Dockerfile.cuda`, downloads the other Docker images, starts the containers, and preloads the models. No GitHub PAT or access to a private Whisper image is required.

Note that the configuration uses `int8_float16` by default for the faster-whisper-server, which may not be supported by all GPUs. You can change the compute type in `docker-compose.yml` if needed for your GPU. Please also note that `translategemma:12b` is a fresh model and may introduce unexpected behavior.

Depending on the Translean server configuration, it may use other models for transcription and translation. In that case, they will be automatically downloaded and loaded by the respective services.

# Prerequisites:

- Docker Compose
- Windows or Linux with NVIDIA GPU support and NVIDIA Container Toolkit
- NVIDIA RTX GPU (>=16GB RAM required, RTX 4xxx or higher recommended)

# Usage:

From the repository root, run:

```bash
docker compose -f aitools/docker-compose.yml up -d --build
```

Or, from this directory:

```bash
docker compose up -d --build
```

Whisper listens on port `7860`; configure TransLean with
`WHISPER_BASE_URL=http://<host>:7860`, where `<host>` is reachable from the
API container. Ollama listens on port `11434`.

## Troubleshooting: GHCR unauthorized

If startup tries to pull `ghcr.io/truburt/faster-whisper-server:latest`, update
your checkout: older Compose configurations referenced that restricted image.
The current configuration builds Whisper locally instead. A successful
`docker login` or a PAT with `read:packages` does not grant access to a package
that your account is not permitted to read. Public source code and GHCR package
visibility are separate settings.

To rebuild and start just Whisper after updating:

```bash
docker compose -f aitools/docker-compose.yml up -d --build faster-whisper-server
```

Run this last command from the repository root. The first build needs internet
access to fetch public base images and dependencies; model downloads also need
internet access.

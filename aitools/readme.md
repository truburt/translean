Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

# TransLean AI Tools

A Docker Compose configuration for launching provided `faster-whisper-server` (with `Systran/faster-whisper-large-v3` as the default model) and Ollama (with `translategemma:12b`) on an NVIDIA GPU. It will automatically download the required Docker images, start the containers, wait for them to be ready, and then preload the models.

Note that the configuration uses `bfloat16` by default for the faster-whisper-server, which may not be supported by all GPUs. You can change this to `float16` or `int8_float16` in the `docker-compose.yml` file if needed. Please also note that `translategemma:12b` is a fresh model and may introduce unexpected behavior.

Depending on the Translean server configuration, it may use other models for transcription and translation. In that case, they will be automatically downloaded and loaded by the respective services.

# Prerequisites:

- Docker Compose
- Windows or Linux with NVIDIA GPU support
- NVIDIA RTX GPU (>=16GB RAM required, RTX 4xxx or higher recommended)

# Usage:

While the translean repo is private, you need to login to docker with a PAT (Personal Access Token) that has read:packages scope:
```bash
echo $CR_PAT | docker login ghcr.io -u <username> --password-stdin
```

Then, run:

```bash
docker compose up -d
```
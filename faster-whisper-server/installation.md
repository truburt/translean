# Faster-Whisper Server Installation

This guide covers installing and running the `faster-whisper-server` service locally or with Docker.

## Prerequisites
- Python 3.11+ (use `python3 -m venv .venv`)
- FFmpeg available on the system path (required by faster-whisper for decoding)
- Optional NVIDIA GPU with CUDA drivers installed (for `DEVICE=cuda`)

## Python setup (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Ubuntu 24.04 + CUDA setup (WSL 2 or host)
For Ubuntu 24.04 on WSL 2 or a native host, you can run:
```bash
./scripts/setup_cuda.sh
```
This installs CUDA + cuDNN via the NVIDIA repo, creates the local venv, and installs requirements.
On native Ubuntu hosts, it also installs the recommended NVIDIA driver if one is not detected.
The script selects the newest available `cuda-toolkit-12-*` package and installs cuDNN 9 for CUDA 12.

## Docker setup
From the `faster-whisper-server` directory:
```bash
docker compose up --build
```

### CUDA / NVIDIA GPU notes
- The container image in this repo is based on `python:3.11-slim`. For GPU acceleration you need:
  - NVIDIA drivers installed on the host.
  - NVIDIA Container Toolkit configured so Docker can mount GPU devices.
- If you need CUDA libraries baked into the image, use the provided CUDA Dockerfile and compose override:
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
  ```
  Enable GPU usage by setting:
  - `DEVICE=cuda`
  - `COMPUTE_TYPE=float16` (or another supported CUDA compute type)

## Model download and cache
The first transcription request (or startup preload) will download the model to the Hugging Face cache location. You can control where models are stored with environment variables:
- `HF_HOME` to change the Hugging Face cache root
- `HF_HUB_CACHE` to point directly at the hub cache directory

Example:
```bash
export HF_HOME=/data/hf
```

## Configuration
The server is configured entirely via environment variables:
- `MODEL_NAME` (default: `Systran/faster-whisper-large-v3`)
- `DEVICE` (default: `auto`)
- `COMPUTE_TYPE` (default: `auto`)
- `NUM_WORKERS` (default: `1`)
- `MODEL_TTL` (empty disables eviction)
- `MODEL_PRELOAD` (default: `false`)
- `LOG_LEVEL` (default: `INFO`)

## Model eviction
When `MODEL_TTL` is set, the server runs a background timer to evict expired models opportunistically
without forcing eviction on the next request.

## Upload format
Send a multipart form with:
- `file`: audio file to transcribe
- `params`: JSON object containing any `WhisperModel.transcribe(...)` arguments
- `model_name`: optional override for the model

Example:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F file=@sample.wav \
  -F params='{"language": "en", "beam_size": 5, "task": "transcribe"}'
```

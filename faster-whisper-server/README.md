# Faster-Whisper Server

A small FastAPI service that exposes `faster-whisper` transcription over HTTP.

## Requirements
- Python 3.11+
- FFmpeg available on your PATH

## Local setup (venv)
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
For Ubuntu 24.04 on WSL 2 or a native host, use:
```bash
./scripts/setup_cuda.sh
```
The script selects the newest available `cuda-toolkit-12-*` package and installs cuDNN 9 for CUDA 12.
On native Ubuntu it will install the recommended NVIDIA driver if one is not detected.

## Docker
```bash
docker compose up --build
```

For GPU acceleration, use the CUDA image override:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Configuration
All configuration uses environment variables:
- `MODEL_NAME` (default: `Systran/faster-whisper-large-v3`)
- `DEVICE` (default: `auto`)
- `COMPUTE_TYPE` (default: `auto`)
- `NUM_WORKERS` (default: `1`)
- `MODEL_TTL` (empty disables eviction)
- `MODEL_PRELOAD` (default: `false`)
- `LOG_LEVEL` (default: `INFO`)
- `BEAM_SIZE` (optional default for `params.beam_size`)

## Logging
Request logs include `upload_filename` and `content_type` in the structured `extra` fields.

## Model eviction
If `MODEL_TTL` is set, a background timer checks for expired models and evicts them opportunistically
without forcing eviction when a new request arrives.

## API usage
Send a multipart form with:
- `file`: audio file to transcribe
- `params`: JSON object containing any `WhisperModel.transcribe(...)` arguments
- `model_name`: optional override for the model

Example:
```bash
curl -X POST http://localhost:7860/transcribe \
  -F file=@sample.wav \
  -F params='{"language": "en", "beam_size": 5, "task": "transcribe"}'
```

Unsupported `params` keys are ignored to stay compatible with multiple faster-whisper versions.

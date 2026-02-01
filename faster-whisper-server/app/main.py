"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

FastAPI server exposing faster-whisper transcription endpoints.
"""

import inspect
import json
import logging
import os
import tempfile
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.model_manager import ModelManager

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("faster_whisper_server")

app = FastAPI(title="Faster Whisper Server", version="0.1.0")
model_manager = ModelManager(
    device=settings.device,
    compute_type=settings.compute_type,
    ttl_seconds=settings.model_ttl,
)


def _parse_params(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        params = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"params must be valid JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="params must be a JSON object")
    return params


def _apply_env_param_defaults(params: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "beam_size": settings.beam_size,
    }
    for key, value in defaults.items():
        if key not in params and value is not None:
            params[key] = value
    return params


def _filter_transcribe_params(model: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Filter params to the model.transcribe signature for cross-version compatibility."""
    try:
        signature = inspect.signature(model.transcribe)
    except (TypeError, ValueError):
        return params
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return params
    allowed = {name for name in signature.parameters if name != "self"}
    filtered = {key: value for key, value in params.items() if key in allowed}
    dropped = set(params) - set(filtered)
    if dropped:
        logger.info(f"Ignoring unsupported transcribe params: {sorted(dropped)}")
    return filtered


def _serialize_word(word: Any) -> dict[str, Any]:
    return {
        "start": getattr(word, "start", None),
        "end": getattr(word, "end", None),
        "word": getattr(word, "word", None),
        "probability": getattr(word, "probability", None),
    }


def _serialize_segment(segment: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": getattr(segment, "id", None),
        "seek": getattr(segment, "seek", None),
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "text": getattr(segment, "text", None),
        "tokens": getattr(segment, "tokens", None),
        "temperature": getattr(segment, "temperature", None),
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "compression_ratio": getattr(segment, "compression_ratio", None),
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
    }
    words = getattr(segment, "words", None)
    if words:
        data["words"] = [_serialize_word(word) for word in words]
    return data


def _serialize_info(info: Any) -> dict[str, Any]:
    return {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
        "all_language_probs": getattr(info, "all_language_probs", None),
    }


@app.on_event("startup")
async def startup_event() -> None:
    logger.info(f"Starting faster-whisper server: {settings.model_name} on {settings.device} with {settings.compute_type}")
    if settings.model_preload:
        model_manager.preload(settings.model_name)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    model_manager.shutdown()


@app.middleware("http")
async def request_logging_middleware(request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"Unhandled error: {request.url.path} {request.method}")
        raise
    duration_ms = (time.time() - start) * 1000
    logger.info(f"Request completed: {request.url.path} {request.method} {response.status_code} {round(duration_ms, 2)}ms")
    return response


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    params: str | None = Form(None),
    model_name: str | None = Form(None),
) -> JSONResponse:
    logger.info(f"Incoming transcription request: {file.filename} {file.content_type}")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        param_dict = _parse_params(params)
        _apply_env_param_defaults(param_dict)
        selected_model = model_name or settings.model_name
        model = model_manager.get_model(selected_model)
        filtered_params = _filter_transcribe_params(model, param_dict)
        segments, info = model.transcribe(temp_path, **filtered_params)
        serialized_segments = [_serialize_segment(segment) for segment in segments]
        response = {
            "model": selected_model,
            "text": "".join(segment.get("text") or "" for segment in serialized_segments),
            "segments": serialized_segments,
            "info": _serialize_info(info),
        }
        return JSONResponse(content=response)
    finally:
        os.remove(temp_path)

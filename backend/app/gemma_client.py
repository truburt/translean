"""
Gemma 4 E4B multimodal client.

This client powers the alternative audio pipeline where a single model performs
both speech transcription and translation from the same audio segment.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .runtime_config import runtime_config

logger = logging.getLogger(__name__)


def _normalize_source_language(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    return normalized or "auto"


async def warm_up() -> bool:
    """Warm up the Gemma service by hitting the health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{runtime_config.gemma_base_url.rstrip('/')}/health")
            response.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Gemma warm-up failed: %r", exc)
        return False


async def transcribe_and_translate_audio(
    audio_bytes: bytes,
    target_language: str,
    source_language: str = "auto",
    *,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Send one audio segment for transcription + translation.

    Expected response JSON:
      {
        "text": "...",
        "translation": "...",
        "language": "en"
      }
    """
    if not audio_bytes:
        return {"text": "", "translation": "", "language": "auto"}

    files = {
        "audio": ("segment.webm", audio_bytes, "audio/webm"),
    }
    data = {
        "model": runtime_config.gemma_model,
        "target_language": target_language,
        "source_language": _normalize_source_language(source_language),
        "keep_alive": str(runtime_config.gemma_keep_alive_seconds),
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{runtime_config.gemma_base_url.rstrip('/')}/transcribe-translate",
            data=data,
            files=files,
        )
        response.raise_for_status()
        payload = response.json()

    return {
        "text": (payload.get("text") or "").strip(),
        "translation": (payload.get("translation") or "").strip(),
        "language": (payload.get("language") or "auto").strip(),
    }

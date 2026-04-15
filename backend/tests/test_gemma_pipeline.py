import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.api import run_warm_up
from app.schemas import ServerConfig


@pytest.mark.asyncio
async def test_run_warm_up_gemma_sets_ready_events(monkeypatch):
    websocket = AsyncMock()
    whisper_ready = asyncio.Event()
    llm_ready = asyncio.Event()

    monkeypatch.setattr("app.api.runtime_config.pipeline_mode", "gemma4_e4b")
    monkeypatch.setattr("app.api.warm_up_gemma", AsyncMock(return_value=True))

    await run_warm_up(websocket, "auto", whisper_ready, llm_ready)

    assert whisper_ready.is_set()
    assert llm_ready.is_set()


def test_server_config_pipeline_mode_validation():
    payload = {
        "whisper_base_url": "http://localhost:8000",
        "whisper_model": "Systran/faster-whisper-large-v3",
        "whisper_keep_alive_seconds": 900,
        "ollama_base_url": "http://localhost:11434",
        "llm_model_translation": "translategemma:12b",
        "ollama_keep_alive_seconds": 900,
        "pipeline_mode": "gemma4_e4b",
        "gemma_base_url": "http://localhost:8010",
        "gemma_model": "google/gemma-4-E4B-it",
        "gemma_keep_alive_seconds": 900,
        "commit_timeout_seconds": 6.0,
        "silence_finalize_seconds": 1.4,
        "min_preview_buffer_seconds": 0.5,
        "stable_window_seconds": 5.0,
        "no_speech_prob_skip": 0.85,
        "no_speech_prob_logprob_skip": 0.6,
        "avg_logprob_skip": -1.0,
        "compression_ratio_skip": 2.4,
    }

    cfg = ServerConfig(**payload)
    assert cfg.pipeline_mode == "gemma4_e4b"

    payload["pipeline_mode"] = "invalid"
    with pytest.raises(ValidationError):
        ServerConfig(**payload)

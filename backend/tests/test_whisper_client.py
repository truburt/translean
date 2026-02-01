from typing import List
import json
import asyncio

import pytest

from app import whisper_client as wc
from app.config import settings


class MockResponse:
    def __init__(self, payload: dict):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class MockClient:
    def __init__(self, payloads: List[dict]):
        self.payloads = list(payloads)
        self.post_calls = []
        self.fallback = {"text": "", "language": "en", "segments": []}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        payload = self.payloads.pop(0) if self.payloads else self.fallback
        return MockResponse(payload)


class DummyTranscoder:
    def __init__(self, pcm_buffer: bytearray, full_buffer=None, on_pcm=None, **kwargs):
        self.pcm_buffer = pcm_buffer
        self.on_pcm = on_pcm

    async def start(self):
        return None

    async def write(self, data: bytes):
        self.pcm_buffer.extend(data)
        if self.on_pcm:
            self.on_pcm()

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_preview_and_stable_flow(monkeypatch):
    monkeypatch.setattr(settings, "whisper_base_url", "http://test-server")
    monkeypatch.setattr(settings, "whisper_model", "unit-test-model")
    monkeypatch.setattr(wc, "MIN_AUDIO_CHUNK_SIZE", 0)
    monkeypatch.setattr(wc, "MIN_PREVIEW_BUFFER_SECONDS", 0)
    # Set stable buffer to require 3 chunks or header+2chunks (7+7+7=21 bytes > 20 bytes)
    # 20 bytes / 32000 ~ 0.000625
    monkeypatch.setattr(wc, "MIN_STABLE_BUFFER_SECONDS", 0.0006)
    monkeypatch.setattr(wc, "PREVIEW_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(wc, "STABLE_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(wc, "SILENCE_FINALIZE_SECONDS", 0.2)
    monkeypatch.setattr(wc, "ContainerToPCMTranscoder", DummyTranscoder)

    header = b"\x1a\x45\xdf\xa3" + b"hdr"

    async def audio_gen():
        yield header
        yield b"chunk-a"
        await asyncio.sleep(0.01)
        yield b"chunk-b"

    payloads = [
        {"text": "hello", "language": "en", "segments": [{"start": 0.0, "end": 0.5, "text": "hello"}]},
        {
            "text": "hello world",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 0.8, "text": "hello "},
                {"start": 0.8, "end": 1.6, "text": "world"},
            ],
        },
        {"text": "", "language": "en", "segments": []},
    ]

    mock_client = MockClient(payloads)
    monkeypatch.setattr(wc.httpx, "AsyncClient", lambda **kwargs: mock_client)

    results = []
    async for item in wc.stream_transcription(audio_gen(), "en"):
        results.append(item)

    preview = [r for r in results if r.get("unstable_text")]
    finals = [r for r in results if r.get("is_final")]
    assert preview[0]["unstable_text"] == "hello"
    assert finals[0]["text"] == "hello world"
    assert results[-1]["status"] == "processing_complete"

    # Verify temperature schedule is used for stable calls and deterministic temp for preview
    for args, kwargs in mock_client.post_calls:
        params = json.loads(kwargs["data"]["params"])
        if params.get("beam_size") == 2:
            assert params["temperature"] == 0.0
        else:
            assert params["beam_size"] == 5
            assert params["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@pytest.mark.asyncio
async def test_filters_empty_and_no_speech(monkeypatch):
    monkeypatch.setattr(settings, "whisper_base_url", "http://test-server")
    monkeypatch.setattr(settings, "whisper_model", "unit-test-model")
    monkeypatch.setattr(wc, "MIN_AUDIO_CHUNK_SIZE", 0)
    monkeypatch.setattr(wc, "MIN_STABLE_BUFFER_SECONDS", 0)
    monkeypatch.setattr(wc, "MIN_PREVIEW_BUFFER_SECONDS", 100) # Disable preview to prevent payload stealing
    monkeypatch.setattr(wc, "PREVIEW_THROTTLE_SECONDS", 100)
    monkeypatch.setattr(wc, "STABLE_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(wc, "ContainerToPCMTranscoder", DummyTranscoder)

    header = b"\x1a\x45\xdf\xa3" + b"hdr"

    async def audio_gen():
        yield header
        yield b"chunk-a"
        await asyncio.sleep(0.01)
        yield b"chunk-b"

    payloads = [
        {
            "text": "silence speech",
            "language": "en",
            "segments": [
                {"avg_logprob": -1.5, "no_speech_prob": 0.9, "text": "silence "},
                {"avg_logprob": -0.2, "no_speech_prob": 0.1, "text": "speech"},
            ],
        },
        {
            "text": "garbled ok",
            "language": "en",
            "segments": [
                {"compression_ratio": 3.0, "text": "garbled "},
                {"compression_ratio": 1.1, "text": "ok"},
            ],
        },
        {"text": "", "language": "en", "segments": []},
    ]

    mock_client = MockClient(payloads)
    monkeypatch.setattr(wc.httpx, "AsyncClient", lambda **kwargs: mock_client)

    results = []
    async for item in wc.stream_transcription(audio_gen(), "en"):
        results.append(item)

    finals = [r for r in results if r.get("is_final")]
    assert all("silence" not in r.get("text", "") for r in finals)
    assert all("garbled" not in r.get("text", "") for r in finals)
    assert any(r.get("text") == "speech" for r in finals)
    assert any(r.get("text") == "ok" for r in finals)


@pytest.mark.asyncio
async def test_stop_recording_triggers_flush(monkeypatch):
    monkeypatch.setattr(settings, "whisper_base_url", "http://test-server")
    monkeypatch.setattr(settings, "whisper_model", "unit-test-model")
    monkeypatch.setattr(wc, "MIN_AUDIO_CHUNK_SIZE", 0)
    monkeypatch.setattr(wc, "MIN_STABLE_BUFFER_SECONDS", 0)
    monkeypatch.setattr(wc, "PREVIEW_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(wc, "STABLE_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(wc, "ContainerToPCMTranscoder", DummyTranscoder)

    header = b"\x1a\x45\xdf\xa3" + b"hdr"

    async def audio_gen():
        yield header
        yield b"chunk-a"
        yield {"type": "stop_recording"}

    payloads = [
        {"text": "hello there", "language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "hello there"}]},
    ]

    mock_client = MockClient(payloads)
    monkeypatch.setattr(wc.httpx, "AsyncClient", lambda **kwargs: mock_client)

    results = []
    async for item in wc.stream_transcription(audio_gen(), "en"):
        results.append(item)

    finals = [r for r in results if r.get("is_final")]
    assert any("hello there" in r.get("text", "") for r in finals)
    statuses = [r for r in results if r.get("status") == "processing_complete"]
    assert len(statuses) == 1
    assert statuses[0]["status"] == "processing_complete"


def test_normalize_recording_config_pcm():
    config = wc._normalize_recording_config("pcm16", "audio/pcm", 16000, 1)
    assert config.mode == "pcm"
    assert config.container_format is None
    assert config.mime_type == "audio/pcm"


@pytest.mark.asyncio
async def test_pcm_streaming_ingests_bytes(monkeypatch):
    monkeypatch.setattr(settings, "whisper_base_url", "http://test-server")
    monkeypatch.setattr(settings, "whisper_model", "unit-test-model")
    monkeypatch.setattr(wc, "MIN_AUDIO_CHUNK_SIZE", 0)
    monkeypatch.setattr(wc, "MIN_STABLE_BUFFER_SECONDS", 0)
    monkeypatch.setattr(wc, "PREVIEW_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(wc, "STABLE_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(wc, "MIN_PREVIEW_BUFFER_SECONDS", 100)

    async def audio_gen():
        yield b"\x01\x00\x02\x00"
        yield {"type": "stop_recording"}

    payloads = [
        {"text": "pcm ok", "language": "en", "segments": [{"start": 0.0, "end": 0.5, "text": "pcm ok"}]},
    ]

    mock_client = MockClient(payloads)
    monkeypatch.setattr(wc.httpx, "AsyncClient", lambda **kwargs: mock_client)

    results = []
    async for item in wc.stream_transcription(audio_gen(), "en", recording_format="pcm"):
        results.append(item)

    finals = [r for r in results if r.get("is_final")]
    assert any("pcm ok" in r.get("text", "") for r in finals)

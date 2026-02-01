import json
import os
import re
from pathlib import Path

import httpx
import pytest


SAMPLE_WEBM = Path(__file__).resolve().parents[1] / "tests" / "sample-30s.webm"
SAMPLE_TEXT = Path(__file__).resolve().parents[1] / "tests" / "sample-30s.txt"


def _normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@pytest.mark.e2e
def test_whisper_transcription_round_trip():
    base_url = os.getenv("E2E_WHISPER_BASE_URL", "http://localhost:7860")
    model_name = os.getenv("E2E_WHISPER_MODEL", "Systran/faster-whisper-large-v3")
    timeout = float(os.getenv("E2E_WHISPER_TIMEOUT", "120"))

    if not base_url or not model_name:
        pytest.skip("E2E Whisper base URL/model not configured.")

    audio_data = SAMPLE_WEBM.read_bytes()
    expected_text = SAMPLE_TEXT.read_text(encoding="utf-8")

    params = {"response_format": "json", "language": "en"}
    data = {"model_name": model_name, "params": json.dumps(params)}
    files = {"file": (SAMPLE_WEBM.name, audio_data, "audio/webm")}

    print(f"Transcribing {SAMPLE_WEBM.name} with {model_name}...")
    base_url = base_url.rstrip("/") + "/"
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        response = client.post("transcribe", data=data, files=files)

    response.raise_for_status()
    payload = response.json()
    transcript = payload.get("text", "")

    assert transcript

    expected_tokens = _normalize_tokens(expected_text)
    actual_tokens = _normalize_tokens(transcript)
    overlap = len(expected_tokens & actual_tokens) / max(len(expected_tokens), 1)

    assert len(actual_tokens) > 20
    assert overlap >= 0.6

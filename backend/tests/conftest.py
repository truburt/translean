"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _force_legacy_pipeline_mode_for_tests(monkeypatch):
    """Keep unit tests deterministic regardless of external PIPELINE_MODE env."""
    monkeypatch.setattr("app.api.runtime_config.pipeline_mode", "legacy_whisper_ollama")

"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Runtime configuration that merges environment defaults with DB overrides.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import settings


@dataclass
class RuntimeConfig:
    """Mutable runtime configuration values used by streaming and model clients."""

    whisper_base_url: str = settings.whisper_base_url
    whisper_model: str = settings.whisper_model
    whisper_keep_alive_seconds: int = settings.whisper_keep_alive_seconds
    ollama_base_url: str = settings.ollama_base_url
    llm_model_translation: str = settings.llm_model_translation
    ollama_keep_alive_seconds: int = settings.ollama_keep_alive_seconds
    commit_timeout_seconds: float = settings.commit_timeout_seconds
    silence_finalize_seconds: float = settings.silence_finalize_seconds
    min_preview_buffer_seconds: float = settings.min_preview_buffer_seconds
    stable_window_seconds: float = settings.stable_window_seconds
    no_speech_prob_skip: float = settings.no_speech_prob_skip
    no_speech_prob_logprob_skip: float = settings.no_speech_prob_logprob_skip
    avg_logprob_skip: float = settings.avg_logprob_skip
    compression_ratio_skip: float = settings.compression_ratio_skip

    def apply_overrides(self, overrides: dict) -> None:
        """Apply a dict of overrides onto this runtime config."""
        for key, value in overrides.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def as_dict(self) -> dict:
        """Return the runtime config as a JSON-serializable dict."""
        return asdict(self)


runtime_config = RuntimeConfig()

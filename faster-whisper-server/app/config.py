"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Configuration helpers for the faster-whisper server.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_name: str
    device: str
    compute_type: str
    log_level: str
    num_workers: int
    model_ttl: int | None
    model_preload: bool
    beam_size: int | None


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def get_settings() -> Settings:
    model_ttl = _parse_int(os.getenv("MODEL_TTL"))
    return Settings(
        model_name=os.getenv("MODEL_NAME", "Systran/faster-whisper-large-v3"),
        device=os.getenv("DEVICE", "auto"),
        compute_type=os.getenv("COMPUTE_TYPE", "auto"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        num_workers=int(os.getenv("NUM_WORKERS", "1")),
        model_ttl=model_ttl,
        model_preload=_parse_bool(os.getenv("MODEL_PRELOAD"), False),
        beam_size=_parse_int(os.getenv("BEAM_SIZE")),
    )

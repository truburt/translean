"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Model lifecycle management for faster-whisper.
"""

import logging
import threading
import time
import os
from dataclasses import dataclass
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class ModelHandle:
    model: WhisperModel
    model_name: str
    last_used: float


class ModelManager:
    def __init__(self, device: str, compute_type: str, ttl_seconds: Optional[int]) -> None:
        self._device = device
        self._compute_type = compute_type
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._handle: Optional[ModelHandle] = None
        self._stop_event = threading.Event()
        self._eviction_thread: Optional[threading.Thread] = None
        self._use_background_eviction = self._ttl_seconds is not None
        if self._ttl_seconds is not None:
            self._eviction_thread = threading.Thread(target=self._eviction_loop, daemon=True)
            self._eviction_thread.start()

    def _load_model(self, model_name: str) -> WhisperModel:
        logger.info(f"Loading faster-whisper model: {model_name} on {self._device} with {self._compute_type}")
        return WhisperModel(model_name, device=self._device, compute_type=self._compute_type)

    def get_model(self, model_name: str) -> WhisperModel:
        with self._lock:
            if self._handle is None or self._handle.model_name != model_name:
                model = self._load_model(model_name)
                self._handle = ModelHandle(model=model, model_name=model_name, last_used=time.time())
            else:
                self._handle.last_used = time.time()
            return self._handle.model

    def _evict_if_expired(self) -> None:
        if self._handle is None or self._ttl_seconds is None:
            return
        if time.time() - self._handle.last_used > self._ttl_seconds:
            logger.info(f"Evicting faster-whisper model due to TTL: {self._handle.model_name} {self._ttl_seconds}")
            self._handle = None

    def _eviction_loop(self) -> None:
        check_interval = max(1, min(self._ttl_seconds or 1, 60))
        while not self._stop_event.wait(check_interval):
            with self._lock:
                self._evict_if_expired()

    def preload(self, model_name: str) -> None:
        with self._lock:
            if self._handle is not None and self._handle.model_name == model_name:
                logger.info(f"Model already preloaded: {model_name}")
                return
            logger.info(f"Preloading faster-whisper model: {model_name}")
            model = self._load_model(model_name)
            self._handle = ModelHandle(model=model, model_name=model_name, last_used=time.time())

    def shutdown(self) -> None:
        if self._eviction_thread is None:
            return
        self._stop_event.set()
        self._eviction_thread.join(timeout=2)
